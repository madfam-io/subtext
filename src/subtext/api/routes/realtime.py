"""
Realtime WebSocket Routes

Handles WebSocket connections for real-time audio streaming and ESP broadcasting.
"""

import asyncio
import base64
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
from fastapi.websockets import WebSocketState

from subtext.core.models import ESPMessage
from subtext.realtime import (
    ConnectionManager,
    RealtimeConnection,
    RealtimeProcessor,
    AudioChunk,
    ESPBroadcaster,
)
from subtext.realtime.protocol import (
    RealtimeMessage,
    RealtimeMessageType,
    SessionConfig,
    SessionStartPayload,
    SessionCreatedPayload,
    AudioChunkPayload,
    ErrorPayload,
)
from subtext.realtime.connection import get_connection_manager
from subtext.realtime.broadcaster import get_esp_broadcaster

logger = structlog.get_logger()

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Dependencies
# ══════════════════════════════════════════════════════════════


async def authenticate_websocket(
    websocket: WebSocket,
    token: str | None = Query(None),
) -> tuple[UUID | None, UUID | None]:
    """
    Authenticate WebSocket connection.

    Returns (user_id, org_id) tuple.
    For now, allows anonymous connections for development.
    """
    # TODO: Integrate with Janua authentication
    # In production, validate the token and extract user/org IDs

    if token:
        # Placeholder for token validation
        # Would call Janua API to validate token
        pass

    # For development, allow anonymous connections
    return None, None


# ══════════════════════════════════════════════════════════════
# WebSocket Endpoint
# ══════════════════════════════════════════════════════════════


@router.websocket("/realtime")
async def realtime_websocket(
    websocket: WebSocket,
    token: str | None = Query(None),
    manager: ConnectionManager = Depends(get_connection_manager),
    broadcaster: ESPBroadcaster = Depends(get_esp_broadcaster),
):
    """
    Real-time audio streaming WebSocket endpoint.

    Protocol:
    1. Client connects to /ws/realtime?token=<auth_token>
    2. Client sends session.start with configuration
    3. Server responds with session.created
    4. Client streams audio.chunk messages
    5. Server sends transcript, signal, esp updates
    6. Client sends session.end or disconnects
    """
    # Authenticate
    user_id, org_id = await authenticate_websocket(websocket, token)

    # Accept connection
    connection = await manager.connect(websocket, user_id, org_id)

    # Processor instance (created when session starts)
    processor: RealtimeProcessor | None = None

    try:
        while True:
            # Receive message
            try:
                raw_data = await websocket.receive()

                if raw_data.get("type") == "websocket.disconnect":
                    break

                # Handle both text and binary messages
                if "text" in raw_data:
                    data = await _parse_json_message(raw_data["text"])
                elif "bytes" in raw_data:
                    # Binary audio data - wrap in message
                    data = {
                        "type": RealtimeMessageType.AUDIO_CHUNK.value,
                        "payload": {
                            "data": base64.b64encode(raw_data["bytes"]).decode(),
                            "timestamp_ms": connection.total_duration_ms,
                        },
                    }
                else:
                    continue

            except Exception as e:
                logger.error(f"Failed to receive message: {e}")
                continue

            # Parse message type
            msg_type = data.get("type")
            payload = data.get("payload", {})

            # ─────────────────────────────────────────────────
            # Handle message types
            # ─────────────────────────────────────────────────

            if msg_type == RealtimeMessageType.SESSION_START.value:
                # Start a new session
                processor = await _handle_session_start(
                    connection,
                    payload,
                    manager,
                    broadcaster,
                )

            elif msg_type == RealtimeMessageType.AUDIO_CHUNK.value:
                # Process audio chunk
                if processor:
                    await _handle_audio_chunk(
                        connection,
                        processor,
                        payload,
                        broadcaster,
                    )
                else:
                    await connection.send_error(
                        "session_not_started",
                        "Session must be started before sending audio",
                    )

            elif msg_type == RealtimeMessageType.AUDIO_END.value:
                # End of audio stream (but session continues)
                if processor:
                    connection.is_streaming = False
                    logger.info(
                        "Audio stream ended",
                        session_id=str(connection.session_id),
                    )

            elif msg_type == RealtimeMessageType.SESSION_END.value:
                # End session
                if processor:
                    await _handle_session_end(
                        connection,
                        processor,
                        manager,
                        broadcaster,
                    )
                    processor = None
                break

            elif msg_type == RealtimeMessageType.CONFIG_UPDATE.value:
                # Update session configuration
                if processor:
                    await _handle_config_update(connection, processor, payload)

            elif msg_type == RealtimeMessageType.PING.value:
                # Respond to ping
                await connection.send_message(
                    RealtimeMessage(
                        type=RealtimeMessageType.PONG,
                        session_id=connection.session_id,
                    )
                )

            else:
                logger.warning(f"Unknown message type: {msg_type}")

            # Update activity timestamp
            connection.update_activity()

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected",
            connection_id=str(connection.connection_id),
        )

    except Exception as e:
        logger.error(
            "WebSocket error",
            connection_id=str(connection.connection_id),
            error=str(e),
        )

    finally:
        # Cleanup
        if processor:
            await processor.finalize()

        if connection.session_id:
            await broadcaster.close_channel(connection.session_id)

        broadcaster.unregister_send_callback(connection.connection_id)
        await manager.disconnect(connection)


async def _parse_json_message(text: str) -> dict[str, Any]:
    """Parse JSON message from text."""
    import orjson

    return orjson.loads(text)


async def _handle_session_start(
    connection: RealtimeConnection,
    payload: dict[str, Any],
    manager: ConnectionManager,
    broadcaster: ESPBroadcaster,
) -> RealtimeProcessor:
    """Handle session.start message."""
    # Parse configuration
    start_payload = SessionStartPayload(**payload)
    config = start_payload.config

    # Generate session ID
    session_id = uuid4()

    # Create processor
    processor = RealtimeProcessor(session_id, config)
    await processor.initialize()

    # Register session
    await manager.register_session(connection, session_id, config)

    # Create ESP channel
    await broadcaster.create_channel(
        session_id=session_id,
        owner_id=connection.user_id,
        org_id=connection.org_id,
        consent_level=config.esp_consent_level,
        rate_limit_ms=config.esp_interval_ms,
    )

    # Register callback for ESP broadcasts
    broadcaster.register_send_callback(
        connection.connection_id,
        connection.send_message,
    )

    # Subscribe self to ESP
    await broadcaster.subscribe(
        subscriber_id=connection.connection_id,
        session_id=session_id,
        subscriber_org_id=connection.org_id,
        subscriber_user_id=connection.user_id,
    )

    # Send session.created response
    await connection.send_message(
        RealtimeMessage(
            type=RealtimeMessageType.SESSION_CREATED,
            session_id=session_id,
            payload=SessionCreatedPayload(
                session_id=session_id,
                audio_config=config.audio,
                features_enabled={
                    "transcription": config.enable_transcription,
                    "diarization": config.enable_diarization,
                    "emotion": config.enable_emotion,
                    "prosodics": config.enable_prosodics,
                    "signals": config.enable_signals,
                    "esp": config.esp_enabled,
                },
            ).model_dump(mode="json"),
        )
    )

    connection.is_streaming = True

    logger.info(
        "Realtime session started",
        session_id=str(session_id),
        connection_id=str(connection.connection_id),
        config={
            "asr_backend": config.asr_backend,
            "esp_enabled": config.esp_enabled,
        },
    )

    return processor


async def _handle_audio_chunk(
    connection: RealtimeConnection,
    processor: RealtimeProcessor,
    payload: dict[str, Any],
    broadcaster: ESPBroadcaster,
) -> None:
    """Handle audio.chunk message."""
    try:
        # Parse chunk payload
        chunk_payload = AudioChunkPayload(**payload)

        # Decode audio data
        if isinstance(chunk_payload.data, str):
            chunk = AudioChunk.from_base64(
                chunk_payload.data,
                chunk_payload.timestamp_ms,
                connection.config.audio,
            )
        else:
            chunk = AudioChunk.from_bytes(
                chunk_payload.data,
                chunk_payload.timestamp_ms,
                connection.config.audio,
            )

        # Update connection state
        connection.samples_received += len(chunk.data)
        connection.total_duration_ms = chunk_payload.timestamp_ms + chunk.duration_ms

        # Process chunk and send results
        async for message in processor.process_chunk(chunk):
            await connection.send_message(message)

            # If it's an ESP update, also broadcast via broadcaster
            if message.type == RealtimeMessageType.ESP_UPDATE:
                esp = ESPMessage(**message.payload)
                await broadcaster.broadcast(connection.session_id, esp)

    except Exception as e:
        logger.error(
            "Failed to process audio chunk",
            session_id=str(connection.session_id),
            error=str(e),
        )
        await connection.send_error(
            "processing_error",
            f"Failed to process audio: {str(e)}",
            recoverable=True,
        )


async def _handle_session_end(
    connection: RealtimeConnection,
    processor: RealtimeProcessor,
    manager: ConnectionManager,
    broadcaster: ESPBroadcaster,
) -> None:
    """Handle session.end message."""
    # Finalize processor
    summary = await processor.finalize()

    # Close ESP channel
    await broadcaster.close_channel(connection.session_id)

    # Send session closed message
    await connection.send_message(
        RealtimeMessage(
            type=RealtimeMessageType.SESSION_CLOSED,
            session_id=connection.session_id,
            payload={
                "summary": summary,
                "duration_ms": connection.total_duration_ms,
            },
        )
    )

    logger.info(
        "Realtime session ended",
        session_id=str(connection.session_id),
        duration_ms=connection.total_duration_ms,
        speaker_count=summary.get("speaker_count", 0),
    )


async def _handle_config_update(
    connection: RealtimeConnection,
    processor: RealtimeProcessor,
    payload: dict[str, Any],
) -> None:
    """Handle config.update message."""
    # Update specific configuration fields
    for key, value in payload.items():
        if hasattr(connection.config, key):
            setattr(connection.config, key, value)
            setattr(processor.config, key, value)

    logger.info(
        "Session config updated",
        session_id=str(connection.session_id),
        updates=list(payload.keys()),
    )


# ══════════════════════════════════════════════════════════════
# ESP Subscription Endpoint
# ══════════════════════════════════════════════════════════════


@router.websocket("/esp/subscribe/{session_id}")
async def esp_subscribe_websocket(
    websocket: WebSocket,
    session_id: UUID,
    token: str | None = Query(None),
    broadcaster: ESPBroadcaster = Depends(get_esp_broadcaster),
):
    """
    Subscribe to ESP updates from a specific session.

    This is a read-only WebSocket that receives ESP broadcasts.
    """
    # Authenticate
    user_id, org_id = await authenticate_websocket(websocket, token)

    # Accept connection
    await websocket.accept()

    subscriber_id = uuid4()

    # Register send callback
    async def send_callback(message: RealtimeMessage):
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(message.model_dump(mode="json"))

    broadcaster.register_send_callback(subscriber_id, send_callback)

    # Subscribe to session
    success = await broadcaster.subscribe(
        subscriber_id=subscriber_id,
        session_id=session_id,
        subscriber_org_id=org_id,
        subscriber_user_id=user_id,
    )

    if not success:
        await websocket.send_json({
            "type": "error",
            "payload": {
                "code": "subscription_denied",
                "message": "Cannot subscribe to this session",
            },
        })
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Send subscription confirmed
    await websocket.send_json({
        "type": "subscription.confirmed",
        "payload": {
            "session_id": str(session_id),
            "subscriber_id": str(subscriber_id),
        },
    })

    try:
        # Keep connection alive, handle pings
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            elif data.get("type") == "unsubscribe":
                break

    except WebSocketDisconnect:
        pass

    finally:
        await broadcaster.unsubscribe(subscriber_id, session_id)
        broadcaster.unregister_send_callback(subscriber_id)


# ══════════════════════════════════════════════════════════════
# HTTP Endpoints for Connection Info
# ══════════════════════════════════════════════════════════════


@router.get("/connections")
async def get_connections(
    manager: ConnectionManager = Depends(get_connection_manager),
) -> dict[str, Any]:
    """Get connection manager statistics."""
    return manager.get_stats()


@router.get("/esp/stats")
async def get_esp_stats(
    broadcaster: ESPBroadcaster = Depends(get_esp_broadcaster),
) -> dict[str, Any]:
    """Get ESP broadcaster statistics."""
    return broadcaster.get_stats()
