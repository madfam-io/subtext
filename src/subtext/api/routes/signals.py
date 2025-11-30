"""
Signals Routes

Information about available signal types from the Signal Atlas.
"""

from fastapi import APIRouter, Depends

from subtext.core.models import SignalType
from subtext.integrations.janua import TokenPayload, get_current_user
from subtext.pipeline.signals import SignalAtlas

router = APIRouter()


@router.get("/types")
async def list_signal_types(
    user: TokenPayload = Depends(get_current_user),
) -> list[dict]:
    """List all available signal types."""
    atlas = SignalAtlas()
    definitions = atlas.get_all_definitions()

    return [
        {
            "type": sig_type.value,
            "name": definition.name,
            "description": definition.description,
            "psychological_interpretation": definition.psychological_interpretation,
            "weight": definition.weight,
            "ui_color": definition.ui_color,
            "ui_icon": definition.ui_icon,
        }
        for sig_type, definition in definitions.items()
    ]


@router.get("/types/{signal_type}")
async def get_signal_type(
    signal_type: SignalType,
    user: TokenPayload = Depends(get_current_user),
) -> dict:
    """Get details for a specific signal type."""
    atlas = SignalAtlas()
    definition = atlas.get_definition(signal_type)

    return {
        "type": signal_type.value,
        "name": definition.name,
        "description": definition.description,
        "psychological_interpretation": definition.psychological_interpretation,
        "thresholds": definition.thresholds,
        "weight": definition.weight,
        "required_features": definition.required_features,
        "ui_color": definition.ui_color,
        "ui_icon": definition.ui_icon,
    }


@router.get("/categories")
async def list_signal_categories(
    user: TokenPayload = Depends(get_current_user),
) -> dict:
    """List signal types by category."""
    atlas = SignalAtlas()

    return {
        "temporal": [s.value for s in atlas.get_temporal_signals()],
        "spectral": [s.value for s in atlas.get_spectral_signals()],
        "composite": [s.value for s in atlas.get_composite_signals()],
    }
