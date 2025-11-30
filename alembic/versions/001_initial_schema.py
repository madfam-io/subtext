"""Initial schema for Subtext

Revision ID: 001_initial
Revises:
Create Date: 2025-01-01

Creates all core tables for the Subtext platform:
- Organizations (multi-tenant)
- Users
- Sessions
- Speakers
- Transcript segments
- Signals
- Prosodics (time-series)
- Insights
- Timeline
- Voice fingerprints
- API keys
- Usage records
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE sessionstatus AS ENUM ('pending', 'uploading', 'processing', 'completed', 'failed')")
    op.execute("CREATE TYPE signaltype AS ENUM ('truth_gap', 'steamroll', 'dead_air', 'micro_tremor', 'monotone', 'uptick', 'echo_chamber', 'coffee_shop', 'stress_spike', 'disengagement', 'deception_marker', 'enthusiasm_surge', 'agreement_signal', 'disagreement_signal')")
    op.execute("CREATE TYPE subscriptiontier AS ENUM ('free', 'personal', 'teams', 'enterprise', 'api')")
    op.execute("CREATE TYPE userrole AS ENUM ('owner', 'admin', 'member', 'viewer')")

    # Organizations table
    op.create_table(
        "organizations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False, unique=True),
        sa.Column("subscription_tier", postgresql.ENUM("free", "personal", "teams", "enterprise", "api", name="subscriptiontier", create_type=False), nullable=False, server_default="free"),
        sa.Column("stripe_customer_id", sa.String(255), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(255), nullable=True),
        sa.Column("settings", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("monthly_minutes_limit", sa.Integer, nullable=False, server_default="300"),
        sa.Column("monthly_minutes_used", sa.Integer, nullable=False, server_default="0"),
        sa.Column("api_calls_limit", sa.Integer, nullable=False, server_default="1000"),
        sa.Column("api_calls_used", sa.Integer, nullable=False, server_default="0"),
        sa.Column("features", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_organizations_slug", "organizations", ["slug"])
    op.create_index("ix_organizations_stripe_customer", "organizations", ["stripe_customer_id"])

    # Users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("role", postgresql.ENUM("owner", "admin", "member", "viewer", name="userrole", create_type=False), nullable=False, server_default="member"),
        sa.Column("janua_user_id", sa.String(255), nullable=True),
        sa.Column("avatar_url", sa.String(500), nullable=True),
        sa.Column("settings", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("voice_fingerprint_enabled", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("voice_fingerprint_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_unique_constraint("uq_users_org_email", "users", ["org_id", "email"])
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_janua_id", "users", ["janua_user_id"])

    # Sessions table
    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("status", postgresql.ENUM("pending", "uploading", "processing", "completed", "failed", name="sessionstatus", create_type=False), nullable=False, server_default="pending"),
        sa.Column("source_type", sa.String(50), nullable=False, server_default="upload"),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column("language", sa.String(10), nullable=True),
        sa.Column("storage_path", sa.String(500), nullable=True),
        sa.Column("speaker_count", sa.Integer, nullable=True),
        sa.Column("signal_count", sa.Integer, nullable=True),
        sa.Column("processing_time_ms", sa.Integer, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("asr_backend", sa.String(50), nullable=True),
        sa.Column("settings", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_sessions_org_id", "sessions", ["org_id"])
    op.create_index("ix_sessions_status", "sessions", ["status"])
    op.create_index("ix_sessions_created_at", "sessions", ["created_at"])

    # Speakers table
    op.create_table(
        "speakers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.String(100), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("fingerprint_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("embedding", postgresql.ARRAY(sa.Float), nullable=True),
        sa.Column("talk_time_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column("segment_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("talk_ratio", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("engagement_score", sa.Float, nullable=True),
        sa.Column("dominance_score", sa.Float, nullable=True),
        sa.Column("stress_index", sa.Float, nullable=True),
    )
    op.create_index("ix_speakers_session_id", "speakers", ["session_id"])
    op.create_index("ix_speakers_user_id", "speakers", ["user_id"])

    # Transcript segments table
    op.create_table(
        "transcript_segments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("speaker_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("speakers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("segment_index", sa.Integer, nullable=False),
        sa.Column("start_ms", sa.Integer, nullable=False),
        sa.Column("end_ms", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("words", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("is_question", sa.Boolean, nullable=False, server_default="false"),
    )
    op.create_index("ix_transcript_segments_session_id", "transcript_segments", ["session_id"])
    op.create_index("ix_transcript_segments_time", "transcript_segments", ["session_id", "start_ms"])

    # Signals table
    op.create_table(
        "signals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("speaker_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("speakers.id", ondelete="CASCADE"), nullable=True),
        sa.Column("signal_type", postgresql.ENUM("truth_gap", "steamroll", "dead_air", "micro_tremor", "monotone", "uptick", "echo_chamber", "coffee_shop", "stress_spike", "disengagement", "deception_marker", "enthusiasm_surge", "agreement_signal", "disagreement_signal", name="signaltype", create_type=False), nullable=False),
        sa.Column("timestamp_ms", sa.Integer, nullable=False),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("intensity", sa.Float, nullable=False),
        sa.Column("metrics", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("context", postgresql.JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_signals_session_id", "signals", ["session_id"])
    op.create_index("ix_signals_type", "signals", ["signal_type"])
    op.create_index("ix_signals_time", "signals", ["session_id", "timestamp_ms"])

    # Prosodics table (time-series)
    op.create_table(
        "prosodics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("speaker_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("speakers.id", ondelete="CASCADE"), nullable=True),
        sa.Column("timestamp_ms", sa.Integer, nullable=False),
        # Pitch features
        sa.Column("pitch_mean", sa.Float, nullable=True),
        sa.Column("pitch_std", sa.Float, nullable=True),
        sa.Column("pitch_range", sa.Float, nullable=True),
        sa.Column("pitch_slope", sa.Float, nullable=True),
        sa.Column("jitter", sa.Float, nullable=True),
        # Energy features
        sa.Column("energy_mean", sa.Float, nullable=True),
        sa.Column("energy_std", sa.Float, nullable=True),
        sa.Column("shimmer", sa.Float, nullable=True),
        # Temporal features
        sa.Column("speech_rate", sa.Float, nullable=True),
        sa.Column("silence_ratio", sa.Float, nullable=True),
        # Voice quality
        sa.Column("hnr", sa.Float, nullable=True),
        sa.Column("spectral_centroid", sa.Float, nullable=True),
        sa.Column("spectral_flux", sa.Float, nullable=True),
        # VAD scores
        sa.Column("valence", sa.Float, nullable=True),
        sa.Column("arousal", sa.Float, nullable=True),
        sa.Column("dominance", sa.Float, nullable=True),
    )
    op.create_index("ix_prosodics_session_time", "prosodics", ["session_id", "timestamp_ms"])

    # Convert prosodics to TimescaleDB hypertable (if TimescaleDB is available)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
                PERFORM create_hypertable('prosodics', 'timestamp_ms', chunk_time_interval => 60000, if_not_exists => TRUE);
            END IF;
        END $$;
    """)

    # Insights table
    op.create_table(
        "insights",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("insight_type", sa.String(50), nullable=False),
        sa.Column("content", postgresql.JSONB, nullable=False),
        sa.Column("importance", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_insights_session_id", "insights", ["session_id"])
    op.create_index("ix_insights_type", "insights", ["insight_type"])

    # Timeline table
    op.create_table(
        "timeline",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("timestamp_ms", sa.Integer, nullable=False),
        sa.Column("valence", sa.Float, nullable=False),
        sa.Column("arousal", sa.Float, nullable=False),
        sa.Column("tension_score", sa.Float, nullable=False),
        sa.Column("active_speaker_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("active_signals", postgresql.ARRAY(sa.String), nullable=False, server_default="{}"),
    )
    op.create_index("ix_timeline_session_time", "timeline", ["session_id", "timestamp_ms"])

    # Voice fingerprints table
    op.create_table(
        "voice_fingerprints",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("embedding", postgresql.ARRAY(sa.Float), nullable=False),
        sa.Column("sample_count", sa.Integer, nullable=False, server_default="1"),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("last_matched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_voice_fingerprints_org_id", "voice_fingerprints", ["org_id"])
    op.create_index("ix_voice_fingerprints_user_id", "voice_fingerprints", ["user_id"])

    # API keys table
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("key_prefix", sa.String(10), nullable=False),
        sa.Column("key_hash", sa.String(255), nullable=False),
        sa.Column("scopes", postgresql.ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("rate_limit", sa.Integer, nullable=False, server_default="100"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("usage_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_api_keys_org_id", "api_keys", ["org_id"])
    op.create_index("ix_api_keys_prefix", "api_keys", ["key_prefix"])

    # Usage records table
    op.create_table(
        "usage_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("usage_type", sa.String(50), nullable=False),
        sa.Column("quantity", sa.Float, nullable=False),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("stripe_reported", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("stripe_reported_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_usage_records_org_time", "usage_records", ["org_id", "recorded_at"])
    op.create_index("ix_usage_records_type", "usage_records", ["usage_type"])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("usage_records")
    op.drop_table("api_keys")
    op.drop_table("voice_fingerprints")
    op.drop_table("timeline")
    op.drop_table("insights")
    op.drop_table("prosodics")
    op.drop_table("signals")
    op.drop_table("transcript_segments")
    op.drop_table("speakers")
    op.drop_table("sessions")
    op.drop_table("users")
    op.drop_table("organizations")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS sessionstatus")
    op.execute("DROP TYPE IF EXISTS signaltype")
    op.execute("DROP TYPE IF EXISTS subscriptiontier")
    op.execute("DROP TYPE IF EXISTS userrole")
