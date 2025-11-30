# Subtext

**Metacognitive Engine - Read the room, not just the transcript.**

Subtext is an open-source Conversational Intelligence Infrastructure platform that analyzes audio to extract emotional signals, psychological states, and conversation dynamics.

## The Problem

90% of human communication is non-verbal. Current meeting tools (Otter, Teams Recap, Fireflies) capture only the text, missing crucial emotional cues, hesitation signals, and psychological states.

## The Solution

Subtext analyzes not just *what* is said, but *how* it's said - providing "EQ Augmentation" for remote teams.

## Features

- **Signal Detection** - Detect 16+ bio-acoustic signals (Truth Gap, Steamroll, Micro-Tremor, etc.)
- **Tension Timeline** - Visualize emotional dynamics over time
- **Speaker Analysis** - Engagement scores, stress indices, dominance patterns
- **AI Insights** - LLM-powered analysis of conversation dynamics
- **Real-time Streaming** - Live audio analysis with ESP (Emotional State Protocol)
- **Spatial Integration** - Native support for WorkAdventure and metaverse platforms

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Node.js 18+ (for frontend)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/madfam-io/subtext.git
cd subtext

# Start infrastructure
docker-compose up -d postgres redis minio

# Install dependencies
pip install -e ".[dev]"

# Run the API
uvicorn subtext.api:app --reload

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Using Enclii (Recommended)

```bash
# Initialize with Enclii
enclii init

# Deploy to development
enclii up

# Deploy to production
enclii deploy
```

## Architecture

```
Audio Input → Cleanse → Diarize → Transcribe → Prosodics → Signals → Insights
             (noise)   (who)      (text)       (features)  (detect)  (AI)
```

### Pipeline Stages

1. **Cleanse** - DeepFilterNet noise suppression
2. **Diarize** - Pyannote speaker identification
3. **Transcribe** - WhisperX speech-to-text
4. **Prosodics** - 47 acoustic feature extraction
5. **Signals** - Signal Atlas pattern detection
6. **Synthesize** - LLM insight generation

## Signal Atlas

The Signal Atlas maps bio-acoustic markers to psychological states:

| Signal | Detection | Interpretation |
|--------|-----------|----------------|
| Truth Gap | Response latency >800ms | Cognitive load, possible fabrication |
| Steamroll | Overlap >2s + volume spike | Dominance, aggression |
| Micro-Tremor | High jitter in vocals | Stress, deception marker |
| Monotone | Flat pitch >30s | Burnout, disengagement |
| Uptick | Rising declarative intonation | Insecurity, validation-seeking |

## Integrations

- **Auth**: Janua (https://github.com/madfam-io/janua)
- **DevOps**: Enclii (https://github.com/madfam-io/enclii)
- **Billing**: Stripe
- **Email**: Resend

## API

### Create Session

```bash
curl -X POST https://api.subtext.live/api/v1/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Team Standup"}'
```

### Upload Audio

```bash
curl -X POST https://api.subtext.live/api/v1/sessions/{id}/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@meeting.wav"
```

### Get Analysis

```bash
curl https://api.subtext.live/api/v1/sessions/{id}/signals \
  -H "Authorization: Bearer $TOKEN"
```

## Configuration

Environment variables:

```bash
# Application
APP_ENV=production
DEBUG=false

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/subtext

# Redis
REDIS_URL=redis://localhost:6379/0

# Auth (Janua)
JANUA_BASE_URL=https://auth.madfam.io
JANUA_CLIENT_ID=subtext
JANUA_CLIENT_SECRET=secret

# Billing (Stripe)
STRIPE_SECRET_KEY=sk_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Email (Resend)
RESEND_API_KEY=re_...

# ML Models
OPENAI_API_KEY=sk_...
```

## Documentation

- [Platform Architecture](./PLATFORM_ARCHITECTURE)
- [Technical Specification](./TECHNICAL_SPEC)
- [Signal Atlas](./SUBTEXT_SIGNAL_ATLAS)
- [Product Requirements](./PRD)

## License

AGPL-3.0 - See [LICENSE](./LICENSE) for details.

## Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.

## Support

- GitHub Issues: https://github.com/madfam-io/subtext/issues
- Email: support@subtext.live

---

**Built by [MADFAM](https://madfam.io)** - Innovaciones MADFAM SAS de CV
