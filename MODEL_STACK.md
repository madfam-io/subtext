# Subtext Model Stack
## Best-in-Class Open Source Models (2025)

This document details the ML model selections for each Subtext pipeline stage, with rationale, benchmarks, and alternatives.

---

## Pipeline Overview

```
Audio → VAD → Cleanse → Diarize → Transcribe → Emotion → Embed → Signals → LLM
        (1)    (2)       (3)        (4)          (5)       (6)      (7)      (8)
```

---

## 1. Voice Activity Detection (VAD)

### Primary: Silero VAD
| Metric | Value |
|--------|-------|
| **True Positive Rate** | 87.7% @ 5% FPR |
| **Latency** | <1ms per 30ms chunk |
| **Languages** | 6000+ |
| **License** | MIT |

**Why Silero VAD:**
- Deep learning-based multi-head attention architecture
- 87.7% TPR vs WebRTC VAD's 50% TPR at same false positive rate
- Trained on diverse domains (meetings, calls, podcasts)
- Runs on single CPU thread with ONNX acceleration
- Zero telemetry, no API keys required

**Links:**
- GitHub: https://github.com/snakers4/silero-vad
- PyTorch Hub: https://pytorch.org/hub/snakers4_silero-vad_vad/

**Alternatives:**
| Model | TPR @ 5% FPR | Use Case |
|-------|--------------|----------|
| WebRTC VAD | 50% | Legacy/simple baseline |
| Pyannote VAD | High | Requires GPU, not real-time |

---

## 2. Noise Suppression / Audio Enhancement

### Primary: DeepFilterNet v3
| Metric | Value |
|--------|-------|
| **Quality** | Studio-grade comparable to iZotope RX |
| **Parameters** | ~2M |
| **Latency** | Real-time on CPU |
| **Prosody Preservation** | Excellent |

**Why DeepFilterNet:**
- Two-stage deep filtering preserves emotional prosody (jitter, shimmer)
- Critical for emotion detection - other denoisers strip emotional cues
- Intel OpenVINO version available for Audacity integration
- Active development with embedded-optimized variants

**Links:**
- GitHub: https://github.com/Rikorose/DeepFilterNet
- HuggingFace: https://huggingface.co/Intel/deepfilternet-openvino

**Alternatives:**
| Model | Best For | Trade-off |
|-------|----------|-----------|
| GT-CRN | Low compute | Newer, less tested |
| ULCNet | Edge/embedded | 0.127 RTF on Cortex-A53 |
| Resemble-Enhance | Complex noise | Higher compute |

---

## 3. Speaker Diarization

### Primary: Pyannote 4.0 Community-1
| Metric | Value |
|--------|-------|
| **Diarization Error Rate** | ~10% |
| **Real-time Factor** | 2.5% on GPU |
| **Fine-tuning** | Supported |
| **License** | MIT |

**Why Pyannote:**
- Best open-source diarization solution (2025 benchmarks)
- Supports fine-tuning on custom data
- Handles overlapping speech and rapid turn-taking
- Integrates with WhisperX for aligned transcripts

**Links:**
- GitHub: https://github.com/pyannote/pyannote-audio
- Blog: https://www.pyannote.ai/blog/community-1

### Alternative: NVIDIA NeMo Sortformer
| Metric | Value |
|--------|-------|
| **Architecture** | End-to-end 18-layer Transformer |
| **Best For** | Production scale with NVIDIA GPUs |
| **Fine-tuning** | Not supported |

**When to use NeMo:**
- High-volume production with NVIDIA hardware
- Monologue-heavy content (podcasts, lectures)
- Integrated ASR+diarization pipeline needed

**Links:**
- Docs: https://docs.nvidia.com/nemo-framework/
- Model: `nvidia/diar_sortformer_4spk-v1`

---

## 4. Speech-to-Text (ASR)

### Option A: Multilingual (Default)
### OpenAI Whisper Large-V3
| Metric | Value |
|--------|-------|
| **Languages** | 40+ |
| **WER (English)** | ~8-10% |
| **License** | MIT |

**Best for:** International content, multiple languages

### Option B: Maximum Accuracy
### NVIDIA Canary Qwen 2.5B
| Metric | Value |
|--------|-------|
| **WER** | 5.63% (SOTA on Open ASR Leaderboard) |
| **Architecture** | FastConformer + Qwen3-1.7B decoder |
| **Capabilities** | Transcription + summarization + Q&A |

**Best for:** English-only content requiring maximum accuracy

**Links:**
- Blog: https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/

### Option C: Maximum Speed
### NVIDIA Parakeet TDT 1.1B
| Metric | Value |
|--------|-------|
| **RTFx** | 2000+ (fastest on Open ASR) |
| **Throughput** | 1 hour audio in ~19 seconds (M4 Pro) |
| **Languages** | 25 European (FluidAudio variant) |

**Best for:** Batch processing backlogs, real-time streaming

**Comparison:**
| Model | WER | Speed (RTFx) | Languages |
|-------|-----|--------------|-----------|
| Canary 2.5B | 5.63% | ~100 | English |
| Parakeet TDT | ~7% | 2000+ | 25 |
| Whisper V3 | ~8% | ~50 | 40+ |
| Granite 8B | 5.85% | ~30 | 5 |

---

## 5. Speech Emotion Recognition (SER)

### Primary: Emotion2Vec+
| Metric | Value |
|--------|-------|
| **Benchmark** | SOTA on 9 multilingual datasets |
| **Architecture** | Self-supervised pre-training with utterance + frame objectives |
| **Emotions** | 9-class (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown) |

**Why Emotion2Vec:**
- **Purpose-built for emotion** (not fine-tuned speech model)
- Superior cross-lingual generalization
- State-of-the-art on WA, UA, WF1 metrics across languages
- Available in base, large variants

**Links:**
- Paper: https://aclanthology.org/2024.findings-acl.931.pdf
- HuggingFace: `iic/emotion2vec_plus_large`

**Model Variants:**
| Model | Classes | Size |
|-------|---------|------|
| emotion2vec_plus_large | 9 | Large |
| emotion2vec_plus_base | 4 | Medium |
| emotion2vec_base_finetuned | 9 | Base |

### Alternative: WavLM / Whisper (for noisy speech)

From Interspeech 2025 SER Challenge:
| Model | Macro F1 | Best For |
|-------|----------|----------|
| Whisper | 0.366 | Diverse/noisy speech |
| WavLM | 0.34 | Spontaneous speech |
| Wav2Vec2 | 0.28 | Clean speech only |
| HuBERT | 0.27 | Clean speech only |

**Key Insight:** WavLM and Whisper outperform Wav2Vec2/HuBERT on naturalistic audio because they were pretrained on diverse, spontaneous, noisy speech.

### Lightweight: DistilHuBERT
| Metric | Value |
|--------|-------|
| **Accuracy** | 70.64% |
| **Size** | 0.02 MB (75% compressed) |
| **Best For** | Edge devices, real-time |

---

## 6. Speaker Embedding (Voice Fingerprinting)

### Primary: ECAPA-TDNN
| Metric | Value |
|--------|-------|
| **EER** | 1.71% (lowest) |
| **Inference** | 69.43ms |
| **Training Data** | VoxCeleb1 + VoxCeleb2 |

**Why ECAPA-TDNN:**
- Lowest Equal Error Rate among open-source models
- Squeeze-excitation blocks + channel attention + multi-scale aggregation
- Validated for forensic speaker recognition
- Excellent balance of accuracy and speed

**Links:**
- HuggingFace: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- GitHub: https://github.com/TaoRuijie/ECAPA-TDNN

### Alternative: TitaNet (NVIDIA NeMo)
| Metric | Value |
|--------|-------|
| **EER** | 1.91% |
| **Architecture** | ContextNet + SE layers |
| **Best For** | NVIDIA production environments |

**Links:**
- Docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_recognition/models.html

**Comparison:**
| Model | EER | Latency | Framework |
|-------|-----|---------|-----------|
| ECAPA-TDNN | 1.71% | 69ms | SpeechBrain |
| TitaNet | 1.91% | ~80ms | NeMo |
| ReDimNet | ~2% | - | Research |

---

## 7. Signal Detection

No external models - uses **Signal Atlas** with rule-based detection on extracted features.

See: `SUBTEXT_SIGNAL_ATLAS` for signal definitions.

---

## 8. LLM (Synthesis & Insights)

### Cloud Options (Default)
| Provider | Model | Best For |
|----------|-------|----------|
| OpenAI | GPT-4 Turbo | General analysis |
| Anthropic | Claude 3.5 Sonnet | Nuanced insights |

### Self-Hosted Open Source

#### Maximum Quality: Llama 3.1 70B Instruct
| Metric | Value |
|--------|-------|
| **Parameters** | 70B |
| **Context** | 128K tokens |
| **License** | Llama 3.1 Community |
| **VRAM Required** | ~140GB (FP16) or 40GB (INT4) |

**Best for:** Maximum insight quality, privacy-critical deployments

#### Balanced: Mixtral 8x22B
| Metric | Value |
|--------|-------|
| **Architecture** | MoE (Mixture of Experts) |
| **Active Parameters** | ~39B per token |
| **License** | Apache 2.0 |

**Best for:** Good quality with lower compute than Llama 70B

#### Fast: Llama 3.1 8B Instruct
| Metric | Value |
|--------|-------|
| **Parameters** | 8B |
| **VRAM Required** | ~16GB (FP16) or 6GB (INT4) |

**Best for:** Real-time streaming, edge deployment

**Deployment Options:**
- **Ollama:** Easy local setup
- **vLLM:** Production-grade serving with PagedAttention
- **TGI:** HuggingFace Text Generation Inference

---

## Hardware Recommendations

### Minimum (Development)
- CPU: 8 cores
- RAM: 32GB
- GPU: RTX 3080 (10GB) or equivalent
- Storage: 100GB SSD

### Production (Per Worker)
- CPU: 16+ cores
- RAM: 64GB
- GPU: A100 40GB or RTX 4090 24GB
- Storage: 500GB NVMe

### High-Volume Production
- GPU Cluster: 4x A100 80GB
- Separate inference servers for each model type
- Redis for job queue distribution

---

## Model Download Sizes

| Model | Size | Notes |
|-------|------|-------|
| Silero VAD | ~5MB | Tiny |
| DeepFilterNet | ~15MB | Small |
| Pyannote 3.1 | ~200MB | Requires HuggingFace token |
| Whisper Large-V3 | ~3GB | Largest Whisper |
| Canary 2.5B | ~5GB | NVIDIA NeMo required |
| Emotion2Vec Large | ~1.5GB | HuggingFace |
| ECAPA-TDNN | ~85MB | SpeechBrain |
| Llama 3.1 70B | ~140GB | Quantized: ~40GB |

---

## License Summary

| Model | License | Commercial Use |
|-------|---------|----------------|
| Silero VAD | MIT | Yes |
| DeepFilterNet | MIT | Yes |
| Pyannote | MIT | Yes |
| Whisper | MIT | Yes |
| NVIDIA Models | CC-BY-4.0 / Apache | Check each |
| Emotion2Vec | Apache 2.0 | Yes |
| ECAPA-TDNN | Apache 2.0 | Yes |
| Llama 3.1 | Llama Community | Yes (with terms) |
| Mixtral | Apache 2.0 | Yes |

---

## References

1. [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
2. [DeepFilterNet GitHub](https://github.com/Rikorose/DeepFilterNet)
3. [Pyannote Blog](https://www.pyannote.ai/blog/community-1)
4. [NVIDIA Canary Blog](https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/)
5. [Emotion2Vec Paper](https://aclanthology.org/2024.findings-acl.931.pdf)
6. [ECAPA-TDNN HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
7. [Open ASR Leaderboard](https://huggingface.co/spaces/open-asr-leaderboard/open-asr-leaderboard)
8. [Interspeech 2025 SER Challenge](https://github.com/alefiury/InterSpeech-SER-2025)

---

*Last Updated: November 2025*
