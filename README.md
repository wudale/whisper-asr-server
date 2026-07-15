# 🎙️ Whisper ASR Service — Self-Hosted Speech-to-Text

> One command to deploy. Auto-detects hardware. OpenAI-compatible API.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)]()
[![Whisper](https://img.shields.io/badge/Whisper-faster--whisper%20%7C%20mlx--whisper-orange.svg)]()

## 🔍 What is this?

**Whisper ASR Service** is a one-command-deploy speech recognition web service that wraps OpenAI Whisper models into a standard REST API. Run your own private STT service on any Linux server or Mac — even a Raspberry Pi.

**Use cases:**
- Transcribe internal meeting recordings without sending audio to third-party clouds
- Batch-convert podcasts, lectures, or interviews to text at zero per-minute cost
- Run a private STT backend on your NAS, VPS, or Mac Studio
- Feed audio to local LLM agents via a standard API

## ✨ Highlights

| Feature | Why it matters |
|---------|---------------|
| 🚀 **One-command install** | `sudo ./ops.sh install` — deps, model download, systemd/launchd, all in one go |
| 🧠 **Hardware-aware** | Detects CPU/GPU/RAM, auto-picks the best model (tiny→large-v3) |
| 📡 **OpenAI-compatible** | `POST /v1/audio/transcriptions` — drop-in replacement, reuse existing SDK code |
| 🔒 **Privacy-first** | Fully local inference, zero network dependency after initial model download |
| 🖥️ **Cross-platform** | Linux (CPU/GPU) + macOS Apple Silicon |
| 👀 **Dry-run mode** | `--dry-run` shows the plan before executing — beginner-friendly |
| 🌍 **99+ languages** | Auto-detection for Chinese, English, Japanese, Korean, French, German, Spanish... |
| 📦 **Zero secret leaks** | No hardcoded passwords, IPs, or API keys |

## 🎯 Who is this for?

- **Teams** — transcribe meetings privately, no data leaves your network
- **Content creators** — batch-generate SRT subtitles for courses, interviews, demos
- **Podcasters** — RSS audio → text archive, fully automated
- **Developers** — local STT backend for voice assistants, chatbots, AI agents
- **Language learners** — compare pronunciation against reference text
- **Homelab enthusiasts** — run your own Speech-to-Text API on a Raspberry Pi or NAS

## ⚡ Quick Start

```bash
git clone https://github.com/<your-username>/whisper-asr-server.git
cd whisper-asr-server

# Preview the installation plan (zero side effects)
sudo ./ops.sh install --dry-run

# Run it
sudo ./ops.sh install

# Open the demo page
open http://localhost:9080
```

For macOS or user-level install (no root):
```bash
./ops.sh install --user
```

## 📊 vs. Alternatives

| Solution | Privacy | Cost | Setup | Hardware | Multilingual |
|----------|:-------:|:----:|:-----:|:--------:|:------------:|
| **Whisper ASR Service (this project)** 🏆 | ✅ Local | Free | ⭐ One command | CPU/GPU auto-detect | ✅ 99+ |
| OpenAI Whisper API | ❌ Cloud | $0.006/min | ⭐ | None | ✅ 99+ |
| Google/AWS/Azure STT | ❌ Cloud | Pay-per-use | ⭐⭐ | None | ⚠️ Limited |
| vanilla openai-whisper | ✅ Local | Free | ⭐⭐⭐ | GPU recommended | ✅ 99+ |
| whisper.cpp | ✅ Local | Free | ⭐⭐⭐ | CPU | ✅ 99+ |
| bare faster-whisper | ✅ Local | Free | ⭐⭐⭐ | GPU/CPU | ✅ 99+ |

## 🏗️ Project Structure

```
whisper-asr-server/
├── whisper_server.py          # FastAPI service
├── ops.sh                     # Ops script (install, start/stop, model switching)
├── whisper-asr-server.service     # systemd unit template
├── LICENSE                    # MIT
└── README.md
```

## 📡 API

OpenAI Audio Transcription API compatible.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/` | Interactive demo page |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio |

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `file` | file | ✅ | Audio/video (mp3, wav, m4a, ogg, flac, mp4, mov, etc.) |
| `language` | string | ❌ | Language code; omit for auto-detection (99+ languages) |
| `response_format` | string | ❌ | `json` (default), `text`, `srt`, `verbose_json` |

### Examples

**Shell**
```bash
curl -X POST http://<your-server>:9080/v1/audio/transcriptions \
  -F "file=@recording.mp3"
```

**Python**
```python
import requests
with open("recording.mp3", "rb") as f:
    r = requests.post("http://<your-server>:9080/v1/audio/transcriptions",
                      files={"file": f})
print(r.json()["text"])
```

**JavaScript**
```javascript
const fd = new FormData(); fd.append("file", fileInput.files[0]);
const r = await fetch("http://<your-server>:9080/v1/audio/transcriptions",
                       { method: "POST", body: fd });
console.log((await r.json()).text);
```

**Java (OkHttp)**
```java
RequestBody body = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", "recording.mp3",
        RequestBody.create(new File("recording.mp3"), MediaType.parse("audio/mpeg")))
    .build();
try (Response r = new OkHttpClient().newCall(
    new Request.Builder().url("http://<your-server>:9080/v1/audio/transcriptions").post(body).build()
).execute()) { System.out.println(r.body().string()); }
```

### Response Formats

**verbose_json** — full segments with timestamps:
```json
{
  "text": "Full transcription text",
  "language": "en",
  "duration": 180.0,
  "segments": [{"id":1, "start":0.0, "end":5.2, "text": "..."}]
}
```

**srt** — import directly into video editors:
```
1
00:00:00,000 --> 00:00:05,200
First segment text
```

## 🔧 Operations

```bash
sudo ./ops.sh <command> [options]
```

| Command | Description |
|---------|-------------|
| `install` | Full install (model + service) |
| `model-install` | Model and pip deps only |
| `service-install` | systemd/launchd only |
| `start` / `stop` / `restart` | Service control |
| `status` | Show PID, memory, model, port |
| `--dry-run` | Preview mode for all write operations |
| `--user` | User-level install (no root, `systemctl --user`) |

### Auto Model Selection

| Platform | Hardware | Selected Model |
|----------|----------|:--------------:|
| macOS ≥64GB | Apple Silicon | `large-v3` 🏆 |
| macOS ≥24GB | Apple Silicon | `large-v3-turbo` |
| macOS ≥16GB | Apple Silicon | `medium` |
| Linux GPU ≥16GB VRAM | NVIDIA | `large-v3` 🏆 |
| Linux GPU ≥8GB VRAM | NVIDIA | `large-v3-turbo` |
| Linux CPU ≥12GB RAM | No GPU | `medium` |
| Linux CPU ≥6GB RAM | No GPU | `small` |

> Override: `WHISPER_MODEL=large-v3 sudo ./ops.sh install`

## 🎵 Sample Audio Files & CLI Quick Test

```bash
# CLI mode: transcribe directly (no server needed)
python3 whisper_server.py --transcribe samples/en.wav
python3 whisper_server.py --transcribe samples/zh.wav --language zh --format json
python3 whisper_server.py --transcribe meeting.mp3 --format srt --output meeting.srt

# All CLI options
python3 whisper_server.py --transcribe <file> \
    --language en|zh|ja|ko|... \   # omit for auto-detect
    --format text|json|srt|verbose_json \
    --model tiny|base|small|medium|large-v3 \
    --output result.txt            # save to file
```

Pre-recorded samples in 9 languages:

| Language | File | Content |
|----------|------|---------|
| 🇬🇧 English | [`samples/en.wav`](samples/en.wav) | "Hello, this is a test of the Whisper speech recognition system..." |
| 🇨🇳 Chinese | [`samples/zh.wav`](samples/zh.wav) | "你好，这是一个语音识别测试..." |
| 🇯🇵 Japanese | [`samples/ja.wav`](samples/ja.wav) | "こんにちは、これは音声認識のテストです..." |
| 🇰🇷 Korean | [`samples/ko.wav`](samples/ko.wav) | "안녕하세요, 이것은 음성 인식 테스트입니다..." |
| 🇫🇷 French | [`samples/fr.wav`](samples/fr.wav) | "Bonjour, ceci est un test du système..." |
| 🇩🇪 German | [`samples/de.wav`](samples/de.wav) | "Hallo, dies ist ein Test des Spracherkennungssystems..." |
| 🇪🇸 Spanish | [`samples/es.wav`](samples/es.wav) | "Hola, esta es una prueba del sistema..." |
| 🇷🇺 Russian | [`samples/ru.wav`](samples/ru.wav) | "Здравствуйте, это тест системы распознавания речи..." |
| 🇸🇦 Arabic | [`samples/ar.wav`](samples/ar.wav) | "مرحباً، هذا اختبار لنظام التعرف على الكلام..." |

```bash
# Quick test any sample
curl -X POST http://localhost:9080/v1/audio/transcriptions \
  -F "file=@samples/zh.wav" \
  -F "response_format=verbose_json"
```

## 📋 Requirements

| OS | Engine | Python | Notes |
|----|--------|:------:|-------|
| **Linux** | faster-whisper (CTranslate2) | 3.10+ | NVIDIA GPU optional but recommended |
| **macOS Apple Silicon** | mlx-whisper (MLX GPU) | 3.10+ | macOS 13+, M-series |
| **macOS Intel** | faster-whisper (CTranslate2 CPU) | 3.10+ | same as Linux CPU |

## 🤝 Contributing

Issues and PRs welcome! Especially:

- Transcription quality feedback for non-English languages
- Windows platform support
- Docker deployment option
- Documentation translations

## 📄 License

MIT License © 2026 Whisper ASR Contributors

Whisper model weights by OpenAI (MIT licensed), downloaded automatically from HuggingFace Hub.

---

**Keywords**: `speech-to-text` `STT` `whisper` `faster-whisper` `mlx-whisper` `self-hosted` `OpenAI-compatible` `voice-recognition` `audio-transcription` `local-ai` `privacy-first`
