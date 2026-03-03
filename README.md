# Live Translator

Real-time **English <-> Spanish** voice translation agent for Teams / Zoom client meetings.

## How it works

```
Your mic --> Whisper STT --> GPT-4o translation --> OpenAI TTS --> Virtual audio cable --> Meeting
```

1. Captures your microphone in real-time
2. Detects silence to segment speech into utterances
3. Sends each utterance to **OpenAI Whisper** for transcription + language detection
4. Translates automatically: English → Mexican Spanish, Spanish → English (via **GPT-4o**)
5. Converts the translation to speech with **OpenAI TTS** (`nova` voice by default)
6. Plays the audio through your chosen output device (route to a virtual cable to share in meetings)

## Requirements

- Python 3.11+
- An **OpenAI API key** with access to Whisper, GPT-4o, and TTS
- (Optional) A virtual audio cable app to route translated audio into Teams/Zoom:
  - Windows: [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)
  - Mac: [BlackHole](https://github.com/ExistentialAudio/BlackHole)

## Quick start

```bash
# 1. Clone
git clone https://github.com/vanayaes-0682/live-translator.git
cd live-translator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Run (uses system default mic & speakers)
python app.py

# 4b. List audio devices first to find the right indexes
#     The app prints them on startup automatically.

# 4c. Route translated audio into Teams/Zoom via virtual cable
#     Set Teams microphone = "CABLE Output" (VB-Audio)
python app.py --input 0 --output 3
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--input INDEX` | system default | PyAudio input device index (your mic) |
| `--output INDEX` | system default | PyAudio output device index (virtual cable) |
| `--voice NAME` | `nova` | TTS voice: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `--speed FLOAT` | `1.0` | TTS playback speed (0.25 – 4.0) |

## Tuning silence detection

Edit the constants at the top of `app.py`:

| Constant | Default | Meaning |
|----------|---------|--------|
| `SILENCE_THRESHOLD` | `500` | RMS amplitude below which audio is considered silence |
| `SILENCE_DURATION` | `1.5` | Seconds of silence to wait before processing an utterance |
| `MIN_AUDIO_DURATION`| `0.5` | Minimum clip length (seconds) to send to Whisper |

## Meeting setup (Teams example)

1. Install **VB-Audio Virtual Cable** (free)
2. Run `python app.py --output <CABLE_Input_index>`
3. In Teams Settings → Devices → Microphone, select **CABLE Output**
4. Your clients now hear the translated voice in real-time

## Cost estimate

| Service | Rate | ~1 hr meeting |
|---------|------|---------------|
| Whisper | $0.006 / min | ~$0.36 |
| GPT-4o | ~$0.005 / 1K tokens | ~$0.50 |
| TTS-1 | $0.015 / 1K chars | ~$1.00 |
| **Total** | | **~$1.86 / hr** |

## License

MIT
