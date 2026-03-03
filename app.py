#!/usr/bin/env python3
"""
Live Translator Agent - Real-time English <-> Spanish voice translation
for Teams/Zoom meetings with Mexican clients.

Architecture:
  1. Capture microphone audio in real-time
  2. Send to OpenAI Whisper for speech-to-text
  3. Detect language (en/es) and translate via OpenAI GPT-4o
  4. Convert translated text to speech via OpenAI TTS
  5. Play audio output through virtual audio cable to meeting
"""

import os
import queue
import threading
import time
import tempfile
from pathlib import Path

import openai
import pyaudio
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500      # RMS below this = silence
SILENCE_DURATION = 1.5       # seconds of silence before processing
MIN_AUDIO_DURATION = 0.5     # ignore clips shorter than this (seconds)

TTS_VOICE = "nova"           # Options: alloy, echo, fable, onyx, nova, shimmer
TTS_SPEED = 1.0

# ---------------------------------------------------------------------------
# Helper: detect dominant language from Whisper segments
# ---------------------------------------------------------------------------
def translate_text(text: str, source_lang: str) -> str:
    """Translate text from source_lang to the opposite language."""
    if source_lang == "en":
        target_lang = "Spanish (Mexican, professional tone)"
        instruction = f"Translate the following English text to {target_lang}. Return ONLY the translated text, no explanations."
    else:
        target_lang = "English (professional tone)"
        instruction = f"Translate the following Spanish text to {target_lang}. Return ONLY the translated text, no explanations."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def text_to_speech(text: str) -> bytes:
    """Convert text to speech audio bytes using OpenAI TTS."""
    response = client.audio.speech.create(
        model="tts-1",
        voice=TTS_VOICE,
        input=text,
        speed=TTS_SPEED,
        response_format="wav",
    )
    return response.content


def play_audio(audio_bytes: bytes, pa: pyaudio.PyAudio, output_device_index: int = None):
    """Play WAV audio bytes through PyAudio (routes to virtual cable if configured)."""
    import wave, io
    with wave.open(io.BytesIO(audio_bytes)) as wf:
        stream = pa.open(
            format=pa.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
            output_device_index=output_device_index,
        )
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        stream.stop_stream()
        stream.close()


def get_rms(data: bytes) -> float:
    """Return RMS amplitude of raw PCM bytes."""
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0


# ---------------------------------------------------------------------------
# Main recording + translation loop
# ---------------------------------------------------------------------------
def transcribe_audio(audio_bytes: bytes) -> tuple[str, str]:
    """Send audio to Whisper, return (transcript, detected_language)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
            )
        return result.text.strip(), result.language  # language is ISO 639-1
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def build_wav_bytes(raw_pcm: bytes, rate: int = RATE, channels: int = CHANNELS) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    import wave, io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(rate)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


def list_audio_devices(pa: pyaudio.PyAudio):
    """Print available audio devices to help the user pick input/output."""
    print("\n=== Available Audio Devices ===")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"  [{i}] {info['name']}  (in:{info['maxInputChannels']} out:{info['maxOutputChannels']})")
    print("================================\n")


def record_and_translate(
    input_device_index: int = None,
    output_device_index: int = None,
):
    """
    Main loop:
      - Records from microphone (or virtual cable input)
      - Detects speech / silence boundaries
      - Transcribes, translates, and plays back translated audio
    """
    pa = pyaudio.PyAudio()
    list_audio_devices(pa)

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=input_device_index,
    )

    print("[LiveTranslator] Listening... (Ctrl+C to stop)")
    print(f"  Input device  : {input_device_index or 'system default'}")
    print(f"  Output device : {output_device_index or 'system default'}")
    print(f"  TTS voice     : {TTS_VOICE}")
    print()

    frames: list[bytes] = []
    silent_chunks = 0
    silence_chunk_limit = int(SILENCE_DURATION * RATE / CHUNK)
    speaking = False

    # Process queue so translation/TTS doesn't block recording
    process_queue: queue.Queue = queue.Queue()

    def worker():
        while True:
            raw_pcm = process_queue.get()
            if raw_pcm is None:
                break
            try:
                wav_bytes = build_wav_bytes(raw_pcm)
                transcript, lang = transcribe_audio(wav_bytes)
                if not transcript:
                    continue
                print(f"  [{lang.upper()}] {transcript}")
                translation = translate_text(transcript, lang)
                print(f"  --> {translation}")
                audio_out = text_to_speech(translation)
                play_audio(audio_out, pa, output_device_index)
            except Exception as exc:
                print(f"  [ERROR] {exc}")
            finally:
                process_queue.task_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = get_rms(data)

            if rms > SILENCE_THRESHOLD:
                speaking = True
                silent_chunks = 0
                frames.append(data)
            elif speaking:
                frames.append(data)
                silent_chunks += 1
                if silent_chunks >= silence_chunk_limit:
                    # End of utterance
                    raw_pcm = b"".join(frames)
                    duration = len(raw_pcm) / (RATE * 2)  # 16-bit = 2 bytes
                    if duration >= MIN_AUDIO_DURATION:
                        process_queue.put(raw_pcm)
                    frames = []
                    speaking = False
                    silent_chunks = 0
    except KeyboardInterrupt:
        print("\n[LiveTranslator] Stopping...")
    finally:
        process_queue.put(None)  # signal worker to exit
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live English <-> Spanish Translator")
    parser.add_argument(
        "--input", type=int, default=None,
        help="PyAudio input device index (default: system default mic)"
    )
    parser.add_argument(
        "--output", type=int, default=None,
        help="PyAudio output device index (default: system default speaker)"
    )
    parser.add_argument(
        "--voice", type=str, default=TTS_VOICE,
        help="OpenAI TTS voice (alloy/echo/fable/onyx/nova/shimmer)"
    )
    parser.add_argument(
        "--speed", type=float, default=TTS_SPEED,
        help="TTS playback speed (0.25 - 4.0)"
    )
    args = parser.parse_args()

    TTS_VOICE = args.voice
    TTS_SPEED = args.speed

    record_and_translate(
        input_device_index=args.input,
        output_device_index=args.output,
    )
