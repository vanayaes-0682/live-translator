#!/usr/bin/env python3
"""
launcher.py - System tray GUI for Live Translator.

Your non-technical team double-clicks this (or the compiled .exe).
They get a system tray icon with simple Start / Stop menu items.
No terminal, no flags, no Python knowledge needed.
"""

import os
import sys
import threading
import queue
import tempfile
import wave
import io
from pathlib import Path

# ── GUI / tray ──────────────────────────────────────────────────────────────
try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    print("Missing deps. Run: pip install pystray Pillow")
    sys.exit(1)

# ── Audio + AI ──────────────────────────────────────────────────────────────
try:
    import openai
    import pyaudio
    import numpy as np
    from dotenv import load_dotenv
except ImportError:
    print("Missing deps. Run: pip install -r requirements.txt")
    sys.exit(1)

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5
MIN_AUDIO_DURATION = 0.5
TTS_VOICE = "nova"
TTS_SPEED = 1.0

# Shared state
_running = False
_stop_event = threading.Event()
_thread = None


# ── Tray icon image (simple green circle) ───────────────────────────────────
def make_icon(color="green"):
    img = Image.new("RGB", (64, 64), color="white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, 56, 56), fill=color)
    return img


# ── Core translation helpers (same as app.py) ────────────────────────────────
def get_rms(data):
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0


def build_wav_bytes(raw_pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


def transcribe_audio(wav_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    try:
        with open(tmp, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="verbose_json"
            )
        return result.text.strip(), result.language
    finally:
        Path(tmp).unlink(missing_ok=True)


def translate_text(text, source_lang):
    if source_lang == "en":
        instruction = "Translate the following English text to Spanish (Mexican, professional tone). Return ONLY the translated text."
    else:
        instruction = "Translate the following Spanish text to English (professional tone). Return ONLY the translated text."
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def text_to_speech(text):
    resp = client.audio.speech.create(
        model="tts-1", voice=TTS_VOICE, input=text,
        speed=TTS_SPEED, response_format="wav"
    )
    return resp.content


def play_audio(audio_bytes, pa, output_device_index=None):
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


def find_virtual_cable_output(pa):
    """Auto-detect VB-Audio CABLE Input device index."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info["name"].lower()
        if "cable input" in name and info["maxOutputChannels"] > 0:
            return i
    return None  # Fall back to default output


# ── Main recording loop ──────────────────────────────────────────────────────
def translation_loop():
    global _running
    pa = pyaudio.PyAudio()
    output_device = find_virtual_cable_output(pa)

    stream = pa.open(
        format=FORMAT, channels=CHANNELS, rate=RATE,
        input=True, frames_per_buffer=CHUNK
    )

    pq = queue.Queue()

    def worker():
        while not _stop_event.is_set() or not pq.empty():
            try:
                raw_pcm = pq.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                wav = build_wav_bytes(raw_pcm)
                text, lang = transcribe_audio(wav)
                if text:
                    translation = translate_text(text, lang)
                    audio_out = text_to_speech(translation)
                    play_audio(audio_out, pa, output_device)
            except Exception as e:
                print(f"[Worker error] {e}")
            finally:
                pq.task_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    frames = []
    silent_chunks = 0
    silence_limit = int(SILENCE_DURATION * RATE / CHUNK)
    speaking = False

    while not _stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
        except Exception:
            break
        rms = get_rms(data)
        if rms > SILENCE_THRESHOLD:
            speaking = True
            silent_chunks = 0
            frames.append(data)
        elif speaking:
            frames.append(data)
            silent_chunks += 1
            if silent_chunks >= silence_limit:
                raw_pcm = b"".join(frames)
                duration = len(raw_pcm) / (RATE * 2)
                if duration >= MIN_AUDIO_DURATION:
                    pq.put(raw_pcm)
                frames = []
                speaking = False
                silent_chunks = 0

    stream.stop_stream()
    stream.close()
    pa.terminate()
    _running = False


# ── Tray menu actions ────────────────────────────────────────────────────────
def on_start(icon, item):
    global _running, _thread
    if _running:
        return
    _running = True
    _stop_event.clear()
    icon.icon = make_icon("green")
    icon.title = "Live Translator - ACTIVE"
    _thread = threading.Thread(target=translation_loop, daemon=True)
    _thread.start()


def on_stop(icon, item):
    global _running
    if not _running:
        return
    _stop_event.set()
    _running = False
    icon.icon = make_icon("gray")
    icon.title = "Live Translator - Stopped"


def on_quit(icon, item):
    on_stop(icon, item)
    icon.stop()


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    menu = pystray.Menu(
        pystray.MenuItem("Start Translating", on_start, default=True),
        pystray.MenuItem("Stop Translating", on_stop),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon(
        "LiveTranslator",
        make_icon("gray"),
        "Live Translator - Ready",
        menu,
    )
    icon.run()


if __name__ == "__main__":
    main()
