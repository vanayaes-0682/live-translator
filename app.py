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
  6. Show latest Spanish translation in a small caption window
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

import tkinter as tk
from tkinter import scrolledtext

load_dotenv()

# Shared state
_is_playing = False


# -------------------------------------ti--------------------------------------
# Configuration
# ---------------------------------------------------------------------------

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 200       # RMS below this = silence
SILENCE_DURATION = 0.6        # seconds of silence before processing
MIN_AUDIO_DURATION = 0.3      # ignore clips shorter than this (seconds)

TTS_VOICE = "onyx"           # Options: alloy, echo, fable, onyx, nova, shimmer
TTS_SPEED = 1.01

# Preferred device name keywords (used when input/output indices are None)
PREFERRED_INPUT_KEYWORD = "Microphone (Realtek HD Audio Mic input)"
PREFERRED_OUTPUT_KEYWORD = "CABLE Output (VB-Audio Virtual Cable)"  # for Teams

# Tkinter globals
_root: tk.Tk | None = None
_caption_box: scrolledtext.ScrolledText | None = None

# ---------------------------------------------------------------------------
# GUI: small Spanish caption window
# ---------------------------------------------------------------------------
def init_caption_window():
    global _root, _caption_box

    _root = tk.Tk()
    _root.title("Spanish Captions - Acronym")
    _root.attributes("-topmost", True)
    _root.geometry("800x300")
    _root.configure(bg="#111111")

    # Blue Acronym title at top
    title_label = tk.Label(
        _root,
        text="Acronym",
        font=("Segoe UI", 16, "bold"),
        fg="#FF6B00",      # orange
        bg="#111111",
    )
    title_label.pack(side=tk.TOP, pady=(8, 0))

    _caption_box = scrolledtext.ScrolledText(
        _root,
        wrap=tk.WORD,
        font=("Segoe UI", 15),
        state="disabled",
        bg="#111111",
        fg="#ffffff",
    )
    _caption_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))


def update_caption(text: str):
    """Update the caption box with the latest Spanish translation."""
    if _root is None or _caption_box is None:
        return

    def _do_update():
        _caption_box.configure(state="normal")
        _caption_box.delete("1.0", tk.END)
        _caption_box.insert(tk.END, text)
        _caption_box.configure(state="disabled")

    _root.after(0, _do_update)

# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------
def translate_text(text: str, source_lang: str) -> str:
    """Translate text from source_lang to the opposite language."""
    if source_lang != "es":
        target_lang = "Spanish (Mexican, professional tone)"
        instruction = (
            f"Translate the following English text to {target_lang}. "
            "Return ONLY the translated text, no explanations."
        )
    else:
        target_lang = "English (professional tone)"
        instruction = (
            f"Translate the following Spanish text to {target_lang}. "
            "Return ONLY the translated text, no explanations."
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

    global _is_playing
    _is_playing = True

    try:
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
    finally:
        _is_playing = False
        time.sleep(1.0)


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
                response_format="json",
            )
        return result.text.strip(), "en"
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


def find_device_index_by_name(
    pa: pyaudio.PyAudio,
    keyword: str,
    require_input: bool = False,
    require_output: bool = False
) -> int | None:
    """
    Return first device index whose name contains keyword.
    Optionally require it to support input and/or output channels.
    Returns None if not found.
    """
    keyword_lower = keyword.lower()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info.get("name", "")
        max_in = info.get("maxInputChannels", 0)
        max_out = info.get("maxOutputChannels", 0)

        if keyword_lower in name.lower():
            if require_input and max_in <= 0:
                continue
            if require_output and max_out <= 0:
                continue
            return i
    return None


def record_and_translate(
    input_device_index: int = None,
    output_device_index: int = None,
):
    """
    Main loop:
      - Records from microphone (or virtual cable input)
      - Detects speech / silence boundaries
      - Transcribes, translates, and plays back translated audio
      - Updates Spanish caption window
    """
    pa = pyaudio.PyAudio()
    list_audio_devices(pa)

    # Auto-select devices by name if indices are not provided
    if input_device_index is None:
        input_device_index = find_device_index_by_name(
            pa,
            PREFERRED_INPUT_KEYWORD,
            require_input=True,
            require_output=False,
        )

    if output_device_index is None:
        output_device_index = find_device_index_by_name(
            pa,
            PREFERRED_OUTPUT_KEYWORD,
            require_input=False,
            require_output=True,
        )

    # Decide whether to use default input
    use_default_input = input_device_index is None

    try:
        if use_default_input:
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
        else:
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=input_device_index,
            )
    except OSError as e:
        print(f"[ERROR] Failed to open input device (index={input_device_index}). Falling back to system default. Error: {e}")
        # Final fallback: system default
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        input_device_index = None  # we are now on default

    print(f"  [DEBUG] Using input index {input_device_index}, output index {output_device_index}")
    print("[LiveTranslator] Listening... (Ctrl+C to stop)")
    print(f"  Input device  : {input_device_index if input_device_index is not None else 'system default'}")
    print(f"  Output device : {output_device_index if output_device_index is not None else 'system default'}")
    print(f"  TTS voice     : {TTS_VOICE}")
    print()

    frames: list[bytes] = []
    silent_chunks = 0
    silence_chunk_limit = int(SILENCE_DURATION * RATE / CHUNK)
    speaking = False

    process_queue: queue.Queue = queue.Queue(maxsize=1)

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

                # Show Spanish captions when you speak English
                if lang != "es":
                    update_caption(translation)

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
            # If we are currently playing TTS audio, skip reading mic input
            if _is_playing:
                time.sleep(0.01)
                stream.read(CHUNK, exception_on_overflow=False)  # drain buffer
                continue

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
        process_queue.put(None)
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

    # Apply CLI overrides
    TTS_VOICE = args.voice
    TTS_SPEED = args.speed

    # Start GUI on main thread
    init_caption_window()

    # Start audio worker in a background thread
    audio_thread = threading.Thread(
        target=record_and_translate,
        kwargs={"input_device_index": args.input, "output_device_index": args.output},
        daemon=True,
    )
    audio_thread.start()

    _root.mainloop()



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

    # Apply CLI overrides
    TTS_VOICE = args.voice
    TTS_SPEED = args.speed

    # Start GUI on main thread
    init_caption_window()

    # Start audio worker in a background thread
    audio_thread = threading.Thread(
        target=record_and_translate,
        kwargs={"input_device_index": args.input, "output_device_index": args.output},
        daemon=True,
    )
    audio_thread.start()

    _root.mainloop()
