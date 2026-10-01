"""
src/sound_alert.py
------------------
Robust, automotive-grade sound alert system for ADAS.
Supports:
1. Native Windows hardware audio playback (winsound) with zero latency.
2. Natural spoken driver voice warnings (pyttsx3) via non-blocking background thread.
3. Browser HTML5 audio (<audio autoplay>) via Streamlit components for browser playback.
"""

import os
import time
import base64
import threading
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components

# Check native audio libraries
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

# Paths to sound assets
ALERT_HIGH_PATH = os.path.abspath("output/alert_high.wav")
ALERT_WARN_PATH = os.path.abspath("output/alert_warning.wav")

# Pre-cache base64 audio for browser HTML5 playback
B64_ALERT_HIGH = ""
B64_ALERT_WARN = ""

if os.path.exists(ALERT_HIGH_PATH):
    try:
        with open(ALERT_HIGH_PATH, "rb") as f:
            B64_ALERT_HIGH = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        pass

if os.path.exists(ALERT_WARN_PATH):
    try:
        with open(ALERT_WARN_PATH, "rb") as f:
            B64_ALERT_WARN = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        pass

# Global cooldown state to prevent repetitive spam
_last_sound_time = 0.0
_last_sound_alert = ""
_COOLDOWN_SECONDS = 3.0
_tts_lock = threading.Lock()


def play_sound_alert(
    risk: str,
    alert_text: str = "",
    mode: str = "Chime + Spoken Voice",
    volume: int = 80,
    force: bool = False,
    container=None
):
    """
    Play sound alert via both hardware speaker (winsound/pyttsx3)
    and browser (HTML5 audio component).
    """
    global _last_sound_time, _last_sound_alert

    if risk not in ["High", "Medium"] and not force:
        return

    now = time.time()
    # Debounce: avoid firing multiple times per second on continuous video
    if not force and (now - _last_sound_time < _COOLDOWN_SECONDS) and (_last_sound_alert == alert_text):
        return

    _last_sound_time = now
    _last_sound_alert = alert_text

    want_chime = ("Chime" in mode) or ("Both" in mode) or (mode == "Chime + Spoken Voice") or (mode == "Warning Chime Tone Only")
    want_voice = ("Voice" in mode) or ("Both" in mode) or (mode == "Chime + Spoken Voice") or (mode == "Spoken Voice Only")

    # 1. Native Windows Hardware Audio (Immediate Speaker Output)
    wav_path = ALERT_HIGH_PATH if risk == "High" else ALERT_WARN_PATH
    if want_chime and HAS_WINSOUND and os.path.exists(wav_path):
        try:
            winsound.PlaySound(wav_path, winsound.SND_ASYNC | winsound.SND_FILENAME)
        except Exception as e:
            print("Winsound error:", e)

    # 2. Spoken Voice Alert (Non-blocking background thread)
    if want_voice and alert_text and HAS_TTS:
        def _tts_worker(text_to_speak, delay_sec):
            if not _tts_lock.acquire(blocking=False):
                # An alert is already speaking, do not overlap
                return
            try:
                time.sleep(delay_sec)
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(text_to_speak)
                engine.runAndWait()
            except Exception as e:
                print("TTS error:", e)
            finally:
                _tts_lock.release()

        # If chime is also playing, wait 0.35s so the chime sounds first, then the voice speaks
        delay = 0.35 if want_chime else 0.0
        threading.Thread(target=_tts_worker, args=(alert_text, delay), daemon=True).start()

    # 3. Browser-Side Audio Fallback (via Streamlit component iframe)
    b64_data = B64_ALERT_HIGH if risk == "High" else B64_ALERT_WARN
    vol_float = max(0.1, min(1.0, volume / 100.0))

    html_code = f"""
    <html>
    <head>
    <script>
    (function() {{
        try {{
            // 1. Play HTML5 Audio element
            var audio = document.getElementById("adas_audio");
            if (audio) {{
                audio.volume = {vol_float};
                audio.play().catch(function(err) {{
                    console.log("Audio autoplay prevented by browser:", err);
                }});
            }}

            // 2. Browser speech synthesis fallback
            var wantVoice = {"true" if want_voice else "false"};
            var text = "{alert_text.replace('"', '')}";
            if (wantVoice && text && window.speechSynthesis) {{
                var delay = {"400" if want_chime else "0"};
                setTimeout(function() {{
                    window.speechSynthesis.cancel();
                    var msg = new SpeechSynthesisUtterance(text);
                    msg.rate = 1.15;
                    msg.pitch = 1.0;
                    msg.volume = {vol_float};
                    window.speechSynthesis.speak(msg);
                }}, delay);
            }}
        }} catch(e) {{
            console.error("Browser alert error:", e);
        }}
    }})();
    </script>
    </head>
    <body style="margin:0; padding:0; background:transparent;">
        <audio id="adas_audio" autoplay src="data:audio/wav;base64,{b64_data}"></audio>
    </body>
    </html>
    """

    target = container if container is not None else st
    try:
        with target:
            components.html(html_code, height=0, width=0)
    except Exception:
        pass
