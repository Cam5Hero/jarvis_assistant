"""
Jarvis voice assistant (v1: "ask Jarvis anything") for Windows / Python 3.12.

Pipeline:
  wake word (openwakeword, JARVIS_WAKE_MODEL) -> chime -> record until silence (RMS VAD)
  -> faster-whisper transcription -> LM Studio (OpenAI-compatible) with tool calling
  -> speak reply (ElevenLabs if configured, else pyttsx3).

Usage:
  python jarvis_assistant.py          # voice mode
  python jarvis_assistant.py --text   # type instead of talk (no mic needed)
  python jarvis_assistant.py --setup  # first-run setup wizard (writes .env)

Configure via environment or a `.env` file next to this script (see .env.example).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"
ENV_EXAMPLE_PATH = APP_DIR / ".env.example"
load_dotenv(ENV_PATH)

log = logging.getLogger("jarvis")

# ---------------------------------------------------------------------------
# Config (from .env)
# ---------------------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name, "1" if default else "0").lower()
    return v not in ("0", "false", "no", "off", "")


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


# --- naming: what the assistant calls itself / the user (spoken name is just a setting) ---
ASSISTANT_NAME = _env("ASSISTANT_NAME", "Jarvis")
USER_TITLE = (os.environ.get("USER_TITLE", "sir") or "").strip()  # blank = no title


def T(text: str) -> str:
    """Fill the user's title: '{t}' (lowercase spot) / '{T}' (sentence start).
    With a blank title, drop it and its comma: 'Done, {t}.' -> 'Done.'"""
    if USER_TITLE:
        return text.replace("{T}", USER_TITLE[:1].upper() + USER_TITLE[1:]).replace("{t}", USER_TITLE)
    text = re.sub(r",?\s*\{t\}", "", text)
    text = re.sub(r"\{T\},?\s*", "", text).strip()
    return text[:1].upper() + text[1:]


# --- wake word: built-in openwakeword name or path to a custom .onnx model ---
BUILTIN_WAKE_MODELS = ("hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy")
WAKE_MODEL = _env("JARVIS_WAKE_MODEL", "hey_jarvis")


def wake_model_key(model: str = "") -> str:
    """'models/hey_friday_v0.1.onnx' -> 'hey_friday' (base used to match prediction keys)."""
    base = os.path.basename((model or WAKE_MODEL).replace("\\", "/"))
    base = re.sub(r"\.(onnx|tflite)$", "", base, flags=re.I)
    return re.sub(r"_v\d+(\.\d+)*$", "", base).lower()


def wake_phrase(model: str = "") -> str:
    """Friendly spoken form: 'hey_jarvis' -> 'Hey Jarvis'."""
    return " ".join(w.capitalize() for w in re.split(r"[_\-\s]+", wake_model_key(model)) if w)


def wake_score(scores: dict, model: str = "") -> float:
    """Score for the configured model; keys may carry version suffixes, so match on the base."""
    if not scores:
        return 0.0
    key = wake_model_key(model)
    hits = [v for k, v in scores.items() if wake_model_key(str(k)) == key or str(k).lower().startswith(key)]
    return float(max(hits if hits else scores.values()))


WAKE_PHRASE = wake_phrase()

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms frames, what openwakeword expects

LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = _env("LLM_MODEL", "qwen3-4b")
LLM_API_KEY = _env("LLM_API_KEY", "lm-studio")
LLM_NO_THINK = _env("LLM_NO_THINK", "1") not in ("0", "false", "no")
HISTORY_TURNS = int(_env_float("JARVIS_HISTORY_TURNS", 6))  # user+assistant pairs kept

WHISPER_MODEL = _env("WHISPER_MODEL", "small")  # 'base' is faster on weak PCs
WHISPER_DEVICE = _env("WHISPER_DEVICE", "auto")
WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "en") or None

WAKE_THRESHOLD = _env_float("JARVIS_WAKE_THRESHOLD", 0.5)
WAKE_SOUND = _env_bool("JARVIS_WAKE_SOUND", True)
WAKE_SOUND_VOLUME = max(0.0, min(1.0, _env_float("JARVIS_WAKE_SOUND_VOLUME", 0.3)))
WAKE_SOUND_STYLE = _env("JARVIS_WAKE_SOUND_STYLE", "bell")
WAKE_SOUND_FILE = _env("JARVIS_WAKE_SOUND_FILE", "")
# Barge-in: say the wake word while the assistant talks to cut it off. Higher threshold so
# the assistant's own voice (through speakers) doesn't trigger it. Headphones work best.
BARGE_IN = _env_bool("JARVIS_BARGE_IN", True)
BARGE_IN_THRESHOLD = _env_float("JARVIS_BARGE_IN_THRESHOLD", 0.7)
SILENCE_SECONDS = _env_float("JARVIS_SILENCE_SECONDS", 1.2)
MAX_RECORD_SECONDS = _env_float("JARVIS_MAX_RECORD_SECONDS", 10.0)
VAD_RMS_THRESHOLD = _env_float("JARVIS_VAD_RMS", 500.0)  # int16 RMS; raise in noisy rooms
INPUT_DEVICE = _env("JARVIS_INPUT_DEVICE") or None
FOLLOWUP_SECONDS = _env_float("JARVIS_FOLLOWUP_SECONDS", 6.0)  # 0 disables follow-up mode
FOLLOWUP_END_PHRASES = [
    p.strip().lower()
    for p in _env("JARVIS_FOLLOWUP_END_PHRASES",
                  "thanks,thank you,that's all,that is all,never mind,nevermind,goodbye,"
                  "bye,no thanks,that will be all").split(",")
    if p.strip()
]
PAUSE_PHRASES = [p.strip().lower() for p in _env(
    "JARVIS_PAUSE_PHRASES", "stop listening,go to sleep,mute yourself,pause listening").split(",")
    if p.strip()]
_PAUSE_REPLY_TEMPLATE = f"Going quiet, {{t}}. Say {WAKE_PHRASE} when you need me."
_RESUME_REPLY_TEMPLATE = "At your service, {t}."
PAUSE_REPLY = T(_PAUSE_REPLY_TEMPLATE)
RESUME_REPLY = T(_RESUME_REPLY_TEMPLATE)

ELEVENLABS_API_KEY = _env("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = _env("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = _env("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
ELEVENLABS_OUTPUT_FORMAT = _env("ELEVENLABS_OUTPUT_FORMAT", "pcm_24000")

def build_system_prompt() -> str:
    """The model's instructions. A function so the title can be changed while running (set_user_title)."""
    return (
    f"You are {ASSISTANT_NAME}, a witty, polite British AI butler. Your replies are spoken aloud, "
    "so answer in 1-3 short sentences of plain conversational English. Never use "
    "markdown, lists, code blocks, emojis or URLs read character by character. "
    + (f"Address the user as '{USER_TITLE}' occasionally. " if USER_TITLE
       else "Do not use 'sir', 'madam' or other titles for the user. ")
    + "Use the provided tools when the user asks "
    "for the time, to open a website, search the web, open an app, set a timer, control media "
    "playback or the system volume, "
    "or control OBS (record, stream, scenes, mic mute, replay clips, live status), or list, preview "
    "or change the wake sound (list_wake_sounds, preview_wake_sound, set_wake_sound). Only start "
    "streaming when the user clearly asked to go live. "
    "Do not end replies with offers like 'What would you like me to do?' or 'How can I "
    "assist you further?'; a closing line is added for you."
    )


SYSTEM_PROMPT = build_system_prompt()

_DEFAULT_CLOSINGS = (
    "Let me know if you need anything else.|Anything else I can do for you?|"
    "I'm here if you need me.|Just say the word if you need anything.|"
    "Happy to help.|Always a pleasure.|Standing by.|Ready when you are."
)
# JARVIS_CLOSING_PHRASE: one phrase, or several separated by | (picked at random). Blank = none.
CLOSING_PHRASES = [p.strip() for p in _env("JARVIS_CLOSING_PHRASE", _DEFAULT_CLOSINGS).split("|") if p.strip()]
CLOSING_PHRASE = CLOSING_PHRASES[0] if CLOSING_PHRASES else ""
_last_closing = [""]


def add_closing(reply: str) -> str:
    """Append a random closing phrase (no repeats back to back) unless the reply is a question or already offers help."""
    r = reply.strip()
    low = r.lower()
    if not CLOSING_PHRASES or not r or r.endswith("?") or "anything else" in low or "need me" in low:
        return r
    import random
    choices = [p for p in CLOSING_PHRASES if p != _last_closing[0]] or CLOSING_PHRASES
    pick = random.choice(choices)
    _last_closing[0] = pick
    return f"{r} {pick}"

# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------


class Speaker:
    """Speaks text via ElevenLabs (if configured) or pyttsx3. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.speaking = threading.Event()  # set while audio plays; listener pauses
        self.interrupted = threading.Event()  # set by stop(); cleared at the start of each say()
        self._engine: Any = None  # current pyttsx3 engine (so stop() can reach it)
        self._eleven = None
        if ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
            try:
                from elevenlabs.client import ElevenLabs

                self._eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)
                log.info("TTS: ElevenLabs voice %s", ELEVENLABS_VOICE_ID)
            except Exception as e:  # ImportError or client error
                log.warning("ElevenLabs unavailable (%s); using pyttsx3.", e)
        if self._eleven is None:
            log.info("TTS: pyttsx3")

    def _pcm_rate(self) -> int:
        try:
            return int(ELEVENLABS_OUTPUT_FORMAT.split("_", 1)[1])
        except (IndexError, ValueError):
            return 24000

    def _say_eleven(self, text: str) -> bool:
        try:
            import numpy as np
            import sounddevice as sd

            chunks = self._eleven.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                text=text,
                model_id=ELEVENLABS_MODEL_ID,
                output_format=ELEVENLABS_OUTPUT_FORMAT,
            )
            raw = b"".join(chunks)
            if not raw:
                return False
            if self.interrupted.is_set():  # stopped while still downloading
                return True
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(pcm, self._pcm_rate())
            sd.wait()  # returns early when stop() calls sd.stop()
            return True
        except Exception as e:
            log.warning("ElevenLabs TTS failed: %s", e)
            return False

    def _say_pyttsx3(self, text: str) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()  # fresh engine per call avoids runAndWait hangs
            self._engine = engine
            want = (_env("JARVIS_VOICE") or "").lower()
            if want:  # pick an installed Windows voice by part of its name, e.g. "david" or "zira"
                for v in engine.getProperty("voices"):
                    if want in (v.name or "").lower():
                        engine.setProperty("voice", v.id)
                        break
            rate = _env("JARVIS_VOICE_RATE")
            if rate.isdigit():
                engine.setProperty("rate", int(rate))  # words per minute, default about 200
            vol = _env("JARVIS_VOICE_VOLUME")
            try:
                if vol:
                    engine.setProperty("volume", max(0.0, min(1.0, float(vol))))
            except ValueError:
                pass
            if self.interrupted.is_set():
                return
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            log.warning("pyttsx3 failed (%s)", e)
        finally:
            self._engine = None

    def stop(self) -> None:
        """Cut speech off now (barge-in). Safe to call from any thread, also when silent.

        ElevenLabs PCM: sd.stop() ends playback and wakes sd.wait(). pyttsx3: engine.stop()
        from another thread; on Windows SAPI5 this usually ends speech at the next word but
        it is best effort (some drivers finish the current sentence)."""
        self.interrupted.set()
        if self._eleven is not None:
            try:
                import sounddevice as sd

                sd.stop()
            except Exception as e:
                log.debug("sd.stop failed: %s", e)
        eng = self._engine
        if eng is not None:
            try:
                eng.stop()
            except Exception as e:
                log.debug("pyttsx3 stop failed: %s", e)

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        print(f"{ASSISTANT_NAME}: {text}")
        with self._lock:
            self.interrupted.clear()
            self.speaking.set()
            try:
                if not (self._eleven and self._say_eleven(text)):
                    self._say_pyttsx3(text)
            finally:
                if not self.interrupted.is_set():
                    time.sleep(0.3)  # let speaker echo die down before listening again
                self.speaking.clear()


class PrintSpeaker:
    """Text-only speaker used by tests / --mute."""

    speaking = threading.Event()
    interrupted = threading.Event()

    def say(self, text: str) -> None:
        print(f"{ASSISTANT_NAME}: {text.strip()}")

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Skills (tools). Plain functions registered in SKILLS; each returns a string
# result that is fed back to the model.
# ---------------------------------------------------------------------------

_speaker: Any = PrintSpeaker()  # replaced in main()

APP_WHITELIST: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "spotify": "spotify:",  # URI handler; works for Store and desktop installs
    "steam": "steam://open/main",  # Steam's URI handler opens the Steam client
    "netflix": "https://www.netflix.com/browse",  # opens in the default browser
    "prime video": "https://www.primevideo.com/",
    "prime": "https://www.primevideo.com/",
}


def get_time() -> str:
    now = _dt.datetime.now()
    return now.strftime("It is %I:%M %p on %A, %B %d, %Y.").replace(" 0", " ")


def open_website(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url}"


def web_search(query: str) -> str:
    webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote_plus(query))
    return f"Opened a web search for '{query}'"


# ---------------------------------------------------------------------------
# App discovery: index Start Menu / Desktop shortcuts, Steam games and Store apps
# so open_app can launch anything installed. Only indexed or whitelisted items are
# ever launched; model text is never executed.
# ---------------------------------------------------------------------------

APP_CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "app_index.json"
APP_CACHE_MAX_AGE = 24 * 3600
SETTINGS_PATH = APP_CACHE_PATH.parent / "settings.json"
CONSENT_QUESTION = f"{ASSISTANT_NAME} here. Would you like me to scan your PC so I can open your apps for you?"
_CONSENT_DECLINED_TEMPLATE = "No problem, {t}. Say scan my apps any time."
CONSENT_DECLINED = T(_CONSENT_DECLINED_TEMPLATE)


# --- opt-in consent (stored in .cache/settings.json: app_scan_consent true/false/null) ---

def load_settings() -> dict:
    try:
        d = json.loads(SETTINGS_PATH.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def get_scan_consent() -> bool | None:
    v = load_settings().get("app_scan_consent")
    return v if isinstance(v, bool) else None


def set_scan_consent(value: bool) -> None:
    d = load_settings()
    d["app_scan_consent"] = value
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(d, indent=1))
    except OSError as e:
        log.warning("Could not save settings: %s", e)


# --- wake sound choice (single source of truth; saved in .cache/settings.json as 'wake_sound') ---

WAKE_SOUND_STYLES: list[tuple[str, str, str]] = [  # (name, friendly label, category)
    ("bell", "Bell", "Classic"), ("chime3", "Three-note chime", "Classic"),
    ("droplet", "Water droplet", "Classic"), ("beep", "Two-tone beep", "Classic"),
    ("warp", "Warp", "Space"), ("comms", "Starship comms", "Space"), ("cosmos", "Cosmos", "Space"),
    ("sonar", "Sonar ping", "Space"), ("blip", "Sci-fi blip", "Space"),
    ("piano", "Piano", "Instruments"), ("acoustic_guitar", "Acoustic guitar", "Instruments"),
    ("harp", "Harp", "Instruments"), ("violin", "Violin", "Instruments"), ("cello", "Cello", "Instruments"),
    ("trumpet", "Trumpet", "Instruments"), ("sax", "Saxophone", "Instruments"),
    ("flute", "Flute", "Instruments"), ("clarinet", "Clarinet", "Instruments"),
    ("marimba", "Marimba", "Instruments"), ("xylophone", "Xylophone", "Instruments"),
    ("vibraphone", "Vibraphone", "Instruments"), ("steel_drum", "Steel drum", "Instruments"),
    ("kalimba", "Kalimba", "Instruments"), ("music_box", "Music box", "Instruments"),
    ("banjo", "Banjo", "Instruments"), ("sitar", "Sitar", "Instruments"),
    ("synth_lead", "Synth lead", "Instruments"), ("organ", "Organ", "Instruments"),
    ("taiko", "Taiko drum", "Instruments"),
    ("chug", "Metal palm-mute chug", "Fun"), ("powerup", "Power-up", "Fun"), ("chirp", "Bird chirp", "Fun"),
]
WAKE_SOUND_NAMES = [n for n, _, _ in WAKE_SOUND_STYLES]
WAKE_SOUND_CATEGORIES = ["Classic", "Space", "Instruments", "Fun"]
_WAKE_ALIASES = {
    "guitar": "acoustic_guitar", "acoustic": "acoustic_guitar", "saxophone": "sax", "synth": "synth_lead",
    "synthesizer": "synth_lead", "drum": "taiko", "drums": "taiko", "taiko drum": "taiko",
    "steel drums": "steel_drum", "steel pan": "steel_drum", "musicbox": "music_box", "vibes": "vibraphone",
    "chime": "chime3", "three chime": "chime3", "drop": "droplet", "power up": "powerup", "ping": "sonar",
    "bird": "chirp", "default": "bell", "original": "beep", "shuffle": "random", "surprise me": "random",
}
_last_random_style = ""


def _norm_sound(text: str) -> str:
    t = re.sub(r"[^a-z0-9 _-]", "", (text or "").lower()).replace("-", " ").replace("_", " ")
    t = re.sub(r"\b(the|a|an|wake|sound|sounds|chime sound|noise|one|please)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def match_wake_sound(text: str) -> str | None:
    """Spoken/typed name -> style name ('random' allowed), fuzzy via difflib; None if no match."""
    import difflib

    t = _norm_sound(text)
    if not t:
        return None
    key = t.replace(" ", "_")
    if key in WAKE_SOUND_NAMES or key == "random":
        return key
    if t in _WAKE_ALIASES:
        return _WAKE_ALIASES[t]
    labels = {lbl.lower(): n for n, lbl, _ in WAKE_SOUND_STYLES}
    if t in labels:
        return labels[t]
    cands = {n.replace("_", " "): n for n in WAKE_SOUND_NAMES + ["random"]}
    cands.update(labels)
    cands.update(_WAKE_ALIASES)
    m = difflib.get_close_matches(t, list(cands), n=1, cutoff=0.72)
    if m:
        v = cands[m[0]]
        return _WAKE_ALIASES.get(v, v) if v in _WAKE_ALIASES else v
    return None


def validate_wake_style(name: str | None, source: str = "setting") -> str:
    n = (name or "").strip().lower()
    if n in WAKE_SOUND_NAMES or n == "random":
        return n
    if n:
        log.warning("Unknown wake sound %r in %s; using bell.", name, source)
    return "bell"


def resolve_wake_style() -> str:
    """Precedence: settings.json 'wake_sound' > env JARVIS_WAKE_SOUND_STYLE > 'bell'."""
    saved = load_settings().get("wake_sound")
    if isinstance(saved, str) and saved.strip():
        return validate_wake_style(saved, "settings.json")
    return validate_wake_style(_env("JARVIS_WAKE_SOUND_STYLE", "bell"), "JARVIS_WAKE_SOUND_STYLE")


def save_wake_style(name: str) -> None:
    global WAKE_SOUND_STYLE
    WAKE_SOUND_STYLE = validate_wake_style(name)
    d = load_settings()
    d["wake_sound"] = WAKE_SOUND_STYLE
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(d, indent=1))
    except OSError as e:
        log.warning("Could not save settings: %s", e)


def pick_play_style(style: str | None = None) -> str:
    """Concrete style to play; 'random' picks a different one than last time."""
    import random

    global _last_random_style
    st = validate_wake_style(style if style is not None else WAKE_SOUND_STYLE)
    if st != "random":
        return st
    st = random.choice([n for n in WAKE_SOUND_NAMES if n != _last_random_style])
    _last_random_style = st
    return st


def wake_label(name: str) -> str:
    if name == "random":
        return "Random (a different one each wake)"
    return next((lbl for n, lbl, _ in WAKE_SOUND_STYLES if n == name), name)


WAKE_SOUND_STYLE = resolve_wake_style()
_SKIP_LNK = re.compile(
    r"\b(uninstall\w*|unins\d*|help|readme|read me|website|web site|documentation|manual|"
    r"release notes|license|licence|faq|support|changelog|what's new)\b", re.I)
_SKIP_STEAM_IDS = {"228980"}  # Steamworks Common Redistributables


def _app_norm(name: str) -> str:
    t = re.sub(r"[™®©]", "", name.lower()).replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _env_dir(var: str, *parts: str) -> Path | None:
    base = os.environ.get(var)
    return Path(base, *parts) if base else None


def default_shortcut_dirs() -> list[Path]:
    dirs = [
        _env_dir("ProgramData", "Microsoft", "Windows", "Start Menu", "Programs"),
        _env_dir("APPDATA", "Microsoft", "Windows", "Start Menu", "Programs"),
        _env_dir("USERPROFILE", "Desktop"),
        _env_dir("PUBLIC", "Desktop"),
    ]
    return [d for d in dirs if d]


def default_steam_path() -> Path | None:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
            return Path(winreg.QueryValueEx(k, "SteamPath")[0])
    except Exception:
        pass
    p = Path(r"C:\Program Files (x86)\Steam")
    return p if p.exists() else None


def default_start_apps() -> list[dict]:
    """Read-only Get-StartApps listing (Store/UWP + desktop apps)."""
    if os.name != "nt":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-StartApps | ConvertTo-Json"],
            capture_output=True, text=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
        data = json.loads(out or "[]")
        return data if isinstance(data, list) else [data]
    except Exception as e:
        log.warning("Get-StartApps failed: %s", e)
        return []


def scan_shortcuts(dirs: list[Path]) -> list[dict]:
    apps = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() not in (".lnk", ".url") or _SKIP_LNK.search(f.stem):
                continue
            apps.append({"name": f.stem, "kind": "shortcut", "target": str(f)})
    return apps


def scan_steam(steam: Path | None) -> list[dict]:
    if not steam:
        return []
    libs = {steam}
    vdf = steam / "steamapps" / "libraryfolders.vdf"
    try:
        for m in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(encoding="utf-8", errors="ignore")):
            libs.add(Path(m.group(1).replace("\\\\", "\\")))
    except OSError:
        pass
    apps = []
    for lib in libs:
        for acf in (lib / "steamapps").glob("appmanifest_*.acf"):
            try:
                txt = acf.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            aid = re.search(r'"appid"\s+"(\d+)"', txt)
            nm = re.search(r'"name"\s+"([^"]+)"', txt)
            if aid and nm and aid.group(1) not in _SKIP_STEAM_IDS:
                apps.append({"name": nm.group(1), "kind": "steam",
                             "target": f"steam://rungameid/{aid.group(1)}"})
    return apps


def scan_start_apps(items: list[dict]) -> list[dict]:
    apps = []
    for it in items:
        name, aid = str(it.get("Name") or "").strip(), str(it.get("AppID") or "").strip()
        if name and aid and not re.search(r'["\r\n]', aid) and not _SKIP_LNK.search(name):
            apps.append({"name": name, "kind": "uwp", "target": aid})
    return apps


class AppIndex:
    def __init__(self, cache_path: Path = APP_CACHE_PATH, shortcut_dirs=None, steam_path=None,
                 start_apps=None) -> None:
        self.cache_path = cache_path
        self._dirs = shortcut_dirs or default_shortcut_dirs
        self._steam = steam_path or default_steam_path
        self._start_apps = start_apps or default_start_apps
        self.apps: list[dict] = []
        self.ready = threading.Event()
        self._lock = threading.Lock()

    def build(self) -> int:
        found = (scan_steam(self._steam()) + scan_shortcuts(self._dirs())
                 + scan_start_apps(self._start_apps()))
        seen, apps = set(), []
        for a in found:  # dedupe by normalized name, first source wins
            k = _app_norm(a["name"])
            if k and k not in seen:
                seen.add(k)
                apps.append(a)
        with self._lock:
            self.apps = apps
        self.ready.set()
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps({"time": time.time(), "apps": apps}, indent=1))
        except OSError as e:
            log.warning("Could not write app cache: %s", e)
        print(f"[apps] indexed {len(apps)} apps")
        return len(apps)

    def load_cache(self) -> bool:
        """Load cache; return True if it is fresh (<24 h)."""
        try:
            data = json.loads(self.cache_path.read_text())
            with self._lock:
                self.apps = list(data["apps"])
            self.ready.set()
            print(f"[apps] indexed {len(self.apps)} apps (cached)")
            return time.time() - float(data["time"]) < APP_CACHE_MAX_AGE
        except Exception:
            return False

    def start(self) -> None:
        """Only with consent: use cache if present; (re)build in background when missing/stale."""
        if get_scan_consent() is not True:
            self.ready.set()  # nothing to wait for; only the whitelist works
            return
        if not self.load_cache():
            self.rescan_async()

    def forget(self) -> None:
        with self._lock:
            self.apps = []
        self.ready.set()
        try:
            self.cache_path.unlink(missing_ok=True)
        except OSError as e:
            log.warning("Could not delete app cache: %s", e)

    def rescan_async(self, on_done: Callable[[int], None] | None = None) -> None:
        self.ready.clear()

        def run():
            try:
                n = self.build()
                if on_done:
                    on_done(n)
            except Exception as e:
                log.warning("App scan failed: %s", e)
                self.ready.set()
        threading.Thread(target=run, daemon=True, name="app-index").start()

    @staticmethod
    def score(query: str, name: str) -> float:
        q, n = _app_norm(query), _app_norm(name)
        if not q or not n:
            return 0.0
        if q == n:
            return 1.0
        if n.startswith(q + " ") or n.startswith(q):
            return 0.92 if n.startswith(q + " ") else 0.85
        qt, nt = set(q.split()), set(n.split())
        if qt <= nt:
            return 0.8
        overlap = len(qt & nt) / len(qt | nt)
        return max(difflib.SequenceMatcher(None, q, n).ratio(), overlap * 0.9)

    def search(self, query: str, limit: int = 3) -> list[tuple[float, dict]]:
        with self._lock:
            ranked = sorted(((self.score(query, a["name"]), a) for a in self.apps),
                            key=lambda x: -x[0])
        return ranked[:limit]


APP_INDEX = AppIndex()
APP_MATCH_THRESHOLD = 0.72


def _launch(kind: str, target: str) -> None:
    """Launch a trusted target (from whitelist or index). Never a shell command."""
    if kind == "uwp":
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{target}"])
    elif hasattr(os, "startfile"):
        os.startfile(target)  # type: ignore[attr-defined]  # .lnk/.url/exe/URI
    else:
        subprocess.Popen(["xdg-open", target])


def open_app(name: str) -> str:
    key = _app_norm(name)
    key = re.sub(r"^(the )|( app)$", "", key).strip()
    # 1) curated whitelist / aliases
    for alias, target in APP_WHITELIST.items():
        if _app_norm(alias) == key:
            try:
                _launch("whitelist", target)
                return f"Opened {alias}"
            except OSError as e:
                return f"Failed to open {alias}: {e}"
    # 2) discovered apps (only if the user opted in)
    if get_scan_consent() is not True:
        return (f"'{name}' isn't one of my built-in apps. Tell the user that if they say "
                "'scan my apps', you can find and open their installed apps.")
    if not APP_INDEX.ready.wait(timeout=15):
        return "I'm still indexing installed apps; try again in a moment."
    matches = APP_INDEX.search(key or name)
    if not matches or matches[0][0] < APP_MATCH_THRESHOLD:
        sugg = [a["name"] for sc, a in matches if sc >= 0.45]
        if not sugg:
            return f"No installed app matching '{name}' found."
        return f"No exact match for '{name}'. Did you mean: {', '.join(sugg)}? Ask the user which."
    app = matches[0][1]
    try:
        _launch(app["kind"], app["target"])
        return f"Opened {app['name']}"
    except OSError as e:
        return f"Failed to open {app['name']}: {e}"


def list_apps(filter: str = "") -> str:  # noqa: A002 (tool arg name)
    APP_INDEX.ready.wait(timeout=15)
    if filter.strip():
        names = [a["name"] for sc, a in APP_INDEX.search(filter, limit=20) if sc >= 0.5]
    else:
        names = sorted(a["name"] for a in APP_INDEX.apps)[:20]
    curated = [k for k in APP_WHITELIST if not filter or _app_norm(filter) in _app_norm(k)]
    names = list(dict.fromkeys(curated + names))[:20]
    return ", ".join(names) if names else "No matching apps."


def _announce_count(n: int) -> None:
    _speaker.say(T(f"Done, {{t}}. I found {n} apps I can open for you."))


def rescan_apps() -> str:
    """User asked to (re)scan: this counts as consent."""
    set_scan_consent(True)
    APP_INDEX.rescan_async(on_done=_announce_count)
    return "Scanning installed apps in the background; I'll announce when done."


def forget_apps() -> str:
    set_scan_consent(False)
    APP_INDEX.forget()
    return CONSENT_DECLINED.replace("No problem", "Done")


_YES_RE = re.compile(r"\b(yes|yeah|yep|yup|sure|ok|okay|please|go ahead|do it|of course|absolutely)\b", re.I)
_NO_RE = re.compile(r"\b(no|nope|nah|don't|do not|not now|never)\b", re.I)


def ask_scan_consent(speaker: Any, get_reply: Callable[[], str]) -> bool | None:
    """First-run question. Yes -> consent + background scan; no/silence -> declined."""
    if get_scan_consent() is not None:
        return get_scan_consent()
    speaker.say(CONSENT_QUESTION)
    try:
        reply = (get_reply() or "").strip()
    except Exception:
        reply = ""
    print(f"You: {reply}" if reply else "[no reply]")
    if reply and _YES_RE.search(reply) and not _NO_RE.search(reply):
        set_scan_consent(True)
        speaker.say(T("Very good, {t}. Scanning now; I'll let you know when I'm done."))
        APP_INDEX.rescan_async(on_done=_announce_count)
        return True
    set_scan_consent(False)
    speaker.say(CONSENT_DECLINED)
    return False


# ---------------------------------------------------------------------------
# close_app: graceful (WM_CLOSE) close of running apps resolved from a spoken name.
# Only acts on resolved process names/windows; never runs model-provided commands.
# ---------------------------------------------------------------------------

CLOSE_ALIASES: dict[str, list[str]] = {
    "steam": ["steam.exe"],
    "spotify": ["spotify.exe"],
    "discord": ["discord.exe"],
    "notepad": ["notepad.exe"],
    "calculator": ["calculatorapp.exe", "calc.exe"],
    "calc": ["calculatorapp.exe", "calc.exe"],
}
BROWSER_TAB_TITLES: dict[str, str] = {"netflix": "Netflix", "prime video": "Prime Video",
                                      "prime": "Prime Video"}
PROTECTED_EXES = {
    "explorer.exe", "csrss.exe", "winlogon.exe", "svchost.exe", "dwm.exe", "lsass.exe",
    "services.exe", "wininit.exe", "smss.exe", "system", "system idle process", "registry",
    "msmpeng.exe", "lm studio.exe", "lmstudio.exe", "lms.exe", "python.exe", "pythonw.exe", "py.exe",
}
CLOSE_WAIT_SECONDS = 3.0


def is_protected(exe: str, pid: int) -> bool:
    e = exe.strip().lower()
    return (e in PROTECTED_EXES or e.startswith("norton") or "lm studio" in e or e.startswith("lm-studio")
            or pid in (0, 4, os.getpid(), os.getppid()))


class WinProcBackend:
    """Windows process/window access via tasklist + ctypes (no psutil)."""

    WM_CLOSE = 0x0010

    def processes(self) -> list[tuple[int, str]]:
        import csv
        import io
        try:
            out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True,
                                 timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        except Exception as e:
            log.warning("tasklist failed: %s", e)
            return []
        procs = []
        for row in csv.reader(io.StringIO(out)):
            if len(row) >= 2 and row[1].isdigit():
                procs.append((int(row[1]), row[0]))
        return procs

    def windows(self) -> list[tuple[int, int, str]]:
        """Visible top-level windows with a title: (hwnd, pid, title)."""
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        found: list[tuple[int, int, str]] = []
        proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def cb(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                n = user32.GetWindowTextLengthW(hwnd)
                if n:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(hwnd, buf, n + 1)
                    pid = wintypes.DWORD()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    found.append((int(hwnd), int(pid.value), buf.value))
            return True

        user32.EnumWindows(proto(cb), 0)
        return found

    def post_close(self, hwnd: int) -> None:
        import ctypes
        ctypes.windll.user32.PostMessageW(hwnd, self.WM_CLOSE, 0, 0)  # type: ignore[attr-defined]

    def force_kill(self, pid: int) -> None:
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/F"], capture_output=True, timeout=15)

    def quit_tree(self, exe: str) -> None:
        """End every process of an (unprotected) app plus its children, e.g. tray apps."""
        if is_protected(exe, -1):
            return
        subprocess.run(["taskkill", "/IM", exe, "/T", "/F"], capture_output=True, timeout=15,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def steam_exit(self) -> None:
        os.startfile("steam://exit")  # type: ignore[attr-defined]

    _lnk_cache: dict[str, str] = {}

    def lnk_target(self, lnk: str) -> str:
        """Read-only .lnk TargetPath lookup (cached)."""
        if lnk in self._lnk_cache:
            return self._lnk_cache[lnk]
        target = ""
        if lnk.lower().endswith(".lnk"):
            try:
                env = dict(os.environ, JARVIS_LNK=lnk)  # pass path via env, not via command text
                target = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(New-Object -ComObject WScript.Shell).CreateShortcut($env:JARVIS_LNK).TargetPath"],
                    capture_output=True, text=True, timeout=15, env=env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
            except Exception as e:
                log.warning("lnk lookup failed: %s", e)
        self._lnk_cache[lnk] = target
        return target


PROC_BACKEND: Any = WinProcBackend()
_last_user_text = ""  # set by Brain.ask; gates force close


def _user_said_force() -> bool:
    return bool(re.search(r"\bforce\b", _last_user_text, re.I))


def _resolve_exes(key: str) -> set[str]:
    """Exe names known for a spoken app name (aliases, whitelist, app index .lnk targets)."""
    exes = set(CLOSE_ALIASES.get(key, []))
    tgt = APP_WHITELIST.get(key, "")
    if tgt.lower().endswith(".exe"):
        exes.add(tgt.lower())
    if not exes and get_scan_consent() is True and APP_INDEX.apps:
        best = APP_INDEX.search(key, limit=1)
        if best and best[0][0] >= APP_MATCH_THRESHOLD and best[0][1]["kind"] == "shortcut":
            t = PROC_BACKEND.lnk_target(best[0][1]["target"])
            if t.lower().endswith(".exe"):
                exes.add(Path(t.replace("\\", "/")).name.lower())
    return exes


def close_app(name: str, force: bool = False) -> str:
    key = re.sub(r"^(the )|( app)$", "", _app_norm(name)).strip()
    if not key:
        return "Which app should I close?"
    if force and not _user_said_force():
        return "I'll only force close if the user explicitly says 'force close'. Try a normal close first."
    procs = PROC_BACKEND.processes()
    exe_of = {pid: exe for pid, exe in procs}
    wins = PROC_BACKEND.windows()

    # Browser tabs (Netflix / Prime Video): close only the matching window.
    if key in BROWSER_TAB_TITLES:
        needle = BROWSER_TAB_TITLES[key].lower()
        hits = [(h, pid, t) for h, pid, t in wins
                if needle in t.lower() and not is_protected(exe_of.get(pid, ""), pid)]
        if not hits:
            return f"I don't see a {BROWSER_TAB_TITLES[key]} window open."
        for h, _, _ in hits:
            PROC_BACKEND.post_close(h)
        return f"Closed the {BROWSER_TAB_TITLES[key]} window."

    # Resolve target processes.
    exes = _resolve_exes(key)
    if exes:
        pids = [pid for pid, exe in procs if exe.lower() in exes]
    else:  # fuzzy: exe stem and window titles
        pids = [pid for pid, exe in procs if AppIndex.score(key, Path(exe).stem) >= 0.8]
        pids += [pid for _, pid, t in wins
                 if AppIndex.score(key, t) >= 0.8 or (len(key) >= 4 and key in _app_norm(t))]
    pids = sorted(set(pids))
    blocked = [exe_of.get(p, "?") for p in pids if is_protected(exe_of.get(p, ""), p)]
    pids = [p for p in pids if not is_protected(exe_of.get(p, ""), p)]
    if not pids:
        if blocked:
            return f"I'm not allowed to close {blocked[0]}; it's protected."
        return f"I don't see {name} running."
    apps = sorted({exe_of.get(p, "?").lower() for p in pids})
    if len(apps) > 1:
        return f"Several apps match '{name}': {', '.join(apps)}. Ask the user which one to close."
    app = apps[0]

    if force:
        for p in pids:
            PROC_BACKEND.force_kill(p)
        return f"Force closed {app}."
    watch = set(pids)
    if key == "steam" or app == "steam.exe":
        PROC_BACKEND.steam_exit()
    else:
        # Multi-process apps (Spotify, Discord, browsers) have helper processes without
        # windows; success = the processes that owned windows are gone.
        watch = {pid for _, pid, _ in wins if pid in pids}
        if not watch:
            return (f"{app} has no open window to close (it may be in the system tray). "
                    f"Tell the user they can say 'force close {name}' to end it.")
        for h, pid, _ in wins:
            if pid in watch:
                PROC_BACKEND.post_close(h)
    deadline = time.time() + CLOSE_WAIT_SECONDS
    while time.time() < deadline:
        running = {pid for pid, _ in PROC_BACKEND.processes()}
        if not running & watch:
            return f"Closed {app}."
        # Apps like Discord, Spotify and Steam hide to the system tray when their window
        # is closed. If the window is gone but the process is still alive, quit it fully.
        win_pids = {pid for _, pid, _ in PROC_BACKEND.windows()}
        if not (win_pids & watch) and app != "steam.exe":
            time.sleep(0.5)
            PROC_BACKEND.quit_tree(app)
            time.sleep(1.0)
            running = {pid for pid, _ in PROC_BACKEND.processes()}
            if not running & watch:
                return f"Closed {app}."
            break
        time.sleep(0.3)
    return (f"{app} didn't close (it may be asking to save). Tell the user they can say "
            f"'force close {name}' to end it.")


_timers: list[threading.Timer] = []


def set_timer(seconds: float, label: str = "timer") -> str:
    seconds = float(seconds)
    if seconds <= 0 or seconds > 24 * 3600:
        return "Timer must be between 1 second and 24 hours."
    label = (label or "timer").strip()

    def _done() -> None:
        _speaker.say(T(f"{{T}}, your {label} is done."))

    t = threading.Timer(seconds, _done)
    t.daemon = True
    t.start()
    _timers.append(t)
    return f"Timer '{label}' set for {int(seconds)} seconds."


# ---------------------------------------------------------------------------
# Media / volume: Windows virtual media keys via user32.keybd_event (no extra deps).
# Works with whatever app owns media keys (Spotify, browser, etc.) and the system volume.
# ---------------------------------------------------------------------------

VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_VOLUME_UP = 0xAF
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_MUTE = 0xAD
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
VOLUME_STEP_PERCENT = 2  # one Windows volume key press = 2%
MEDIA_UNAVAILABLE = "Media keys only work on Windows."
_key_delay = 0.01  # seconds between presses (tests set 0)


def _user32():
    """user32 or None off Windows. Separate function so tests can stub it."""
    try:
        import ctypes

        return ctypes.windll.user32  # type: ignore[attr-defined]
    except Exception:
        return None


def press_key(vk: int, times: int = 1) -> bool:
    u = _user32()
    if u is None:
        return False
    for i in range(max(0, int(times))):
        u.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY, 0)
        u.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
        if _key_delay and i + 1 < times:
            time.sleep(_key_delay)
    return True


def _steps(v: Any, default: int = 5) -> int:
    try:
        return max(1, min(50, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


def media_control(action: str) -> str:
    action = (action or "").lower().replace("-", "_").replace(" ", "_")
    keys = {"play_pause": (VK_MEDIA_PLAY_PAUSE, "Done."), "play": (VK_MEDIA_PLAY_PAUSE, "Done."),
            "pause": (VK_MEDIA_PLAY_PAUSE, "Done."), "next": (VK_MEDIA_NEXT_TRACK, "Next track."),
            "previous": (VK_MEDIA_PREV_TRACK, "Previous track."), "prev": (VK_MEDIA_PREV_TRACK, "Previous track.")}
    if action not in keys:
        return "Action must be play_pause, next or previous."
    vk, msg = keys[action]
    return msg if press_key(vk) else MEDIA_UNAVAILABLE


QUIET_MEDIA = _env("JARVIS_QUIET_MEDIA", "1").strip().lower() not in ("0", "false", "no", "off")

_QUICK_MEDIA = [
    (r"\b(next|skip)\b.*\b(song|track|tune)\b|^\W*(next|skip)( it| this)?\W*$", "next"),
    (r"\b(previous|last|go back)\b.*\b(song|track|tune)\b", "previous"),
    (r"\b(pause|resume|unpause)\b.*\b(music|song|track|spotify|it)\b|^\W*(pause|resume|unpause)\W*$", "play_pause"),
    (r"\b(play)\b.*\b(music|song|track)\b$", "play_pause"),
]


def _quick_media(text: str) -> str | None:
    """Handle simple song commands instantly, skipping the LLM. Returns '' (silent) or a spoken
    message, or None if the text isn't a simple media command."""
    t = re.sub(r"\b(hey|ok|okay|please|can you|could you|jarvis|" + re.escape(ASSISTANT_NAME.lower()) + r")\b",
               "", (text or "").lower()).strip(" ,.!?")
    if len(t.split()) > 6:
        return None
    for pat, action in _QUICK_MEDIA:
        if re.search(pat, t):
            result = media_control(action)
            if result == MEDIA_UNAVAILABLE:
                return result
            return "" if QUIET_MEDIA else result
    return None


def volume_control(action: str, steps: Any = 5) -> str:
    """action up/down (steps presses, 2% each) or mute (toggle)."""
    action = (action or "").lower()
    if action == "mute":
        return "Mute toggled." if press_key(VK_VOLUME_MUTE) else MEDIA_UNAVAILABLE
    if action not in ("up", "down"):
        return "Action must be up, down or mute."
    n = _steps(steps)
    ok = press_key(VK_VOLUME_UP if action == "up" else VK_VOLUME_DOWN, n)
    return f"Volume {action}." if ok else MEDIA_UNAVAILABLE


def set_volume(percent: Any) -> str:
    """Approximate: 50 down presses (to 0%), then percent/2 up presses. Each press is 2%,
    so odd values round down to an even percent. Unmutes as a side effect of volume-up on most PCs."""
    try:
        pct = max(0, min(100, int(round(float(percent)))))
    except (TypeError, ValueError):
        return "Percent must be a number from 0 to 100."
    if not press_key(VK_VOLUME_DOWN, 100 // VOLUME_STEP_PERCENT):
        return MEDIA_UNAVAILABLE
    press_key(VK_VOLUME_UP, pct // VOLUME_STEP_PERCENT)
    return f"Volume about {pct // VOLUME_STEP_PERCENT * VOLUME_STEP_PERCENT} percent."


# ---------------------------------------------------------------------------
# OBS Studio control (OBS WebSocket v5 via obsws-python; built into OBS 28+)
# ---------------------------------------------------------------------------

OBS_HOST = _env("OBS_HOST", "localhost")
OBS_PORT = int(_env_float("OBS_PORT", 4455))
OBS_PASSWORD = _env("OBS_PASSWORD")
OBS_MIC_INPUT = _env("OBS_MIC_INPUT", "Mic/Aux")
OBS_TIMEOUT = _env_float("OBS_TIMEOUT", 3.0)
_OBS_UNAVAILABLE_TEMPLATE = ("I can't reach OBS, {t}. Make sure OBS is running and the WebSocket server is on, "
                             "under Tools, WebSocket Server Settings.")
OBS_UNAVAILABLE = T(_OBS_UNAVAILABLE_TEMPLATE)
_GO_LIVE_RE = re.compile(r"\b(go(ing)? live|start(ing)? (the |my )?stream(ing)?|begin (the |my )?stream"
                         r"|stream now|start broadcasting)\b", re.I)


class ObsUnavailable(Exception):
    pass


def _obs_client():
    """Connect lazily, per call, with a short timeout."""
    try:
        import obsws_python as obs
    except ImportError as e:
        raise ObsUnavailable("OBS control needs the obsws-python package; run pip install obsws-python.") from e
    try:
        return obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD or "", timeout=OBS_TIMEOUT)
    except Exception as e:  # refused, timeout, auth failure...
        log.warning("OBS connect failed: %s", e)
        if "auth" in str(e).lower() or "password" in str(e).lower():
            raise ObsUnavailable(T("OBS rejected the password, {t}. Check OBS_PASSWORD in the .env file.")) from e
        raise ObsUnavailable(OBS_UNAVAILABLE) from e


def _obs_call(fn: Callable[[Any], str]) -> str:
    try:
        cl = _obs_client()
    except ObsUnavailable as e:
        return str(e)
    try:
        return fn(cl)
    except Exception as e:
        log.warning("OBS request failed: %s", e)
        return f"OBS couldn't do that: {e}"
    finally:
        try:
            cl.disconnect()
        except Exception:
            pass


def obs_record(action: str = "start") -> str:
    action = (action or "start").lower()

    def run(cl):
        active = cl.get_record_status().output_active
        if action == "start":
            if active:
                return "Already recording."
            cl.start_record()
            return "Recording started."
        if action == "stop":
            if not active:
                return "Not recording right now."
            cl.stop_record()
            return "Recording stopped."
        return "Action must be start or stop."
    return _obs_call(run)


def obs_stream(action: str = "start") -> str:
    action = (action or "start").lower()
    if action == "start" and not _GO_LIVE_RE.search(_last_user_text):
        return ("Not going live: the user didn't clearly ask to start streaming. "
                "Ask them to confirm by saying 'go live' or 'start streaming'.")

    def run(cl):
        active = cl.get_stream_status().output_active
        if action == "start":
            if active:
                return "You're already live."
            cl.start_stream()
            return T("You're live, {t}.")
        if action == "stop":
            if not active:
                return "You're not streaming."
            cl.stop_stream()
            return "Stream ended."
        return "Action must be start or stop."
    return _obs_call(run)


def _scene_names(cl) -> list[str]:
    return [s.get("sceneName", "") for s in (cl.get_scene_list().scenes or [])]


def match_scene(query: str, names: list[str]) -> str | None:
    q = _app_norm(re.sub(r"\bscene\b", "", query or "", flags=re.I))
    norm = {_app_norm(n): n for n in names}
    if q in norm:
        return norm[q]
    subs = [n for k, n in norm.items() if q and (q in k or k in q) and k]
    if len(subs) == 1:
        return subs[0]
    best = difflib.get_close_matches(q, list(norm), n=1, cutoff=0.6)
    return norm[best[0]] if best else None


def obs_switch_scene(scene: str) -> str:
    def run(cl):
        names = _scene_names(cl)
        hit = match_scene(scene, names)
        if not hit:
            return f"No scene like '{scene}'. Scenes are: {', '.join(names) or 'none'}."
        cl.set_current_program_scene(hit)
        return f"Switched to {hit}."
    return _obs_call(run)


def obs_list_scenes() -> str:
    def run(cl):
        names = _scene_names(cl)
        cur = cl.get_current_program_scene().current_program_scene_name
        return f"Scenes: {', '.join(names) or 'none'}. Current: {cur}."
    return _obs_call(run)


def obs_mic(action: str = "toggle", input_name: str = "") -> str:
    name = (input_name or OBS_MIC_INPUT).strip()
    action = (action or "toggle").lower()

    def run(cl):
        if action == "mute":
            muted = True
        elif action == "unmute":
            muted = False
        else:
            muted = not cl.get_input_mute(name).input_muted
        cl.set_input_mute(name, muted)
        return f"{name} {'muted' if muted else 'unmuted'}."
    return _obs_call(run)


def obs_save_replay() -> str:
    def run(cl):
        if not cl.get_replay_buffer_status().output_active:
            return "The replay buffer isn't running in OBS, so I can't clip that."
        cl.save_replay_buffer()
        return "Clipped it."
    return _obs_call(run)


def obs_status() -> str:
    def run(cl):
        live = cl.get_stream_status().output_active
        rec = cl.get_record_status().output_active
        return f"You're {'live' if live else 'not live'} and {'recording' if rec else 'not recording'}."
    return _obs_call(run)


SKILLS: dict[str, Callable[..., str]] = {
    "get_time": get_time,
    "open_website": open_website,
    "web_search": web_search,
    "open_app": open_app,
    "list_apps": list_apps,
    "close_app": close_app,
    "rescan_apps": rescan_apps,
    "set_timer": set_timer,
    "obs_record": obs_record,
    "obs_stream": obs_stream,
    "obs_switch_scene": obs_switch_scene,
    "obs_list_scenes": obs_list_scenes,
    "obs_mic": obs_mic,
    "obs_save_replay": obs_save_replay,
    "obs_status": obs_status,
    "media_control": media_control,
    "volume_control": volume_control,
    "set_volume": set_volume,
}


def _fn(name: str, desc: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": required or []},
        },
    }


TOOLS: list[dict] = [
    _fn("get_time", "Get the current local date and time."),
    _fn("open_website", "Open a website in the browser.",
        {"url": {"type": "string", "description": "URL or domain, e.g. youtube.com"}}, ["url"]),
    _fn("web_search", "Search the web in the browser.",
        {"query": {"type": "string"}}, ["query"]),
    _fn("open_app", "Open any installed app or Steam game by name (e.g. notepad, calculator, spotify, "
        "steam, netflix, prime video, discord, a game title). If it returns suggestions, ask the user.",
        {"name": {"type": "string", "description": "App or game name as spoken"}}, ["name"]),
    _fn("close_app", "Close a running app or game by name (gracefully). Set force=true ONLY if the "
        "user explicitly said 'force close'.",
        {"name": {"type": "string"}, "force": {"type": "boolean", "description": "Only for 'force close'"}},
        ["name"]),
    _fn("list_apps", "List up to 20 installed apps matching a filter.",
        {"filter": {"type": "string", "description": "Optional name filter"}}),
    _fn("rescan_apps", "Rescan installed apps and games (when the user installed something new)."),
    _fn("set_timer", "Set a countdown timer that announces when done.",
        {"seconds": {"type": "number"}, "label": {"type": "string"}}, ["seconds"]),
    _fn("obs_record", "Start or stop OBS recording.",
        {"action": {"type": "string", "enum": ["start", "stop"]}}, ["action"]),
    _fn("obs_stream", "Start or stop OBS streaming. Use action=start ONLY when the user clearly said to go "
        "live or start streaming; never guess.",
        {"action": {"type": "string", "enum": ["start", "stop"]}}, ["action"]),
    _fn("obs_switch_scene", "Switch the OBS scene (fuzzy name match).",
        {"scene": {"type": "string", "description": "Scene name as spoken"}}, ["scene"]),
    _fn("obs_list_scenes", "List OBS scenes and the current one."),
    _fn("obs_mic", "Mute, unmute or toggle the user's mic (an OBS audio input).",
        {"action": {"type": "string", "enum": ["mute", "unmute", "toggle"]},
         "input_name": {"type": "string", "description": "Optional OBS input name; default is the mic"}},
        ["action"]),
    _fn("obs_save_replay", "Save the OBS replay buffer ('clip that', 'save replay')."),
    _fn("obs_status", "Say whether OBS is currently streaming (live) and/or recording."),
    _fn("media_control", "Control media playback (Spotify, browser, etc.): play/pause toggle, next or "
        "previous track.",
        {"action": {"type": "string", "enum": ["play_pause", "next", "previous"]}}, ["action"]),
    _fn("volume_control", "Turn the system volume up or down, or toggle mute. Each step is about 2 percent; "
        "default 5 steps. Use more steps for 'a lot', fewer for 'a little'.",
        {"action": {"type": "string", "enum": ["up", "down", "mute"]},
         "steps": {"type": "integer", "description": "1-50 key presses, default 5"}}, ["action"]),
    _fn("set_volume", "Set the system volume to a percent (0-100), e.g. 'volume 40 percent'.",
        {"percent": {"type": "number"}}, ["percent"]),
]


def list_wake_sounds() -> str:
    cur = WAKE_SOUND_STYLE
    parts = [f"{c}: " + ", ".join(n.replace("_", " ") for n, _, cc in WAKE_SOUND_STYLES if cc == c)
             for c in WAKE_SOUND_CATEGORIES]
    return f"Current: {cur.replace('_', ' ')}. " + ". ".join(parts) + ". Or random."


def play_wake_style(name: str, sd: Any = None) -> None:
    """Play one built-in style once (ignores WAKE_SOUND_FILE); never raises."""
    try:
        if sd is None:
            import sounddevice as sd
        tone = make_chime(WAKE_SOUND_VOLUME, style=name)
        sd.play(tone, CHIME_RATE)
        sd.wait()
    except Exception as e:
        log.warning("Could not play wake sound: %s", e)


def preview_wake_sound(name: str = "") -> str:
    st = match_wake_sound(name)
    if st is None:
        return f"I don't know a sound called {name}. Say list wake sounds for options."
    play_wake_style(st)
    return f"That was {wake_label(st).lower()}." if st != "random" else "That was a random one."


def set_wake_sound(name: str = "") -> str:
    st = match_wake_sound(name)
    if st is None:
        return f"I don't know a sound called {name}. Say list wake sounds for options."
    save_wake_style(st)
    play_wake_style(st)
    return f"Wake sound set to {st.replace('_', ' ')}."


TOOLS += [
    _fn("list_wake_sounds", "List the available wake sounds by category, and the current one."),
    _fn("preview_wake_sound", "Play a wake sound once without changing it.",
        {"name": {"type": "string", "description": "Sound name as spoken, e.g. piano, steel drum"}}, ["name"]),
    _fn("set_wake_sound", "Change (and save) the wake sound; 'random' picks a different one each wake.",
        {"name": {"type": "string", "description": "Sound name as spoken, or random"}}, ["name"]),
]
SKILLS.update(list_wake_sounds=list_wake_sounds, preview_wake_sound=preview_wake_sound,
              set_wake_sound=set_wake_sound)

_QUICK_SOUND_SET = re.compile(
    r"^(?:change|set|switch|make)\s+(?:your|the|my|wake)\s+(?:wake\s+)?(?:sound|chime)\s+(?:to|into)\s+(.+)$"
    r"|^use\s+(?:the\s+|a\s+)?(.+?)\s+(?:wake\s+)?(?:sound|chime)$", re.I)
_QUICK_SOUND_PREVIEW = re.compile(
    r"^(?:play|preview)\s+(?:me\s+)?(?:the\s+|a\s+)?(.+?)\s+(?:wake\s+)?(?:sound|chime)$", re.I)
_QUICK_SOUND_LIST = re.compile(r"\b(list|what|which)\b.*\b(wake|chime)\s*sounds?\b|^wake sounds$", re.I)


def _quick_sound(text: str) -> str | None:
    """Fast wake-sound commands (skip the LLM). None if not one."""
    t = re.sub(r"\b(hey|ok|okay|please|can you|could you|jarvis|" + re.escape(ASSISTANT_NAME.lower()) + r")\b",
               "", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip(" ,.!?")
    m = _QUICK_SOUND_SET.match(t)
    if m:
        name = m.group(1) or m.group(2)
        return set_wake_sound(name) if match_wake_sound(name) else None
    m = _QUICK_SOUND_PREVIEW.match(t)
    if m and match_wake_sound(m.group(1)):
        return preview_wake_sound(m.group(1))
    if _QUICK_SOUND_LIST.search(t):
        return list_wake_sounds()
    return None


def dispatch_tool(name: str, raw_args: str | dict | None) -> str:
    """Run a skill by name with JSON args; never raises."""
    fn = SKILLS.get(name)
    if fn is None:
        return f"Unknown tool '{name}'."
    try:
        args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            args = {}
        return str(fn(**args))
    except Exception as e:
        log.warning("Tool %s failed: %s", name, e)
        return f"Tool {name} failed: {e}"


# ---------------------------------------------------------------------------
# LLM brain
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)


def clean_reply(text: str | None) -> str:
    """Strip Qwen3 <think> blocks and stray markdown so it speaks nicely."""
    t = _THINK_RE.sub("", text or "")
    t = re.sub(r"[*_`#>]+", "", t)
    return re.sub(r"\s+", " ", t).strip()


class Brain:
    def __init__(self, client: Any = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=120)
        self.client = client
        self.history: list[dict] = []

    def _trim(self) -> None:
        # Keep the last N user turns (and everything after each, incl. tool msgs).
        user_idx = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        if len(user_idx) > HISTORY_TURNS:
            self.history = self.history[user_idx[-HISTORY_TURNS]:]

    def _chat(self, messages: list[dict], use_tools: bool = True):
        kw: dict[str, Any] = {"model": LLM_MODEL, "messages": messages, "temperature": 0.6}
        if use_tools:
            kw["tools"] = TOOLS
            kw["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kw)

    def ask(self, user_text: str) -> str:
        global _last_user_text
        _last_user_text = user_text
        if re.search(r"\b(stop scanning|forget)\b.*\b(apps|applications|games|programs)\b",
                     user_text, re.I):
            return forget_apps()
        if re.search(r"\b(re-?scan|scan|refresh|re-?index)\b.*\b(apps|applications|games|programs)\b",
                     user_text, re.I):
            rescan_apps()
            return T("Scanning your apps now, {t}. I'll tell you when I'm done.")
        quick = _quick_sound(user_text)
        if quick is None:
            quick = _quick_media(user_text)
        if quick is not None:
            return quick
        content = user_text + (" /no_think" if LLM_NO_THINK else "")
        self.history.append({"role": "user", "content": content})
        self._trim()
        try:
            for _ in range(3):  # allow a couple of tool rounds
                msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
                msg = self._chat(msgs).choices[0].message
                calls = getattr(msg, "tool_calls", None) or []
                if not calls:  # plain answer
                    reply = add_closing(clean_reply(msg.content) or T("I'm afraid I have nothing to add, {t}."))
                    self.history.append({"role": "assistant", "content": reply})
                    return reply
                self.history.append({
                    "role": "assistant",
                    "content": clean_reply(msg.content) or None,
                    "tool_calls": [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                        for c in calls
                    ],
                })
                quiet = QUIET_MEDIA
                for c in calls:
                    result = dispatch_tool(c.function.name, c.function.arguments)
                    log.info("tool %s(%s) -> %s", c.function.name, c.function.arguments, result)
                    self.history.append({"role": "tool", "tool_call_id": c.id, "content": result})
                    if c.function.name != "media_control" or result == MEDIA_UNAVAILABLE \
                            or result.startswith("Action must"):
                        quiet = False
                if quiet:  # song changes happen silently
                    self.history.append({"role": "assistant", "content": "(done silently)"})
                    return ""
            # Too many tool rounds: ask for a final answer without tools.
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
            reply = clean_reply(self._chat(msgs, use_tools=False).choices[0].message.content)
            reply = add_closing(reply or T("Done, {t}."))
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            self.history.pop() if self.history and self.history[-1]["role"] == "user" else None
            return self._error_text(e)

    @staticmethod
    def _error_text(e: Exception) -> str:
        name = type(e).__name__
        log.warning("LLM error: %s: %s", name, e)
        if "Connection" in name or "Timeout" in name:
            return (T("I can't reach my brain, {t}. Please start the LM Studio server on "
                    f"{LLM_BASE_URL}."))
        if "NotFound" in name:
            return T(f"LM Studio doesn't have the model {LLM_MODEL} loaded, {{t}}.")
        return T("Something went wrong with my language model, {t}.")


# ---------------------------------------------------------------------------
# Audio: wake word, recording, transcription
# ---------------------------------------------------------------------------


class PauseState:
    """Shared paused flag, toggled by voice phrases or the 'p' key."""

    def __init__(self) -> None:
        self.paused = False

    def set(self, paused: bool) -> None:
        self.paused = paused
        print("[paused]" if paused else "[listening]")


def poll_keyboard(state: PauseState, speaker: Any = None, getkey: Callable[[], str | None] | None = None) -> None:
    """Non-blocking: 'p' in the console toggles pause. No-op off Windows."""
    if getkey is None:
        try:
            import msvcrt  # Windows only
        except ImportError:
            return
        getkey = lambda: msvcrt.getwch() if msvcrt.kbhit() else None  # noqa: E731
    while (k := getkey()) is not None:
        if k.lower() == "p":
            state.set(not state.paused)


def _norm(text: str) -> str:
    t = re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'"))
    return re.sub(r"\s+", " ", t).strip()


def contains_phrase(text: str, phrases: list[str]) -> bool:
    """Loose match: phrase appears as whole words anywhere (punctuation-insensitive)."""
    t = f" {_norm(text)} "
    return any(f" {_norm(p)} " in t for p in phrases if _norm(p))


def is_pause_phrase(text: str) -> bool:
    return contains_phrase(text, PAUSE_PHRASES)


def _is_cuda_error(e: BaseException) -> bool:
    m = str(e).lower()
    return any(k in m for k in ("cublas", "cudnn", "cuda"))


def _run_whisper(model, audio) -> str:
    # Segments are a lazy generator; the encoder (and CUDA libs) run while iterating.
    segs, _ = model.transcribe(audio, language=WHISPER_LANGUAGE, beam_size=1, vad_filter=False)
    return " ".join(s.text for s in segs).strip()


def load_whisper(device: str, model_cls: Any = None):
    """Load faster-whisper, warming up CUDA so missing DLLs fail here, not mid-chat."""
    if model_cls is None:
        from faster_whisper import WhisperModel as model_cls
    import numpy as np

    device = (device or "auto").lower()
    tries = ["cuda", "cpu"] if device == "auto" else [device, "cpu"]
    for dev in dict.fromkeys(tries):
        ct = "float16" if dev == "cuda" else "int8"
        try:
            m = model_cls(WHISPER_MODEL, device=dev, compute_type=ct)
            if dev != "cpu":
                _run_whisper(m, np.zeros(SAMPLE_RATE, dtype=np.float32))  # 1 s warm-up
            log.info("Whisper '%s' on %s (%s)", WHISPER_MODEL, dev, ct)
            return m
        except Exception as e:
            log.warning("Whisper on %s failed: %s", dev, e)
    raise RuntimeError("Could not load faster-whisper on any device.")


class Ears:
    def __init__(self) -> None:
        import numpy as np
        import openwakeword
        from openwakeword.model import Model

        self.np = np
        try:
            openwakeword.utils.download_models()  # no-op once cached
        except Exception as e:
            log.warning("openwakeword model download failed: %s", e)
        model = WAKE_MODEL
        if model.lower() not in BUILTIN_WAKE_MODELS:
            path = Path(model).expanduser()
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            if not path.is_file():
                raise FileNotFoundError(f"JARVIS_WAKE_MODEL: no built-in model or file named {model!r}")
            model = str(path)
        else:
            model = model.lower()
        log.info("wake model: %s (say %r)", model, WAKE_PHRASE)
        try:
            self.wake = Model(wakeword_models=[model], inference_framework="onnx")
        except TypeError:  # older/newer API variants
            self.wake = Model(wakeword_models=[model])
        self.whisper = self._load_whisper()

    @staticmethod
    def _load_whisper(force_cpu: bool = False):
        return load_whisper("cpu" if force_cpu else WHISPER_DEVICE)

    def _stream(self):
        import sounddevice as sd

        dev = int(INPUT_DEVICE) if INPUT_DEVICE and INPUT_DEVICE.isdigit() else INPUT_DEVICE
        return sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                              blocksize=FRAME_SAMPLES, device=dev)

    def wait_for_wake(self, speaking: threading.Event, poll: Callable[[], None] = lambda: None,
                      speaker: Any = None) -> None:
        """Block until the wake word. While `speaking` (e.g. a timer announcement from another
        thread) it ignores the mic, unless barge-in is on and a speaker is given: then it uses the
        stricter BARGE_IN_THRESHOLD and stops the speech on a hit."""
        self.wake.reset()
        was_speaking = False
        with self._stream() as s:
            while True:
                frame, _ = s.read(FRAME_SAMPLES)
                poll()  # keyboard pause toggle
                talking = speaking.is_set()
                if talking != was_speaking:
                    self.wake.reset()  # don't carry scores across speech boundaries
                    was_speaking = talking
                if talking and not (BARGE_IN and speaker is not None):
                    continue  # don't listen while talking
                scores = self.wake.predict(frame[:, 0])
                if wake_score(scores) >= (BARGE_IN_THRESHOLD if talking else WAKE_THRESHOLD):
                    self.wake.reset()
                    if talking:
                        print("[barge-in]")
                        speaker.stop()
                    return

    def watch_barge_in(self, speaker: Any, stop: threading.Event) -> bool:
        """Run in a thread while the speaker talks (main thread blocked in say()). Returns True
        and calls speaker.stop() if the wake word is heard above BARGE_IN_THRESHOLD."""
        self.wake.reset()
        try:
            with self._stream() as s:
                while not stop.is_set():
                    frame, _ = s.read(FRAME_SAMPLES)
                    if not speaker.speaking.is_set():
                        continue
                    if wake_score(self.wake.predict(frame[:, 0])) >= BARGE_IN_THRESHOLD:
                        print("[barge-in]")
                        speaker.stop()
                        return True
        except Exception as e:
            log.warning("barge-in watcher failed: %s", e)
        finally:
            self.wake.reset()
        return False

    def record_utterance(self, start_timeout: float = 3.0, poll: Callable[[], None] = lambda: None):
        """Record until SILENCE_SECONDS of quiet (after speech) or MAX_RECORD_SECONDS.

        Returns None if no speech starts within start_timeout seconds.
        """
        np = self.np
        frames, silent, heard = [], 0.0, False
        speech_start = 0.0
        frame_s = FRAME_SAMPLES / SAMPLE_RATE
        start = time.time()
        with self._stream() as s:
            while True:
                if heard and time.time() - speech_start >= MAX_RECORD_SECONDS:
                    break
                f, _ = s.read(FRAME_SAMPLES)
                poll()
                frames.append(f[:, 0].copy())
                rms = float(np.sqrt(np.mean(f.astype(np.float32) ** 2)))
                if rms >= VAD_RMS_THRESHOLD:
                    if not heard:
                        speech_start = time.time()
                    heard, silent = True, 0.0
                else:
                    silent += frame_s
                # Stop after trailing silence; give up early if nothing said in ~3 s.
                if (heard and silent >= SILENCE_SECONDS) or (not heard and silent >= start_timeout):
                    break
        if not heard:
            return None
        return np.concatenate(frames).astype(np.float32) / 32768.0

    def transcribe(self, audio) -> str:
        try:
            return _run_whisper(self.whisper, audio)
        except RuntimeError as e:
            if not _is_cuda_error(e):
                raise
            # CUDA libs failed lazily (e.g. cublas64_12.dll missing): reload on CPU, retry once.
            log.warning("Whisper CUDA failure (%s); reloading on cpu int8.", e)
            print("[whisper: CUDA failed, switching to CPU]")
            self.whisper = self._load_whisper(force_cpu=True)
            return _run_whisper(self.whisper, audio)


def soft_cue() -> None:
    """Quiet descending tone: 'going back to sleep'."""
    try:
        import numpy as np
        import sounddevice as sd

        t = np.linspace(0, 0.1, int(22050 * 0.1), False)
        tone = np.concatenate([np.sin(2 * np.pi * 660 * t), np.sin(2 * np.pi * 440 * t)])
        sd.play((0.12 * tone).astype(np.float32), 22050)
        sd.wait()
    except Exception:
        pass


def is_end_phrase(text: str) -> bool:
    """True if the utterance is basically just a 'we're done' phrase (never a pause phrase)."""
    if is_pause_phrase(text):
        return False
    t = re.sub(r"[^a-z' ]+", " ", text.lower()).strip()
    names = "|".join(re.escape(n.lower()) for n in {"jarvis", ASSISTANT_NAME, USER_TITLE or "sir"} if n)
    t = re.sub(rf"^(ok(ay)?|alright|no|{names})\s+", "", t)
    t = re.sub(rf"\s+({names}|sir|mate)$", "", t).strip()
    return any(t == p or (t.startswith(p) and len(t) <= len(p) + 12) for p in FOLLOWUP_END_PHRASES)


def say_with_barge_in(ears: Any, speaker: Any, text: str, enabled: bool | None = None) -> bool:
    """Speak `text`; if barge-in is on, watch for the wake word meanwhile.
    Returns True if the user interrupted (speech was stopped)."""
    enabled = BARGE_IN if enabled is None else enabled
    watch = getattr(ears, "watch_barge_in", None)
    if not enabled or watch is None or not hasattr(speaker, "stop"):
        speaker.say(text)
        return False
    stop = threading.Event()
    hit: list[bool] = []
    th = threading.Thread(target=lambda: hit.append(watch(speaker, stop)), daemon=True, name="barge-in")
    th.start()
    try:
        speaker.say(text)
    finally:
        stop.set()
        th.join(timeout=1.0)
    interrupted = getattr(speaker, "interrupted", None)
    return bool(hit and hit[0]) or bool(interrupted is not None and interrupted.is_set())


def resume_from_pause(speaker: Any, state: PauseState) -> None:
    """Wake word heard while paused: resume immediately."""
    speaker.say(RESUME_REPLY)
    state.set(False)  # prints [listening]


def conversation(ears: Any, brain: Any, speaker: Any, *, followup_seconds: float = FOLLOWUP_SECONDS,
                 cue: Callable[[], None] = soft_cue, state: PauseState | None = None,
                 poll: Callable[[], None] = lambda: None, listen_cue: Callable[[], None] = lambda: None,
                 wake_cue: Callable[[], None] = lambda: None, barge_in: bool | None = None) -> None:
    """One wake-word session: first request, then follow-ups until silence, end or pause phrase.
    If the user barges in (wake word while a reply is spoken), speech stops and it listens for
    a fresh request right away."""
    state = state or PauseState()
    first = True
    while True:
        if first:
            print("[listening]")
            audio = ears.record_utterance(poll=poll)
        else:
            print("[listening for follow-up]")
            listen_cue()
            audio = ears.record_utterance(start_timeout=followup_seconds, poll=poll)
        if state.paused:  # 'p' pressed mid-conversation
            return
        text = ears.transcribe(audio) if audio is not None else ""
        if not text:
            print("[heard nothing - back to wake word]")
            cue()
            return
        print(f"You: {text}")
        if is_pause_phrase(text):  # checked first; also exits follow-up
            speaker.say(PAUSE_REPLY)
            state.set(True)
            return
        if not first and is_end_phrase(text):
            speaker.say(T("Very good, {t}."))
            print("[follow-up ended - back to wake word]")
            return
        if say_with_barge_in(ears, speaker, brain.ask(text), barge_in):  # blocks while speaking
            wake_cue()
            first = True  # interrupted: treat like a fresh wake word
            continue
        first = False
        if followup_seconds <= 0:
            return


CHIME_RATE = 22050


def _instrument_chime(style: str, rate: int = CHIME_RATE):
    """Instrument wake sounds (pure numpy). Returns an un-normalized float array, or None if not an instrument."""
    import numpy as np

    rng = np.random.default_rng(11)
    tau = 2 * np.pi

    def _t(sec):
        return np.arange(int(rate * sec)) / rate

    def _adsr(n, a, d, s, r):
        t = np.arange(n) / rate
        e = np.where(t < a, t / max(a, 1e-6), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-6)))
        rel = int(rate * r)
        if rel and n > rel:
            e[-rel:] *= np.linspace(1, 0, rel) ** 2
        return e

    def _ks(freq, dur, damp=0.996, bright=1.0):
        # Karplus-Strong, vectorized one period at a time (so ~dur*freq numpy steps, not per-sample)
        n = int(rate * dur)
        per = max(2, int(round(rate / freq)))
        buf = rng.uniform(-1, 1, per)
        if bright < 1.0:
            buf = np.convolve(buf, np.ones(3) / 3, mode="same") * bright + buf * (1 - bright)
        blocks = []
        for _ in range(n // per + 1):
            blocks.append(buf.copy())
            buf = damp * 0.5 * (buf + np.roll(buf, -1))
        return np.concatenate(blocks)[:n]

    def _add(freq, t, amps, decays, inharm=0.0):
        return sum(a * np.sin(tau * freq * k * np.sqrt(1 + inharm * k * k) * t) * np.exp(-t * dk)
                   for k, (a, dk) in enumerate(zip(amps, decays), 1))

    def _lp(x, cutoff):
        # one-pole low-pass via exponential filter (scipy-free): IIR approximated with lfilter-like cumsum trick
        a = np.exp(-2 * np.pi * cutoff / rate)
        k = int(min(len(x), max(8, 6 / (1 - a))))
        h = (1 - a) * a ** np.arange(k)
        return np.convolve(x, h)[:len(x)]

    def _place(parts, total):
        out = np.zeros(int(rate * total))
        for st, sig in parts:
            s = int(rate * st)
            m = min(len(sig), len(out) - s)
            out[s:s + m] += sig[:m]
        return out

    if style == "piano":  # C5+G5 dyad, hammer harmonics with stretched partials
        t = _t(0.8)
        out = sum(_add(f, t, (1, .6, .35, .2, .12, .07), (5, 7, 9, 12, 15, 18), inharm=0.0004) for f in (523.25, 783.99))
        out = out * np.minimum(1, t / 0.003) + 0.15 * rng.uniform(-1, 1, len(t)) * np.exp(-t * 300)
    elif style == "acoustic_guitar":  # quick G-major down-strum
        notes = (98.0, 123.47, 146.83, 196.0, 246.94, 392.0)
        out = _place([(0.012 * i, _ks(f, 0.75, 0.996, 0.5) * np.exp(-_t(0.75) * 3)) for i, f in enumerate(notes)], 0.8)
        out = _lp(out, 4500)
    elif style == "harp":  # quick upward gliss on a C-major pentatonic
        notes = (261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25, 783.99, 880.0)
        out = _place([(0.035 * i, _add(f, _t(0.55), (1, .3, .1), (6, 10, 16)) * np.minimum(1, _t(0.55) / 0.002))
                      for i, f in enumerate(notes)], 0.9)
    elif style == "violin":  # bowed A5 with delayed vibrato and bow noise
        t = _t(0.7)
        vib = 1 + 0.006 * np.sin(tau * 5.8 * t) * np.minimum(1, t / 0.25)
        ph = tau * np.cumsum(880.0 * vib) / rate
        saw = sum(np.sin(k * ph) / k * (0.85 ** k) for k in range(1, 12))
        out = _lp(saw, 5000) * _adsr(len(t), 0.08, 0.2, 0.8, 0.15) + 0.03 * rng.uniform(-1, 1, len(t))
    elif style == "cello":  # warm bowed C3 with slow vibrato
        t = _t(0.85)
        vib = 1 + 0.005 * np.sin(tau * 5 * t) * np.minimum(1, t / 0.3)
        ph = tau * np.cumsum(130.81 * vib) / rate
        saw = sum(np.sin(k * ph) / k * (0.9 ** k) for k in range(1, 18))
        out = _lp(saw, 2200) * _adsr(len(t), 0.1, 0.25, 0.8, 0.2)
    elif style == "trumpet":  # two-note fanfare G4 -> C5, brassy
        def _brass(f, dur):
            t = _t(dur)
            env = _adsr(len(t), 0.03, 0.1, 0.75, 0.05)
            # brightness rises with loudness (brass)
            return sum(np.sin(tau * f * k * t) * env ** (1 + 0.4 * k) / k ** 0.6 for k in range(1, 12))
        out = _place([(0, _brass(392.0, 0.16)), (0.19, _brass(523.25, 0.42))], 0.62)
    elif style == "sax":  # breathy, reedy D5 with scoop and vibrato
        t = _t(0.6)
        f = 587.33 * (1 - 0.03 * np.exp(-t * 25)) * (1 + 0.007 * np.sin(tau * 5.5 * t))
        ph = tau * np.cumsum(f) / rate
        tone = sum(np.sin(k * ph) * (0.75 ** k if k % 2 else 0.6 * 0.75 ** k) for k in range(1, 10))
        breath = _lp(rng.uniform(-1, 1, len(t)), 3000) * 0.25
        out = np.tanh(1.5 * tone) * _adsr(len(t), 0.04, 0.15, 0.7, 0.12) + breath * _adsr(len(t), 0.02, 0.1, 0.4, 0.1)
    elif style == "flute":  # airy G5, nearly pure with breath
        t = _t(0.6)
        f = 783.99 * (1 + 0.004 * np.sin(tau * 5 * t) * np.minimum(1, t / 0.2))
        ph = tau * np.cumsum(f) / rate
        tone = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
        breath = (_lp(rng.uniform(-1, 1, len(t)), 6000) - _lp(rng.uniform(-1, 1, len(t)), 1500)) * 0.35
        out = (tone + breath * np.exp(-t * 4)) * _adsr(len(t), 0.06, 0.2, 0.75, 0.15)
    elif style == "clarinet":  # hollow odd-harmonic Bb4
        t = _t(0.6)
        ph = tau * 466.16 * t
        tone = sum(np.sin(k * ph) / k for k in (1, 3, 5, 7, 9, 11)) + 0.05 * np.sin(2 * ph)
        out = _lp(tone, 3500) * _adsr(len(t), 0.04, 0.15, 0.8, 0.15) + 0.02 * rng.uniform(-1, 1, len(t)) * np.exp(-t * 20)
    elif style == "marimba":  # woody FM mallets C5 -> G5
        def _mar(f):
            t = _t(0.5)
            mod = 1.8 * np.exp(-t * 40) * np.sin(tau * 4 * f * t)
            return (np.sin(tau * f * t + mod) + 0.25 * np.sin(tau * 3.93 * f * t) * np.exp(-t * 30)) * np.exp(-t * 9) \
                * np.minimum(1, t / 0.001)
        out = _place([(0, _mar(523.25)), (0.1, _mar(783.99))], 0.6)
    elif style == "xylophone":  # bright hard-mallet G6, E7
        def _xy(f):
            t = _t(0.3)
            return (np.sin(tau * f * t) + 0.5 * np.sin(tau * 3.0 * f * t) * np.exp(-t * 25)
                    + 0.2 * np.sin(tau * 6.27 * f * t) * np.exp(-t * 50)) * np.exp(-t * 14) * np.minimum(1, t / 0.0007)
        out = _place([(0, _xy(1567.98)), (0.08, _xy(2637.0))], 0.4)
    elif style == "vibraphone":  # mellow metal bars with motor tremolo
        t = _t(0.85)
        trem = 1 - 0.45 * (0.5 + 0.5 * np.sin(tau * 6 * t))
        out = sum((np.sin(tau * f * t) + 0.3 * np.sin(tau * 4.0 * f * t) * np.exp(-t * 12)) * np.exp(-t * 3.2)
                  for f in (523.25, 659.25, 783.99)) * trem * np.minimum(1, t / 0.002)
    elif style == "steel_drum":  # Caribbean pan, FM ring
        def _pan(f):
            t = _t(0.55)
            mod = 2.2 * np.exp(-t * 12) * np.sin(tau * 2 * f * t)
            return (np.sin(tau * f * t + mod) + 0.35 * np.sin(tau * 1.5 * f * t) * np.exp(-t * 8)) * np.exp(-t * 6) \
                * np.minimum(1, t / 0.002)
        out = _place([(0, _pan(659.25)), (0.11, _pan(880.0))], 0.66)
    elif style == "kalimba":  # thumb piano tines, plinky
        def _tine(f):
            t = _t(0.5)
            return (np.sin(tau * f * t) + 0.4 * np.sin(tau * 5.4 * f * t) * np.exp(-t * 60)
                    + 0.15 * np.sin(tau * 2 * f * t) * np.exp(-t * 20)) * np.exp(-t * 7) * np.minimum(1, t / 0.001)
        out = _place([(0, _tine(783.99)), (0.09, _tine(1046.5)), (0.18, _tine(1318.5))], 0.68)
    elif style == "music_box":  # tiny high comb notes, twinkly
        def _comb(f):
            t = _t(0.45)
            return (np.sin(tau * f * t) + 0.3 * np.sin(tau * 3.0 * f * t) * np.exp(-t * 15)
                    + 0.12 * np.sin(tau * 5.8 * f * t) * np.exp(-t * 40)) * np.exp(-t * 6) * np.minimum(1, t / 0.0005)
        out = _place([(0, _comb(2093.0)), (0.12, _comb(2637.0)), (0.24, _comb(3136.0))], 0.69)
    elif style == "banjo":  # twangy bright roll, G-D-g
        out = _place([(0.06 * i, _ks(f, 0.45, 0.99) * np.exp(-_t(0.45) * 6)) for i, f in enumerate((392.0, 587.33, 783.99))], 0.6)
        out = out - 0.6 * np.concatenate([np.zeros(12), out[:-12]])  # bright drum-head comb
    elif style == "sitar":  # buzzy jawari with sympathetic shimmer
        t = _t(0.85)
        s = _ks(277.18, 0.85, 0.998)
        buzz = np.tanh(6 * s) * 0.5 + s  # bridge buzz
        f = 1 + 0.02 * np.minimum(1, t / 0.3) ** 2  # slight meend (pitch bend)
        idx = np.clip(np.cumsum(f) - 1, 0, len(buzz) - 1)
        main = np.interp(idx, np.arange(len(buzz)), buzz) * np.exp(-t * 3.5)
        symp = sum(np.sin(tau * fr * t + i) for i, fr in enumerate((554.37, 830.6, 1108.7, 1661.2))) \
            * 0.12 * np.minimum(1, t / 0.15) * np.exp(-t * 2.5) * (1 + 0.3 * np.sin(tau * 9 * t))
        out = main + symp
    elif style == "synth_lead":  # retro saw lead, filter sweep
        def _saw(f, dur):
            t = _t(dur)
            ph = (f * t) % 1.0
            x = (2 * ph - 1) + (2 * ((f * 1.007 * t) % 1.0) - 1)
            # sweep filter: blend of dark and bright versions
            sweep = np.minimum(1, t / 0.08) * np.exp(-t * 4)
            return (_lp(x, 900) * (1 - sweep) + _lp(x, 6000) * sweep) * _adsr(len(t), 0.005, 0.1, 0.7, 0.04)
        out = _place([(0, _saw(440.0, 0.13)), (0.14, _saw(659.25, 0.35))], 0.5)
    elif style == "organ":  # drawbar 888000000-ish with key click
        t = _t(0.6)
        f = 392.0
        bars = ((0.5, .8), (1.5, .6), (1, 1.0), (2, .6), (3, .3), (4, .25), (6, .1))
        tone = sum(a * np.sin(tau * f * m * t) for m, a in bars) * (1 + 0.08 * np.sin(tau * 6.7 * t))
        click = rng.uniform(-1, 1, len(t)) * np.exp(-t * 400) * 0.3
        out = tone * _adsr(len(t), 0.005, 0.05, 1.0, 0.08) + click
    elif style == "taiko":  # deep drum hit with a small rim click
        t = _t(0.7)
        f = 55 + 70 * np.exp(-t * 30)
        body = np.sin(tau * np.cumsum(f) / rate) * np.exp(-t * 6) + 0.3 * np.sin(tau * 1.6 * np.cumsum(f) / rate) * np.exp(-t * 12)
        skin = _lp(rng.uniform(-1, 1, len(t)), 800) * np.exp(-t * 40) * 1.5
        rim = _place([(0.0, (np.sin(tau * 3200 * _t(0.03)) + rng.uniform(-1, 1, int(rate * 0.03)))
                      * np.exp(-_t(0.03) * 200) * 0.25)], 0.7)
        out = np.tanh(1.3 * (body + skin)) + rim
    else:
        return None
    return np.nan_to_num(out)


def make_chime(volume: float = WAKE_SOUND_VOLUME, freqs: tuple[float, ...] | None = None,
               tone_s: float = 0.0, gap_s: float = 0.0, rate: int = CHIME_RATE, style: str | None = None):
    """Wake sound. JARVIS_WAKE_SOUND_STYLE: bell (default), warp, comms, cosmos, chug, chime3, droplet, sonar, powerup, chirp,
    blip (quick rising sci-fi blip), or beep (the original two tones).
    Instruments: piano, acoustic_guitar, harp, violin, cello, trumpet, sax, flute, clarinet, marimba,
    xylophone, vibraphone, steel_drum, kalimba, music_box, banjo, sitar, synth_lead, organ, taiko. float32 mono."""
    import numpy as np

    style = pick_play_style(style)
    vol = max(0.0, min(1.0, volume))
    def _t(sec):
        return np.arange(int(rate * sec)) / rate

    inst = _instrument_chime(style, rate)
    if inst is not None:
        out = inst
    elif style == "warp":  # rising sci-fi sweep with airy whoosh, like jumping to lightspeed
        t = _t(0.55)
        f = 120 * (12 ** (t / t[-1]) ** 2)
        ph = 2 * np.pi * np.cumsum(f) / rate
        noise = np.random.default_rng(3).uniform(-1, 1, len(t))
        k = max(1, int(rate * 0.002))
        noise = np.convolve(noise, np.ones(k) / k, mode="same")
        env = np.sin(np.pi * (t / t[-1]) ** 0.7) ** 2
        out = (np.sin(ph) + 0.4 * np.sin(1.5 * ph) + 0.6 * noise * (t / t[-1])) * env
    elif style == "comms":  # starship comm badge: bright double tone with shimmer
        out = []
        for fr, dur in ((1760.0, 0.08), (2637.0, 0.16)):
            t = _t(dur)
            trem = 1 + 0.35 * np.sin(2 * np.pi * 38 * t)
            tone = (np.sin(2 * np.pi * fr * t) + 0.3 * np.sin(2 * np.pi * fr * 1.5 * t)) * trem
            out.append(tone * np.minimum(1, t / 0.004) * np.exp(-t * (6 if dur > 0.1 else 2)))
            out.append(np.zeros(int(rate * 0.015)))
        out = np.concatenate(out)
    elif style == "cosmos":  # deep-space shimmer: detuned bell pad with fading echoes
        t = _t(0.25)
        dry = sum(np.sin(2 * np.pi * fr * t + d) for fr, d in ((523.25, 0), (527.0, 1), (783.99, 2), (1567.98, 3)))
        dry *= np.minimum(1, t / 0.02) * np.exp(-t * 9) * (1 + 0.3 * np.sin(2 * np.pi * 7 * t))
        d = int(rate * 0.13)
        out = np.zeros(len(dry) + d * 4)
        for i in range(5):
            out[i * d:i * d + len(dry)] += dry * (0.55 ** i)
    elif style == "chug":  # palm-muted metal power chord (E5) through high-gain distortion
        rng = np.random.default_rng(7)

        def _string(freq, dur, damp):
            # Karplus-Strong plucked string; low damp factor = palm-muted thud
            n = int(rate * dur)
            per = max(2, int(rate / freq))
            buf = rng.uniform(-1, 1, per)
            out_s = np.zeros(n)
            for i in range(n):
                j = i % per
                out_s[i] = buf[j]
                buf[j] = damp * 0.5 * (buf[j] + buf[(j + 1) % per])
            return out_s

        def _chug(dur):
            chord = sum(_string(f, dur, 0.985) * (0.85 ** k)
                        for k, f in enumerate((82.41, 123.47, 164.81)))
            t = np.arange(len(chord)) / rate
            chord *= np.exp(-t * 14)  # palm mute: fast choke
            x = np.tanh(chord * 18)  # high-gain distortion
            # simple speaker-cab low-pass (one-pole, run twice)
            a = np.exp(-2 * np.pi * 3200 / rate)
            for _ in range(2):
                y = np.zeros_like(x)
                acc = 0.0
                for i, v in enumerate(x):
                    acc = (1 - a) * v + a * acc
                    y[i] = acc
                x = y
            return x * np.minimum(1, t / 0.002)

        c = _chug(0.16)
        out = np.concatenate([c, np.zeros(int(rate * 0.02)), _chug(0.32)])  # chug-chuuug
    elif style == "chime3":  # three-note rising arpeggio (C6, E6, G6), like a notification
        total = int(rate * 0.6)
        out = np.zeros(total)
        for i, fr in enumerate((1046.5, 1318.5, 1568.0)):
            st = int(rate * 0.07 * i)
            t = np.arange(total - st) / rate
            out[st:] += np.sin(2 * np.pi * fr * t) * np.minimum(1, t / 0.003) * np.exp(-t * 9)
    elif style == "droplet":  # water drop: fast downward-then-up pitch sweep
        t = _t(0.12)
        f = 1800 * np.exp(-t * 30) + 500 + 900 * (t / t[-1]) ** 3
        out = np.sin(2 * np.pi * np.cumsum(f) / rate) * np.exp(-t * 25) * np.minimum(1, t / 0.002)
    elif style == "sonar":  # single sonar ping with a long soft tail
        t = _t(0.7)
        out = (np.sin(2 * np.pi * 1200 * t) + 0.2 * np.sin(2 * np.pi * 2400 * t)) \
            * np.minimum(1, t / 0.005) * np.exp(-t * 6) * (1 + 0.15 * np.sin(2 * np.pi * 6 * t))
    elif style == "powerup":  # deep sci-fi hum that swells upward (HUD powering on)
        t = _t(0.35)
        f = np.linspace(180, 520, len(t))
        ph = 2 * np.pi * np.cumsum(f) / rate
        env = np.sin(np.pi * t / t[-1]) ** 1.5
        out = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)) * env
    elif style == "chirp":  # two quick bird-like upward chirps
        t = _t(0.06)
        f = np.linspace(1500, 2600, len(t))
        c = np.sin(2 * np.pi * np.cumsum(f) / rate) * np.sin(np.pi * t / t[-1]) ** 2
        out = np.concatenate([c, np.zeros(int(rate * 0.04)), c])
    elif style == "blip":  # short upward pitch glide, Iron-Man-HUD-ish
        n = int(rate * 0.14)
        t = np.arange(n) / rate
        f = np.linspace(600, 1500, n)
        phase = 2 * np.pi * np.cumsum(f) / rate
        env = np.minimum(1, t / 0.01) * np.exp(-t * 18)
        out = (np.sin(phase) + 0.3 * np.sin(2 * phase)) * env
    elif style == "beep":
        n = int(rate * 0.07)
        t = np.arange(n) / rate
        env = np.sin(np.pi * np.arange(n) / max(1, n - 1)) ** 2
        out = np.concatenate([np.sin(2 * np.pi * 880 * t) * env, np.zeros(int(rate * 0.02)),
                              np.sin(2 * np.pi * 1318.5 * t) * env])
    else:  # bell: two overlapping notes (E5, B5) with soft harmonics and natural decay
        notes = freqs or (659.25, 987.77)
        total = int(rate * 0.55)
        out = np.zeros(total)
        for i, fr in enumerate(notes):
            start = int(rate * 0.09 * i)
            n = total - start
            t = np.arange(n) / rate
            env = np.minimum(1, t / 0.004) * np.exp(-t * 7)
            tone = (np.sin(2 * np.pi * fr * t) + 0.35 * np.sin(2 * np.pi * fr * 2.01 * t) * np.exp(-t * 6)
                    + 0.12 * np.sin(2 * np.pi * fr * 3.0 * t) * np.exp(-t * 10))
            out[start:] += tone * env
    peak = float(np.max(np.abs(out))) or 1.0
    return (vol * out / peak).astype(np.float32)


def _load_wake_wav(rate: int = CHIME_RATE):
    """Optional custom sound from JARVIS_WAKE_SOUND_FILE (a .wav). Returns float32 mono or None."""
    path = WAKE_SOUND_FILE
    if not path:
        return None
    try:
        import wave
        import numpy as np
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        with wave.open(str(p), "rb") as w:
            ch, sw, fr = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(min(w.getnframes(), fr * 3))  # cap at 3 seconds
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
        a = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        a = (a - 128) / 128 if sw == 1 else a / float(np.iinfo(dtype).max)
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        if fr != rate:
            idx = np.linspace(0, len(a) - 1, int(len(a) * rate / fr))
            a = np.interp(idx, np.arange(len(a)), a)
        return a.astype(np.float32)
    except Exception as e:
        log.warning("Couldn't load JARVIS_WAKE_SOUND_FILE (%s); using built-in sound.", e)
        return None


def chime(volume: float | None = None, enabled: bool | None = None, sd: Any = None) -> None:
    """Wake chime (~0.16 s). Played to completion BEFORE recording starts, so it barely
    reaches the mic. Never raises; times out quickly if the audio device hangs."""
    if not (WAKE_SOUND if enabled is None else enabled):
        return
    try:
        if sd is None:
            import sounddevice as sd
        v = WAKE_SOUND_VOLUME if volume is None else volume
        custom = _load_wake_wav()
        tone = (custom * max(0.0, min(1.0, v / max(WAKE_SOUND_VOLUME, 1e-6)))).astype(custom.dtype) \
            if custom is not None else make_chime(v)
        sd.play(tone, CHIME_RATE)
        done = threading.Event()
        threading.Thread(target=lambda: (sd.wait(), done.set()), daemon=True).start()
        if not done.wait(len(tone) / CHIME_RATE + 1.0):
            sd.stop()
        time.sleep(0.03)  # let the tail decay before the mic opens
    except Exception:
        print("\a", end="", flush=True)


def followup_chime() -> None:
    """Quieter chime when follow-up listening starts."""
    chime(volume=WAKE_SOUND_VOLUME * 0.5)


# ---------------------------------------------------------------------------
# First-run setup wizard (python jarvis_assistant.py --setup)
# ---------------------------------------------------------------------------

GEORGE_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"


def mask_secret(v: str) -> str:
    v = v or ""
    return "(not set)" if not v else ("*" * len(v) if len(v) <= 8 else f"{v[:3]}...{v[-4:]}")


def list_input_devices(sd: Any = None) -> list[tuple[int, str]]:
    try:
        if sd is None:
            import sounddevice as sd
        return [(i, str(d.get("name", f"device {i}"))) for i, d in enumerate(sd.query_devices())
                if int(d.get("max_input_channels", 0) or 0) > 0]
    except Exception as e:
        log.warning("Could not list audio devices: %s", e)
        return []


def fetch_lm_models(base_url: str, timeout: float = 5.0, opener: Callable | None = None) -> list[str] | None:
    """Model ids from an OpenAI-compatible /models endpoint, or None if unreachable."""
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    try:
        with (opener or urllib.request.urlopen)(url, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        return [str(m.get("id")) for m in data.get("data", []) if m.get("id")]
    except Exception as e:
        log.warning("LM Studio check failed: %s", e)
        return None


def render_env(template: str, values: dict[str, str]) -> str:
    """Replace KEY=... lines (also '# KEY=' commented ones) keeping all comments; append unknown keys."""
    out, done = [], set()
    for line in template.splitlines():
        m = re.match(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=", line)
        if m and m.group(1) in values and m.group(1) not in done:
            k = m.group(1)
            out.append(f"{k}={values[k]}")
            done.add(k)
        else:
            out.append(line)
    extra = [f"{k}={v}" for k, v in values.items() if k not in done]
    if extra:
        out += ["", "# Added by setup wizard"] + extra
    return "\n".join(out) + "\n"


def write_env(values: dict[str, str], env_path: Path = ENV_PATH, example_path: Path = ENV_EXAMPLE_PATH) -> Path | None:
    """Write .env from .env.example with values filled in. Backs up an existing .env to .env.bak.
    Returns the backup path (or None)."""
    import shutil

    try:
        template = example_path.read_text(encoding="utf-8")
    except OSError:
        template = ""
    backup = None
    if env_path.exists():
        backup = env_path.with_name(env_path.name + ".bak")
        shutil.copy2(env_path, backup)
    env_path.write_text(render_env(template, values), encoding="utf-8")
    return backup


def print_wake_sounds(out: Callable[[str], None] = print) -> list[str]:
    """Numbered list by category; returns the names in number order (last is 'random')."""
    order: list[str] = []
    for c in WAKE_SOUND_CATEGORIES:
        out(f"{c}:")
        for n, lbl, cc in WAKE_SOUND_STYLES:
            if cc == c:
                order.append(n)
                mark = " *" if n == WAKE_SOUND_STYLE else ""
                out(f"  {len(order):2d}. {n:<16} {lbl}{mark}")
    order.append("random")
    out(f"  {len(order):2d}. random           {wake_label('random')}" + (" *" if WAKE_SOUND_STYLE == "random" else ""))
    return order


def _pick_num(s: str, order: list[str]) -> str | None:
    s = s.strip()
    if s.isdigit() and 1 <= int(s) <= len(order):
        return order[int(s) - 1]
    return match_wake_sound(s) if s else None


def wizard_wake_sound(q: Callable[..., str], out: Callable[[str], None], sd: Any = None) -> str:
    """Setup step: choose a wake sound by number; 'p N' previews."""
    out("\nWake sound (* = current). Type a number to choose, or 'p N' to preview it first.")
    order = print_wake_sounds(out)
    cur = _env("JARVIS_WAKE_SOUND_STYLE", "bell")
    cur = validate_wake_style(cur, "JARVIS_WAKE_SOUND_STYLE")
    while True:
        a = q("Wake sound number", str(order.index(cur) + 1) if cur in order else "1")
        m = re.match(r"^\s*p(?:review)?\s+(.+)$", a, re.I)
        if m:
            st = _pick_num(m.group(1), order)
            if st:
                out(f"  Playing {st}...")
                play_wake_style(st, sd)
            else:
                out("  Not a listed number.")
            continue
        st = _pick_num(a, order)
        if st:
            return st
        out("  Not a listed number, try again.")


def sounds_picker(ask: Callable[[str], str] = input, out: Callable[[str], None] = print, sd: Any = None) -> str:
    """python jarvis_assistant.py --sounds : N previews, 'set N' saves, q quits. Returns the current style."""
    out(f"=== {ASSISTANT_NAME} wake sounds === (* = current)")
    order = print_wake_sounds(out)
    out("Type a number to preview, 'set N' to choose it, 'list' to show again, or 'q' to quit.")
    while True:
        try:
            a = ask("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if a.lower() in ("q", "quit", "exit", ""):
            break
        if a.lower() == "list":
            order = print_wake_sounds(out)
            continue
        m = re.match(r"^set\s+(.+)$", a, re.I)
        st = _pick_num(m.group(1) if m else a, order)
        if not st:
            out("  Not a listed number.")
            continue
        if m:
            save_wake_style(st)
            out(f"  Wake sound set to {st} (saved in {SETTINGS_PATH.parent.name}/{SETTINGS_PATH.name}).")
        else:
            out(f"  Playing {st} ({wake_label(st)})...")
        play_wake_style(st, sd)
    return WAKE_SOUND_STYLE


def run_setup_wizard(ask: Callable[[str], str] = input, sd: Any = None, opener: Callable | None = None,
                     env_path: Path = ENV_PATH, example_path: Path = ENV_EXAMPLE_PATH,
                     out: Callable[[str], None] = print) -> dict[str, str]:
    """Interactive console setup. Enter keeps the [default]. Returns the values written."""
    def q(prompt: str, default: str = "") -> str:
        try:
            a = ask(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
        except (EOFError, StopIteration):
            a = ""
        return a or default

    vals: dict[str, str] = {}
    out(f"=== {ASSISTANT_NAME} setup === (press Enter to keep the value in brackets)\n")

    # 1) microphone
    mics = list_input_devices(sd)
    if mics:
        out("Microphones:")
        for i, name in mics:
            out(f"  {i}: {name}")
        cur = INPUT_DEVICE or ""
        while True:
            pick = q("Mic number (blank = Windows default)", cur)
            if not pick or (pick.isdigit() and int(pick) in {i for i, _ in mics}):
                break
            out("  Not a listed input number, try again.")
        vals["JARVIS_INPUT_DEVICE"] = pick
    else:
        out("No microphones found (sounddevice unavailable?); using the default mic.")

    # 2) names
    vals["ASSISTANT_NAME"] = q("Assistant name", ASSISTANT_NAME)
    t = q("What should it call you (e.g. sir, boss; type 'none' for no title)", USER_TITLE or "none")
    vals["USER_TITLE"] = "" if t.lower() == "none" else t

    # 3) ElevenLabs
    out("\nElevenLabs voice (optional; blank = free built-in Windows voice).")
    out("  Note: free ElevenLabs accounts can't use Voice Library voices through the API;")
    out("  use a default voice such as George (" + GEORGE_VOICE_ID + ").")
    key_prompt = "ElevenLabs API key" + (f" (Enter keeps {mask_secret(ELEVENLABS_API_KEY)})" if ELEVENLABS_API_KEY else "")
    key = q(key_prompt) or ELEVENLABS_API_KEY
    vals["ELEVENLABS_API_KEY"] = key
    if key:
        out(f"  Key saved: {mask_secret(key)}")
        vals["ELEVENLABS_VOICE_ID"] = q("Voice ID", ELEVENLABS_VOICE_ID or GEORGE_VOICE_ID)
    else:
        vals["ELEVENLABS_VOICE_ID"] = ELEVENLABS_VOICE_ID

    # 4) LM Studio
    base = q("\nLM Studio server URL", LLM_BASE_URL)
    vals["LLM_BASE_URL"] = base
    models = fetch_lm_models(base, opener=opener)
    if models is None:
        out(f"  Can't reach LM Studio at {base}/models. Start its server (Developer tab) later.")
        vals["LLM_MODEL"] = q("Model identifier", LLM_MODEL)
    elif not models:
        out("  LM Studio is running but no model is loaded.")
        vals["LLM_MODEL"] = q("Model identifier", LLM_MODEL)
    else:
        out("  Loaded models:")
        for n, m in enumerate(models, 1):
            out(f"    {n}: {m}")
        default = str(models.index(LLM_MODEL) + 1) if LLM_MODEL in models else "1"
        pick = q("Model number (or type an identifier)", default)
        vals["LLM_MODEL"] = models[int(pick) - 1] if pick.isdigit() and 1 <= int(pick) <= len(models) else pick

    # 5) OBS
    pw = q("\nOBS WebSocket password (optional" + (", Enter keeps current" if OBS_PASSWORD else "") + ")") or OBS_PASSWORD
    vals["OBS_PASSWORD"] = pw

    # 6) wake sound
    vals["JARVIS_WAKE_SOUND_STYLE"] = wizard_wake_sound(q, out, sd)

    backup = write_env(vals, env_path, example_path)
    if backup:
        out(f"\nBacked up the old settings to {backup.name}.")
    out(f"Saved {env_path}. Run: python {Path(__file__).name}")
    return vals


def set_user_title(title: str) -> None:
    """Change what the assistant calls the user while it is running (used by the desktop app).
    Blank = no title. Rebuilds the few texts that were filled in once at start-up."""
    global USER_TITLE, SYSTEM_PROMPT, PAUSE_REPLY, RESUME_REPLY, CONSENT_DECLINED, OBS_UNAVAILABLE
    USER_TITLE = (title or "").strip()
    SYSTEM_PROMPT = build_system_prompt()
    PAUSE_REPLY = T(_PAUSE_REPLY_TEMPLATE)
    RESUME_REPLY = T(_RESUME_REPLY_TEMPLATE)
    CONSENT_DECLINED = T(_CONSENT_DECLINED_TEMPLATE)
    OBS_UNAVAILABLE = T(_OBS_UNAVAILABLE_TEMPLATE)


# ---------------------------------------------------------------------------
# Main loops
# ---------------------------------------------------------------------------


def text_loop(brain: Brain, speaker: Any) -> None:
    print("Text mode. Type a request (or 'quit').")
    if get_scan_consent() is None:
        ask_scan_consent(speaker, lambda: input("You (yes/no): "))
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            speaker.say(brain.ask(q))


def voice_loop(brain: Brain, speaker: Any) -> None:
    ears = Ears()
    speaker.say(f"{ASSISTANT_NAME} online.")
    if get_scan_consent() is None:
        def _voice_reply() -> str:
            audio = ears.record_utterance(start_timeout=6.0)
            return ears.transcribe(audio) if audio is not None else ""
        ask_scan_consent(speaker, _voice_reply)
    state = PauseState()
    poll = lambda: poll_keyboard(state)  # noqa: E731
    print(f"Say '{WAKE_PHRASE}'... (press 'p' to pause/resume, Ctrl+C to quit)")
    while True:
        ears.wait_for_wake(speaker.speaking, poll=poll, speaker=speaker)
        if state.paused:
            resume_from_pause(speaker, state)  # then straight into a normal first turn
        else:
            chime()
        conversation(ears, brain, speaker, state=state, poll=poll, listen_cue=followup_chime, wake_cue=chime)
        if not state.paused:
            print(f"Say '{WAKE_PHRASE}'...")


def main() -> int:
    global _speaker
    ap = argparse.ArgumentParser(description=f"{ASSISTANT_NAME} voice assistant")
    ap.add_argument("--text", action="store_true", help="type instead of talk (no mic)")
    ap.add_argument("--mute", action="store_true", help="print replies instead of speaking")
    ap.add_argument("--setup", action="store_true", help="run the setup wizard (writes .env)")
    ap.add_argument("--sounds", action="store_true", help="list, preview and choose the wake sound")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.sounds:
        sounds_picker()
        return 0
    if args.setup:
        run_setup_wizard()
        return 0
    if not ENV_PATH.exists() and sys.stdin.isatty():
        try:
            a = input("No .env found. Run the setup wizard now? [Y/n]: ").strip().lower()
        except EOFError:
            a = "n"
        if a in ("", "y", "yes"):
            run_setup_wizard()
            print("Restart to use the new settings.")
            return 0

    _speaker = PrintSpeaker() if args.mute else Speaker()
    APP_INDEX.start()  # background app discovery (cache-first)
    try:
        brain = Brain()
    except ImportError:
        print("Missing dependency: pip install -r requirements.txt")
        return 1
    try:
        (text_loop if args.text else voice_loop)(brain, _speaker)
    except KeyboardInterrupt:
        pass
    except ImportError as e:
        print(f"Missing dependency ({e}). Run: pip install -r requirements.txt  (or use --text)")
        return 1
    print(T("Goodbye, {t}."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
