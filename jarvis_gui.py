"""
Jarvis desktop app: a CustomTkinter (dark) window around jarvis_assistant.py.

Run it with:   python jarvis_gui.py      (or double-click start_jarvis_gui.bat)

How it fits together (the short version):
  * jarvis_assistant.py is NOT changed. This file imports it and reuses its Ears (mic + Whisper),
    Brain (LM Studio), Speaker (voice), conversation loop and settings helpers.
  * Small subclasses (GuiEars, GuiBrain, GuiSpeaker, GuiPauseState) do exactly what the originals do,
    plus tell the window what is happening ("Listening", "Thinking", ...).
  * The voice loop runs in a background thread so the window never freezes.
  * Tkinter is only safe to touch from the main thread. So background threads never touch the
    window directly: they drop a message in a queue (App.post) and the main thread picks it up
    every 50 ms using after(). That is the one rule that keeps the GUI stable.
"""

from __future__ import annotations

import contextlib
import io
import logging
import queue
import re
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so "import jarvis_assistant" always works
_LOG_PATH = Path(__file__).resolve().parent / "jarvis_gui.log"


def _fatal(message: str) -> None:
    """Show a start-up problem in a pop-up (the .bat uses pythonw, which has no console)."""
    print(message, file=sys.stderr)
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Jarvis", message)
        root.destroy()
    except Exception:
        pass
    raise SystemExit(1)


try:
    import customtkinter as ctk
    import pystray
    from dotenv import dotenv_values
    from PIL import Image, ImageDraw, ImageFont

    import jarvis_assistant as J
except ImportError as _e:
    _fatal(f"A required package is missing ({_e}).\n\nOpen a terminal in the Jarvis folder and run:\n"
           "pip install -r requirements.txt")

from tkinter import messagebox

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# status key -> (big text, colour)
STATUS_STYLE: dict[str, tuple[str, str]] = {
    "stopped": ("Stopped", "#374151"),
    "starting": ("Starting...", "#6d28d9"),
    "listening": ("Listening", "#15803d"),
    "thinking": ("Thinking", "#b45309"),
    "speaking": ("Speaking", "#1d4ed8"),
    "paused": ("Paused", "#475569"),
    "stopping": ("Stopping...", "#374151"),
}


def record_crash(context: str) -> None:
    """Append the current exception to jarvis_gui.log (handy because there is no console)."""
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {context}\n{traceback.format_exc()}\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Status tracking (no Tk in here, so any thread may use it)
# ---------------------------------------------------------------------------

def compute_status(phase: str, paused: bool, thinking: int, speaking: int) -> str:
    """Pick the one status to show. Speaking beats Thinking beats everything else."""
    if speaking > 0:
        return "speaking"
    if thinking > 0:
        return "thinking"
    if phase in ("starting", "stopping", "stopped"):
        return phase
    return "paused" if paused else "listening"


class StatusTracker:
    """Thread-safe record of what Jarvis is doing. `on_change` is called after every change."""

    def __init__(self, on_change: Callable[[], None]) -> None:
        self._lock = threading.Lock()
        self._on_change = on_change
        self.phase = "stopped"  # stopped | starting | running | stopping
        self.paused = False
        self._busy = {"thinking": 0, "speaking": 0}
        self._detail = {"thinking": "Working on it...", "listening": ""}

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
        self._on_change()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self.paused = paused
        self._on_change()

    def set_listen_detail(self, text: str) -> None:
        with self._lock:
            self._detail["listening"] = text
        self._on_change()

    @contextlib.contextmanager
    def busy(self, kind: str, detail: str = ""):
        """`with tracker.busy("thinking"):` shows Thinking for as long as the block runs."""
        with self._lock:
            self._busy[kind] += 1
            if detail:
                self._detail[kind] = detail
        self._on_change()
        try:
            yield
        finally:
            with self._lock:
                self._busy[kind] -= 1
            self._on_change()

    def status(self) -> tuple[str, str]:
        """(status key, small explanatory line)."""
        with self._lock:
            key = compute_status(self.phase, self.paused, self._busy["thinking"], self._busy["speaking"])
            listen = self._detail["listening"] or f"Say '{J.WAKE_PHRASE}'"
            think = self._detail["thinking"]
        detail = {
            "stopped": "Press Start to begin listening",
            "starting": "Loading the speech models (the first start can take a minute)...",
            "listening": listen,
            "thinking": think,
            "speaking": "Talking...",
            "paused": f"Say '{J.WAKE_PHRASE}' or press Resume",
            "stopping": "Finishing what it was doing...",
        }[key]
        return key, detail


# ---------------------------------------------------------------------------
# Saving settings to .env (pure functions so they are easy to test)
# ---------------------------------------------------------------------------

def env_quote(value: str) -> str:
    """Quote a value only when a plain `KEY=value` line would be read back differently."""
    risky = (value != value.strip() or value[:1] in ("'", '"', "#")
             or re.search(r"\s#", value) is not None or "${" in value)
    if not risky:
        return value
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def save_env_values(updates: dict[str, str], env_path: Path | None = None,
                    example_path: Path | None = None) -> Path | None:
    """Write `updates` into .env and return the backup path (.env.bak) if there was an old .env.

    * Your existing .env values are kept (merged), so other tweaks are not reset.
    * The new text is parsed back BEFORE writing; if it wouldn't read back identically we stop.
    * jarvis_assistant.write_env copies the old .env to .env.bak first.
    """
    env_path = env_path or J.ENV_PATH
    example_path = example_path or J.ENV_EXAMPLE_PATH
    existing = {}
    if env_path.exists():
        existing = {k: v for k, v in dotenv_values(env_path, interpolate=False).items() if v is not None}
    merged = {**existing, **updates}
    for k, v in merged.items():
        if "\n" in v or "\r" in v:
            raise ValueError(f"{k} can't contain a line break.")
    quoted = {k: env_quote(v) for k, v in merged.items()}
    try:
        template = example_path.read_text(encoding="utf-8")
    except OSError:
        template = ""
    text = J.render_env(template, quoted)
    parsed = dotenv_values(stream=io.StringIO(text), interpolate=False)
    bad = [k for k, v in merged.items() if parsed.get(k) != v]
    if bad:
        raise ValueError("Couldn't safely write these values to .env: " + ", ".join(bad))
    return J.write_env(quoted, env_path, example_path)


class ObsLookupError(Exception):
    """A plain-English reason why asking OBS for its microphones didn't work."""


def find_obs_mics(host: str, port: int, password: str, timeout: float = 4.0,
                  client_factory: Callable[[], Any] | None = None) -> list[str]:
    """Ask a running OBS (through its WebSocket) for the names of its microphone inputs."""
    if client_factory is None:
        try:
            import obsws_python as obs
        except ImportError as e:
            raise ObsLookupError("OBS control needs a package that isn't installed yet. "
                                 "Run: pip install obsws-python") from e
        client_factory = lambda: obs.ReqClient(host=host, port=port, password=password or "",  # noqa: E731
                                               timeout=timeout)
    try:
        cl = client_factory()
    except Exception as e:  # OBS closed, WebSocket off, wrong password...
        words = str(e).lower()
        if "auth" in words or "password" in words:
            raise ObsLookupError("OBS rejected the password. Check that it matches the Server Password "
                                 "in OBS (Tools > WebSocket Server Settings).") from e
        raise ObsLookupError("Couldn't connect to OBS. Make sure OBS is open, 'Enable WebSocket server' is "
                             "ticked, and the password above is right.") from e
    try:
        names: list[str] = []
        for item in cl.get_input_list().inputs:
            if "input_capture" in str(item.get("inputKind", "")):  # wasapi_input_capture = a microphone
                names.append(str(item.get("inputName", "")))
        special = cl.get_special_inputs()  # the "Mic/Aux" style global audio devices
        for attr in ("mic1", "mic2", "mic3", "mic4"):
            n = getattr(special, attr, None)
            if n:
                names.append(str(n))
    except Exception as e:
        raise ObsLookupError(f"OBS answered, but the microphone list couldn't be read ({type(e).__name__}).") from e
    finally:
        with contextlib.suppress(Exception):
            cl.disconnect()
    return list(dict.fromkeys(n for n in names if n))


def list_windows_voices() -> list[str]:
    """Names of installed Windows (SAPI5) voices. Same names pyttsx3 shows, read from the registry."""
    names: list[str] = []
    try:
        import winreg

        for sub in (r"SOFTWARE\Microsoft\Speech\Voices\Tokens",
                    r"SOFTWARE\WOW6432Node\Microsoft\Speech\Voices\Tokens"):
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, sub) as key:
                    i = 0
                    while True:
                        try:
                            token = winreg.EnumKey(key, i)
                        except OSError:
                            break
                        i += 1
                        try:
                            with winreg.OpenKey(key, token) as tk_key:
                                name = str(winreg.QueryValueEx(tk_key, "")[0]).strip()
                            if name and name not in names:
                                names.append(name)
                        except OSError:
                            continue
            except OSError:
                continue
    except ImportError:
        pass
    if not names:  # fallback: ask pyttsx3
        try:
            import pyttsx3

            engine = pyttsx3.init()
            names = [str(v.name) for v in engine.getProperty("voices") if getattr(v, "name", "")]
            del engine
        except Exception as e:
            J.log.warning("Could not list Windows voices: %s", e)
    return names


# ---------------------------------------------------------------------------
# Subclasses of the real Jarvis pieces. Same behaviour + they report to the window.
# ---------------------------------------------------------------------------

class StopRequested(Exception):
    """Raised inside the voice loop to unwind it quickly when Stop is pressed."""


class GuiSpeaker(J.Speaker):
    """Speaker that also shows each reply in the chat and flips the status to Speaking."""

    def __init__(self, app: "JarvisApp") -> None:
        super().__init__()
        self._app = app

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._app.chat("assistant", text)
        if self._app.voice_thread_is_stopping():
            return  # Stop was pressed while this was being worked out: show it, don't speak it
        with self._app.tracker.busy("speaking"):
            super().say(text)


class GuiBrain(J.Brain):
    """Brain that shows Thinking and lets only one request use the chat history at a time."""

    def __init__(self, app: "JarvisApp") -> None:
        super().__init__()
        self._app = app
        self._ask_lock = threading.Lock()

    def ask(self, user_text: str) -> str:
        with self._ask_lock, self._app.tracker.busy("thinking", "Thinking about your request..."):
            return super().ask(user_text)


class GuiEars(J.Ears):
    """Ears that report 'waiting for wake word', 'listening' and 'transcribing'."""

    def __init__(self, app: "JarvisApp") -> None:
        self._app = app
        super().__init__()  # loads the wake-word model and Whisper (slow the first time)

    def wait_for_wake(self, speaking, poll=lambda: None, speaker=None) -> None:
        self._app.tracker.set_listen_detail(f"Say '{J.WAKE_PHRASE}'")
        super().wait_for_wake(speaking, poll=poll, speaker=speaker)

    def record_utterance(self, start_timeout: float = 3.0, poll=lambda: None):
        self._app.tracker.set_listen_detail("Go ahead, I'm listening...")
        return super().record_utterance(start_timeout=start_timeout, poll=poll)

    def transcribe(self, audio) -> str:
        with self._app.tracker.busy("thinking", "Working out what you said..."):
            text = super().transcribe(audio)
        if text:
            self._app.chat("user", text)
        return text


class GuiPauseState(J.PauseState):
    """PauseState that tells the window when it changes (voice phrase, button or tray)."""

    def __init__(self, app: "JarvisApp") -> None:
        super().__init__()
        self._app = app

    def set(self, paused: bool) -> None:
        super().set(paused)
        self._app.on_pause_changed(paused)


class _ChatLogHandler(logging.Handler):
    """Shows Jarvis warnings (e.g. 'Whisper on cuda failed') in the chat so they aren't invisible."""

    def __init__(self, app: "JarvisApp") -> None:
        super().__init__(level=logging.WARNING)
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        self._app.chat("system", f"[{record.levelname.lower()}] {record.getMessage()}")


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

class JarvisApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self._q: queue.Queue = queue.Queue()  # background threads -> main thread
        self._closing = False
        self._after_id: str | None = None
        self._stop = threading.Event()  # set = "voice loop, please stop"
        self._voice_thread: threading.Thread | None = None
        self._backend_lock = threading.Lock()
        self.speaker: GuiSpeaker | None = None
        self.brain: GuiBrain | None = None
        self.ears: GuiEars | None = None  # kept after Stop so Start again is instant
        self.pause_state: GuiPauseState | None = None
        self.tracker = StatusTracker(lambda: self.post(self._refresh_status))
        self._tray: Any = None
        self._tray_images: dict[str, Any] = {}
        self._tray_key = ""
        self._tray_hint_shown = False
        self._preview_busy = False

        self.title(f"{J.ASSISTANT_NAME} Assistant")
        self.geometry("840x720")
        self.minsize(660, 580)
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=12)
        self._build_main_tab(self.tabs.add("Assistant"))
        self._build_settings_tab(self.tabs.add("Settings"))

        handler = _ChatLogHandler(self)
        J.log.addHandler(handler)
        self._log_handler = handler

        self.protocol("WM_DELETE_WINDOW", self.hide_to_tray)  # the X hides to the tray
        self._start_tray()
        self._after_id = self.after(50, self._drain)
        self._refresh_status()
        self.chat("system", "Welcome. Press Start to begin listening, or just type a request below.")

    # -- thread-safe bridge --------------------------------------------------

    def post(self, fn: Callable[..., Any], *args: Any) -> None:
        """Call from ANY thread: run fn(*args) later on the main (window) thread."""
        self._q.put((fn, args))

    def _drain(self) -> None:
        """Runs on the main thread every 50 ms (via after) and executes queued messages."""
        if self._closing:
            return
        try:
            for _ in range(200):
                fn, args = self._q.get_nowait()
                try:
                    fn(*args)
                except Exception:
                    record_crash("GUI callback failed")
        except queue.Empty:
            pass
        self._after_id = self.after(50, self._drain)

    # -- main tab ------------------------------------------------------------

    def _build_main_tab(self, tab: ctk.CTkFrame) -> None:
        self.status_frame = ctk.CTkFrame(tab, corner_radius=18, fg_color=STATUS_STYLE["stopped"][1])
        self.status_frame.pack(fill="x", pady=(4, 10))
        self.status_label = ctk.CTkLabel(self.status_frame, text="Stopped", text_color="white",
                                         font=ctk.CTkFont(size=40, weight="bold"))
        self.status_label.pack(pady=(14, 0))
        self.detail_label = ctk.CTkLabel(self.status_frame, text="", text_color="#e5e7eb",
                                         font=ctk.CTkFont(size=14))
        self.detail_label.pack(pady=(0, 14))

        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", pady=(0, 10))
        for c in range(3):
            row.columnconfigure(c, weight=1, uniform="btn")
        self.start_btn = ctk.CTkButton(row, text="Start", height=40, command=self.start,
                                       fg_color="#16a34a", hover_color="#15803d")
        self.pause_btn = ctk.CTkButton(row, text="Pause", height=40, command=self.toggle_pause,
                                       fg_color="#d97706", hover_color="#b45309")
        self.stop_btn = ctk.CTkButton(row, text="Stop", height=40, command=self.stop,
                                      fg_color="#dc2626", hover_color="#b91c1c")
        for c, b in enumerate((self.start_btn, self.pause_btn, self.stop_btn)):
            b.grid(row=0, column=c, padx=5, sticky="ew")

        self.chat_box = ctk.CTkTextbox(tab, wrap="word", font=ctk.CTkFont(size=14), state="disabled")
        self.chat_box.pack(fill="both", expand=True)
        self.chat_box.tag_config("user", foreground="#7dd3fc")
        self.chat_box.tag_config("assistant", foreground="#86efac")
        self.chat_box.tag_config("system", foreground="#9ca3af")
        self.chat_box.tag_config("error", foreground="#f87171")

        bottom = ctk.CTkFrame(tab, fg_color="transparent")
        bottom.pack(fill="x", pady=(10, 0))
        bottom.columnconfigure(0, weight=1)
        self.entry = ctk.CTkEntry(bottom, height=38, placeholder_text="Type a request instead of speaking, then press Enter")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.entry.bind("<Return>", lambda _e: self.send_typed())
        ctk.CTkButton(bottom, text="Send", width=90, height=38, command=self.send_typed).grid(row=0, column=1)
        self.speak_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(bottom, text="Speak replies out loud", variable=self.speak_var).grid(
            row=1, column=0, sticky="w", pady=(8, 0))

    def _append_chat(self, role: str, text: str) -> None:
        stamp = time.strftime("%I:%M %p").lstrip("0")
        who = {"user": "You", "assistant": J.ASSISTANT_NAME, "error": "Error"}.get(role)
        line = f"{stamp}  {who}: {text}\n\n" if who else f"{stamp}  {text}\n\n"
        box = self.chat_box
        box.configure(state="normal")
        box.insert("end", line, role)
        box.configure(state="disabled")
        box.see("end")

    def chat(self, role: str, text: str) -> None:
        """Add a line to the chat log. Safe from any thread."""
        self.post(self._append_chat, role, text)

    # -- status display ------------------------------------------------------

    def _refresh_status(self) -> None:
        key, detail = self.tracker.status()
        text, color = STATUS_STYLE[key]
        self.status_frame.configure(fg_color=color)
        self.status_label.configure(text=text)
        self.detail_label.configure(text=detail)
        phase, paused = self.tracker.phase, self.tracker.paused
        self.start_btn.configure(state="normal" if phase == "stopped" else "disabled")
        self.pause_btn.configure(state="normal" if phase == "running" else "disabled",
                                 text="Resume" if paused else "Pause")
        self.stop_btn.configure(state="normal" if phase in ("starting", "running") else "disabled")
        self._update_tray(key, text)

    def is_running(self) -> bool:
        return self.tracker.phase in ("starting", "running")

    def is_paused(self) -> bool:
        return self.tracker.paused

    def voice_thread_is_stopping(self) -> bool:
        """True when Stop was pressed and the caller is the voice-loop thread."""
        return self._stop.is_set() and threading.current_thread() is self._voice_thread

    def on_pause_changed(self, paused: bool) -> None:
        self.tracker.set_paused(paused)
        self.chat("system", "Paused. Say the wake word or press Resume." if paused else "Listening again.")
        self.post(self._update_tray_menu)

    # -- start / pause / stop ------------------------------------------------

    def _ensure_backend(self) -> None:
        """Create the speaker and brain once (needed by both voice and typed requests)."""
        with self._backend_lock:
            if self.brain is not None:
                return
            speaker = GuiSpeaker(self)
            J._speaker = speaker  # timers and app-scan announcements speak through this
            J.APP_INDEX.start()  # background app discovery, same as the console version
            self.speaker = speaker
            self.brain = GuiBrain(self)

    def start(self) -> None:
        if self.tracker.phase != "stopped" or (self._voice_thread and self._voice_thread.is_alive()):
            return
        self._stop.clear()
        self.tracker.set_phase("starting")
        self._voice_thread = threading.Thread(target=self._voice_worker, daemon=True, name="jarvis-voice")
        self._voice_thread.start()

    def stop(self) -> None:
        if not self.is_running():
            return
        self._stop.set()
        self.tracker.set_phase("stopping")
        if self.speaker is not None:
            self.speaker.stop()  # cut off speech right now

    def toggle_pause(self) -> None:
        if self.pause_state is not None and self.tracker.phase == "running":
            self.pause_state.set(not self.pause_state.paused)

    def _poll(self) -> None:
        """Passed to the Ears; called about every 80 ms while listening. Raising here is how Stop works."""
        if self._stop.is_set():
            raise StopRequested()

    def _wake_cue(self) -> None:
        if not self._stop.is_set():
            J.chime()

    def _consent_reply(self) -> str:
        """Answer for 'may I scan your PC for apps?': a Yes/No pop-up instead of speaking."""
        answer: list[str] = []
        done = threading.Event()

        def ask() -> None:
            self.show_window()
            ok = messagebox.askyesno(
                f"{J.ASSISTANT_NAME}: open your apps?",
                "Would you like Jarvis to scan your PC (Start Menu, Desktop, Steam games) so it can open "
                "your apps for you?\n\nThe scan only reads shortcut names. You can turn it off any time "
                "by saying 'stop scanning my apps'.")
            answer.append("yes" if ok else "no")
            done.set()

        self.post(ask)
        while not done.wait(0.5):
            if self._stop.is_set():
                raise StopRequested()
        return answer[0]

    def _voice_worker(self) -> None:
        """The voice loop (a GUI version of jarvis_assistant.voice_loop). Runs off the main thread."""
        try:
            self._ensure_backend()
            if self.ears is None:
                self.ears = GuiEars(self)
            self._poll()
            ears, speaker, brain = self.ears, self.speaker, self.brain
            state = GuiPauseState(self)
            self.pause_state = state
            self.tracker.set_paused(False)
            self.tracker.set_phase("running")
            speaker.say(f"{J.ASSISTANT_NAME} online.")
            if J.get_scan_consent() is None:
                J.ask_scan_consent(speaker, self._consent_reply)
            while True:
                self._poll()
                ears.wait_for_wake(speaker.speaking, poll=self._poll, speaker=speaker)
                if state.paused:
                    J.resume_from_pause(speaker, state)
                else:
                    self._wake_cue()
                J.conversation(ears, brain, speaker, state=state, poll=self._poll,
                               listen_cue=J.followup_chime, wake_cue=self._wake_cue)
        except StopRequested:
            pass
        except Exception as e:
            record_crash("voice loop crashed")
            self.chat("error", f"The voice loop stopped: {type(e).__name__}: {e}")
        finally:
            self.pause_state = None
            self.tracker.set_paused(False)
            self.tracker.set_phase("stopped")
            self.chat("system", "Stopped.")

    # -- typed requests ------------------------------------------------------

    def send_typed(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self.chat("user", text)
        speak = bool(self.speak_var.get())  # read here: tk variables belong to the main thread
        threading.Thread(target=self._typed_worker, args=(text, speak), daemon=True, name="jarvis-typed").start()

    def _typed_worker(self, text: str, speak: bool) -> None:
        try:
            self._ensure_backend()
            reply = self.brain.ask(text)
            if not reply:
                self.chat("system", "(done)")
            elif speak:
                self.speaker.say(reply)
            else:
                self.chat("assistant", reply)
        except Exception as e:
            record_crash("typed request failed")
            self.chat("error", f"{type(e).__name__}: {e}")

    # -- settings tab --------------------------------------------------------

    # Settings that are only read when the assistant starts, so changing them needs a restart.
    _RESTART_LABELS = {
        "JARVIS_INPUT_DEVICE": "microphone", "JARVIS_VOICE": "Windows voice", "LLM_MODEL": "model",
        "ASSISTANT_NAME": "assistant name", "ELEVENLABS_API_KEY": "ElevenLabs key",
        "ELEVENLABS_VOICE_ID": "ElevenLabs voice ID",
    }

    @staticmethod
    def _current_env() -> dict[str, str]:
        """Current values: the .env file first, then whatever jarvis_assistant loaded."""
        file_vals = dotenv_values(J.ENV_PATH, interpolate=False) if J.ENV_PATH.exists() else {}

        def get(key: str, fallback: str) -> str:
            v = file_vals.get(key)
            return v if v is not None else fallback

        return {
            "JARVIS_INPUT_DEVICE": get("JARVIS_INPUT_DEVICE", J.INPUT_DEVICE or ""),
            "JARVIS_VOICE": get("JARVIS_VOICE", J._env("JARVIS_VOICE")),
            "LLM_BASE_URL": get("LLM_BASE_URL", J.LLM_BASE_URL),
            "LLM_MODEL": get("LLM_MODEL", J.LLM_MODEL),
            "ASSISTANT_NAME": get("ASSISTANT_NAME", J.ASSISTANT_NAME),
            "USER_TITLE": get("USER_TITLE", J.USER_TITLE),
            "ELEVENLABS_API_KEY": get("ELEVENLABS_API_KEY", J.ELEVENLABS_API_KEY),
            "ELEVENLABS_VOICE_ID": get("ELEVENLABS_VOICE_ID", J.ELEVENLABS_VOICE_ID),
            "OBS_PASSWORD": get("OBS_PASSWORD", J.OBS_PASSWORD),
            "OBS_MIC_INPUT": get("OBS_MIC_INPUT", J.OBS_MIC_INPUT),
            "JARVIS_WAKE_SOUND_VOLUME": get("JARVIS_WAKE_SOUND_VOLUME", f"{J.WAKE_SOUND_VOLUME:g}"),
        }

    def _build_settings_tab(self, tab: ctk.CTkFrame) -> None:
        cur = self._current_env()
        self._cur = cur
        sc = ctk.CTkScrollableFrame(tab)
        sc.pack(fill="both", expand=True)
        sc.columnconfigure(1, weight=1)
        self._srow = 0

        def header(text: str) -> None:
            ctk.CTkLabel(sc, text=text, font=ctk.CTkFont(size=16, weight="bold")).grid(
                row=self._srow, column=0, columnspan=2, sticky="w", pady=(14, 4))
            self._srow += 1

        def field(label: str, widget: Any, hint: str = "") -> None:
            ctk.CTkLabel(sc, text=label).grid(row=self._srow, column=0, sticky="w", padx=(0, 14), pady=4)
            widget.grid(row=self._srow, column=1, sticky="ew", pady=4)
            self._srow += 1
            if hint:
                ctk.CTkLabel(sc, text=hint, text_color="#9ca3af", font=ctk.CTkFont(size=12),
                             wraplength=460, justify="left").grid(row=self._srow, column=1, sticky="w")
                self._srow += 1

        header("Voice and audio")
        self.mic_menu = ctk.CTkOptionMenu(sc, values=["Loading microphones..."], dynamic_resizing=False)
        self._mic_map: dict[str, str] = {}
        field("Microphone", self.mic_menu)

        self.voice_menu = ctk.CTkOptionMenu(sc, values=["Loading voices..."], dynamic_resizing=False)
        self._voice_map: dict[str, str] = {}
        field("Windows voice", self.voice_menu, "Used when ElevenLabs isn't set up (or fails).")

        self._sound_map = {J.wake_label(n): n for n in J.WAKE_SOUND_NAMES + ["random"]}
        sound_row = ctk.CTkFrame(sc, fg_color="transparent")
        sound_row.columnconfigure(0, weight=1)
        self.sound_menu = ctk.CTkOptionMenu(sound_row, values=list(self._sound_map), dynamic_resizing=False,
                                            command=self._sound_changed)
        self.sound_menu.set(J.wake_label(J.WAKE_SOUND_STYLE))
        self.sound_menu.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.preview_btn = ctk.CTkButton(sound_row, text="Preview", width=90, command=self.preview_sound)
        self.preview_btn.grid(row=0, column=1)
        field("Wake sound", sound_row, "Plays when it hears the wake word. Takes effect as soon as you pick it. "
                                       "Preview uses your current volume setting.")

        try:
            vol0 = max(0.0, min(1.0, float(cur["JARVIS_WAKE_SOUND_VOLUME"])))
        except ValueError:
            vol0 = J.WAKE_SOUND_VOLUME
        vol_row = ctk.CTkFrame(sc, fg_color="transparent")
        vol_row.columnconfigure(0, weight=1)
        self.volume_slider = ctk.CTkSlider(vol_row, from_=0, to=1, number_of_steps=20,
                                           command=self._volume_changed)
        self.volume_slider.set(vol0)
        self.volume_slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.volume_label = ctk.CTkLabel(vol_row, text=f"{round(vol0 * 100)}%", width=48, anchor="e")
        self.volume_label.grid(row=0, column=1)
        field("Wake sound volume", vol_row,
              "How loud the wake sound is. It changes as you slide, so use Preview to hear it. "
              "Press Save settings to keep it next time. 0% is silent.")

        header("Brain (LM Studio)")
        model_row = ctk.CTkFrame(sc, fg_color="transparent")
        model_row.columnconfigure(0, weight=1)
        self.model_combo = ctk.CTkComboBox(model_row, values=[cur["LLM_MODEL"]])
        self.model_combo.set(cur["LLM_MODEL"])
        self.model_combo.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(model_row, text="Refresh", width=90, command=self.refresh_models).grid(row=0, column=1)
        field("Model", model_row)
        self.model_status = ctk.CTkLabel(sc, text="", text_color="#9ca3af", font=ctk.CTkFont(size=12),
                                         wraplength=460, justify="left")
        self.model_status.grid(row=self._srow, column=1, sticky="w")
        self._srow += 1

        header("Names")
        self.name_entry = ctk.CTkEntry(sc)
        self.name_entry.insert(0, cur["ASSISTANT_NAME"])
        field("Assistant name", self.name_entry, "Changing this doesn't change the wake word (see the README).")
        self.title_entry = ctk.CTkEntry(sc)
        self.title_entry.insert(0, cur["USER_TITLE"])
        field("What it calls you", self.title_entry,
              "For example sir, boss or captain. Leave blank for no title. Press Save to apply it.")

        header("ElevenLabs voice (optional)")
        self.eleven_key = ctk.CTkEntry(sc, show="\u2022")
        self.eleven_key.insert(0, cur["ELEVENLABS_API_KEY"])
        field("API key", self.eleven_key,
              "When you create the key, turn on these permissions: Speech to Text, Text to Speech, "
              "and Voices (Read). Otherwise ElevenLabs will refuse the key.")
        links = ctk.CTkFrame(sc, fg_color="transparent")
        links.grid(row=self._srow, column=1, sticky="w", pady=(0, 4))
        self._srow += 1
        for i, (text, url) in enumerate(self._ELEVEN_LINKS):
            ctk.CTkButton(
                links, text=text, width=0, height=22, fg_color="transparent", hover_color="#1f2937",
                text_color="#93c5fd", font=ctk.CTkFont(size=12, underline=True),
                command=lambda u=url: self.open_link(u)).grid(row=0, column=i, padx=(0 if i == 0 else 6, 0))
        self.eleven_voice = ctk.CTkEntry(sc)
        self.eleven_voice.insert(0, cur["ELEVENLABS_VOICE_ID"])
        field("Voice ID", self.eleven_voice, f"Free accounts need a default voice, e.g. George: {J.GEORGE_VOICE_ID}")

        header("OBS Studio (optional)")
        self.obs_pw = ctk.CTkEntry(sc, show="\u2022")
        self.obs_pw.insert(0, cur["OBS_PASSWORD"])
        field("WebSocket password", self.obs_pw,
              "Only needed if you want Jarvis to control OBS (record, stream, scenes, mute your mic). "
              "Otherwise leave it blank.")
        # A small drop-down: click to show or hide the setup steps.
        self.obs_help_btn = ctk.CTkButton(
            sc, text=self._OBS_HELP_CLOSED, width=0, height=24, anchor="w", fg_color="transparent",
            hover_color="#1f2937", text_color="#93c5fd", command=self._toggle_obs_help)
        self.obs_help_btn.grid(row=self._srow, column=1, sticky="w", pady=(2, 0))
        self._srow += 1
        self.obs_help_text = ctk.CTkLabel(
            sc, text=self._OBS_HELP_TEXT, text_color="#9ca3af", font=ctk.CTkFont(size=12),
            wraplength=460, justify="left", anchor="w")
        self.obs_help_text.grid(row=self._srow, column=1, sticky="w", pady=(2, 0))
        self.obs_help_text.grid_remove()  # hidden until the drop-down is opened
        self._srow += 1

        obs_mic_row = ctk.CTkFrame(sc, fg_color="transparent")
        obs_mic_row.columnconfigure(0, weight=1)
        self.obs_mic_combo = ctk.CTkComboBox(obs_mic_row, values=[cur["OBS_MIC_INPUT"]])
        self.obs_mic_combo.set(cur["OBS_MIC_INPUT"])
        self.obs_mic_combo.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.obs_find_btn = ctk.CTkButton(obs_mic_row, text="Find in OBS", width=100, command=self.find_obs_mic)
        self.obs_find_btn.grid(row=0, column=1)
        field("Mic name in OBS", obs_mic_row,
              "Used by 'mute my mic'. Open OBS, enter the password above, then click Find in OBS and pick yours. "
              "The default, Mic/Aux, is right for most people.")
        self.obs_status = ctk.CTkLabel(sc, text="", text_color="#9ca3af", font=ctk.CTkFont(size=12),
                                       wraplength=460, justify="left")
        self.obs_status.grid(row=self._srow, column=1, sticky="w")
        self._srow += 1

        self.show_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sc, text="Show keys and passwords", variable=self.show_var,
                        command=self._toggle_secrets).grid(row=self._srow, column=1, sticky="w", pady=(10, 0))
        self._srow += 1

        ctk.CTkButton(sc, text="Save settings", height=40, command=self.save_settings).grid(
            row=self._srow, column=0, columnspan=2, sticky="ew", pady=(18, 6))
        self._srow += 1
        self.save_msg = ctk.CTkLabel(sc, text="", wraplength=560, justify="left")
        self.save_msg.grid(row=self._srow, column=0, columnspan=2, sticky="w")

        # Fill the dropdowns without freezing the window: do the slow lookups in threads.
        threading.Thread(target=lambda: self.post(self._apply_mics, J.list_input_devices()), daemon=True).start()
        threading.Thread(target=lambda: self.post(self._apply_voices, list_windows_voices()), daemon=True).start()
        self.refresh_models()

    _ELEVEN_LINKS = (
        ("Log in", "https://elevenlabs.io/app/sign-in"),
        ("Create an account", "https://elevenlabs.io/sign-up"),
        ("Get an API key", "https://elevenlabs.io/app/settings/api-keys"),
    )

    def open_link(self, url: str) -> None:
        """Open a web page in the user's normal browser."""
        try:
            webbrowser.open(url)
        except Exception as e:  # never let a link click crash the app
            messagebox.showinfo("Open link", f"Couldn't open your browser. Please visit:\n{url}\n\n({e})")

    _OBS_HELP_CLOSED = "How to set this up  \u25B8"
    _OBS_HELP_OPEN = "How to set this up  \u25BE"
    _OBS_HELP_TEXT = (
        "1. In OBS, open Tools > WebSocket Server Settings.\n"
        "2. Tick 'Enable WebSocket server' (leave the port at 4455).\n"
        "3. Keep 'Enable Authentication' on, click 'Show Connect Info' and copy the Server Password.\n"
        "4. Paste it above and press Save settings (no restart needed).\n"
        "5. Click 'Find in OBS' below and pick your microphone, so 'mute my mic' works.\n\n"
        "If you turn authentication off in OBS, leave the password box blank. This is not an account "
        "login: it only protects the link between OBS and Jarvis on your own PC.")

    def find_obs_mic(self) -> None:
        """'Find in OBS' button: connect to OBS, list its microphones, let the user pick one."""
        password = self.obs_pw.get().strip()  # read here: tk widgets belong to the main thread
        self.obs_find_btn.configure(state="disabled")
        self.obs_status.configure(text="Asking OBS...", text_color="#9ca3af")

        def run() -> None:
            try:
                result: Any = find_obs_mics(J.OBS_HOST, J.OBS_PORT, password)
            except ObsLookupError as e:
                result = e
            except Exception as e:  # never leave the button stuck
                record_crash("OBS mic lookup failed")
                result = ObsLookupError(f"Something went wrong: {type(e).__name__}: {e}")
            self.post(self._apply_obs_mics, result)

        threading.Thread(target=run, daemon=True, name="obs-lookup").start()

    def _apply_obs_mics(self, result: Any) -> None:
        self.obs_find_btn.configure(state="normal")
        if isinstance(result, Exception):
            self.obs_status.configure(text=str(result), text_color="#f87171")
            return
        names: list[str] = result
        if not names:
            self.obs_status.configure(
                text="Connected to OBS, but it has no microphone inputs. Add one in OBS's Audio Mixer, "
                     "then try again.", text_color="#fbbf24")
            return
        current = self.obs_mic_combo.get().strip()
        self.obs_mic_combo.configure(values=names)
        if len(names) == 1 or current not in names:
            self.obs_mic_combo.set(names[0])  # a good starting point; the list lets them change it
        if len(names) == 1:
            text = f"Found your mic: {names[0]}. Press Save settings to keep it."
        else:
            text = (f"Found {len(names)} microphones in OBS. Pick yours from the list, "
                    "then press Save settings.")
        self.obs_status.configure(text=text, text_color="#86efac")

    def _toggle_obs_help(self) -> None:
        if self.obs_help_text.grid_info():  # non-empty = currently shown
            self.obs_help_text.grid_remove()
            self.obs_help_btn.configure(text=self._OBS_HELP_CLOSED)
        else:
            self.obs_help_text.grid()
            self.obs_help_btn.configure(text=self._OBS_HELP_OPEN)

    def _toggle_secrets(self) -> None:
        mask = "" if self.show_var.get() else "\u2022"
        self.eleven_key.configure(show=mask)
        self.obs_pw.configure(show=mask)

    def _apply_mics(self, devices: list[tuple[int, str]]) -> None:
        cur = self._cur["JARVIS_INPUT_DEVICE"].strip()
        self._mic_map = {"Windows default": ""}
        for i, name in devices:
            self._mic_map[f"{i}: {name}"] = str(i)
        selected = "Windows default"
        if cur:
            if cur.isdigit():  # a device number: must match exactly (not "2" inside "12")
                hit = next((lbl for lbl, v in self._mic_map.items() if v == cur), None)
            else:  # a name fragment, matched against the device names only
                hit = next((f"{i}: {name}" for i, name in devices if cur.lower() in name.lower()), None)
            if hit is None:  # a saved device that isn't plugged in right now: keep it as an option
                hit = f"{cur} (not found)"
                self._mic_map[hit] = cur
            selected = hit
        self.mic_menu.configure(values=list(self._mic_map))
        self.mic_menu.set(selected)
        if not devices:
            self.save_msg.configure(text="Couldn't list microphones; the default one will be used.", text_color="#fbbf24")

    def _apply_voices(self, names: list[str]) -> None:
        cur = self._cur["JARVIS_VOICE"].strip()
        self._voice_map = {"(default voice)": ""}
        for n in names:
            self._voice_map[n] = n
        selected = "(default voice)"
        if cur:
            hit = next((n for n in names if cur.lower() in n.lower()), None)  # same matching as Speaker
            if hit is None:
                hit = f"{cur} (not found)"
                self._voice_map[hit] = cur
            selected = hit
        self.voice_menu.configure(values=list(self._voice_map))
        self.voice_menu.set(selected)
        # If the saved value was a short name like "david", keep it unless the user picks a different voice.
        self._voice_initial = (selected, cur)

    def refresh_models(self) -> None:
        base = self._cur["LLM_BASE_URL"] or J.LLM_BASE_URL
        self.model_status.configure(text="Checking LM Studio...", text_color="#9ca3af")
        threading.Thread(target=lambda: self.post(self._apply_models, J.fetch_lm_models(base), base),
                         daemon=True).start()

    def _apply_models(self, models: list[str] | None, base: str) -> None:
        if models is None:
            self.model_status.configure(
                text=f"Can't reach LM Studio at {base}. Start its server (Developer tab), then press Refresh. "
                     "You can also type a model name.", text_color="#f87171")
        elif not models:
            self.model_status.configure(text="LM Studio is running but no model is loaded.", text_color="#fbbf24")
        else:
            current = self.model_combo.get()
            self.model_combo.configure(values=models)
            self.model_combo.set(current)
            note = "" if current in models else " (the current model isn't loaded right now)"
            self.model_status.configure(text=f"Found {len(models)} model(s) in LM Studio{note}.",
                                        text_color="#86efac" if not note else "#fbbf24")

    def _sound_changed(self, label: str) -> None:
        """Wake sound picked in the dropdown: use it right now (no restart, no Save needed).
        This is the same thing that happens when you say 'change your sound to piano'."""
        name = self._sound_map.get(label, "bell")
        J.save_wake_style(name)  # updates the running assistant and .cache/settings.json
        self.save_msg.configure(
            text=f"Wake sound is now {label}. (Press Save too if you want it written to .env as well.)",
            text_color="#86efac")

    def _volume_changed(self, value: float) -> None:
        """Volume slider moved: use it right now (Preview and the real wake sound both follow it)."""
        v = round(float(value), 2)
        J.WAKE_SOUND_VOLUME = v
        self.volume_label.configure(text=f"{round(v * 100)}%")

    def preview_sound(self) -> None:
        if self._preview_busy:
            return
        name = self._sound_map.get(self.sound_menu.get(), "bell")
        self._preview_busy = True
        self.preview_btn.configure(state="disabled")

        def run() -> None:
            try:
                J.play_wake_style(name)
            finally:
                self.post(self._preview_done)

        threading.Thread(target=run, daemon=True, name="sound-preview").start()

    def _preview_done(self) -> None:
        self._preview_busy = False
        self.preview_btn.configure(state="normal")

    def save_settings(self) -> None:
        name = self.name_entry.get().strip()
        if not name:
            self.save_msg.configure(text="The assistant name can't be empty.", text_color="#f87171")
            return
        sound = self._sound_map.get(self.sound_menu.get(), "bell")
        mic_label, voice_label = self.mic_menu.get(), self.voice_menu.get()
        updates = {
            "JARVIS_INPUT_DEVICE": self._mic_map.get(mic_label, self._cur["JARVIS_INPUT_DEVICE"]),
            "JARVIS_VOICE": self._voice_map.get(voice_label, self._cur["JARVIS_VOICE"]),
            "JARVIS_WAKE_SOUND_STYLE": sound,
            "LLM_MODEL": self.model_combo.get().strip() or self._cur["LLM_MODEL"],
            "ASSISTANT_NAME": name,
            "USER_TITLE": self.title_entry.get().strip(),
            "ELEVENLABS_API_KEY": self.eleven_key.get().strip(),
            "ELEVENLABS_VOICE_ID": self.eleven_voice.get().strip(),
            "OBS_PASSWORD": self.obs_pw.get().strip(),
            "OBS_MIC_INPUT": self.obs_mic_combo.get().strip() or "Mic/Aux",
            "JARVIS_WAKE_SOUND_VOLUME": f"{round(float(self.volume_slider.get()), 2):g}",
        }
        initial = getattr(self, "_voice_initial", None)
        if initial and voice_label == initial[0]:
            updates["JARVIS_VOICE"] = initial[1]  # untouched: keep exactly what was in .env
        try:
            backup = save_env_values(updates)
            J.save_wake_style(sound)  # .cache/settings.json would otherwise override the .env choice
        except (OSError, ValueError) as e:
            record_crash("saving settings failed")
            self.save_msg.configure(text=f"Couldn't save: {e}", text_color="#f87171")
            return
        # Apply live (no restart): the wake sound (saved above) and what it calls you.
        applied = ["wake sound"]
        if updates["JARVIS_WAKE_SOUND_VOLUME"] != self._cur["JARVIS_WAKE_SOUND_VOLUME"]:
            J.WAKE_SOUND_VOLUME = float(updates["JARVIS_WAKE_SOUND_VOLUME"])  # already live; keep it in sync
            applied.append("wake sound volume")
        if updates["USER_TITLE"] != J.USER_TITLE:
            J.set_user_title(updates["USER_TITLE"])
            if self.brain is not None:
                with self.brain._ask_lock:
                    self.brain.history.clear()  # old replies that said "sir" would keep nudging the model
            applied.append("what it calls you")
        # OBS password and mic name are read each time an OBS command runs, so they apply live too.
        if updates["OBS_PASSWORD"] != J.OBS_PASSWORD or updates["OBS_MIC_INPUT"] != J.OBS_MIC_INPUT:
            J.OBS_PASSWORD = updates["OBS_PASSWORD"]
            J.OBS_MIC_INPUT = updates["OBS_MIC_INPUT"]
            applied.append("OBS settings")
        # Everything else is read once at start-up, so it needs a restart.
        needs_restart = [label for key, label in self._RESTART_LABELS.items()
                         if updates[key] != self._cur[key]]
        self._cur.update({k: v for k, v in updates.items() if k in self._cur})
        extra = ""
        if updates["ELEVENLABS_API_KEY"] and not updates["ELEVENLABS_VOICE_ID"]:
            extra = "\nNote: ElevenLabs needs a Voice ID too, otherwise the Windows voice is used."
        bak = f" Your old settings are in {backup.name}." if backup else ""
        msg = f"Saved to .env.{bak}\nApplied now: {', '.join(applied)}."
        if needs_restart:
            msg += (f"\nRestart {J.ASSISTANT_NAME} (right-click the tray icon > Quit, then start it again) "
                    f"to apply: {', '.join(needs_restart)}.")
        msg += extra
        self.save_msg.configure(text=msg, text_color="#86efac")
        messagebox.showinfo("Settings saved", msg)

    # -- tray icon -----------------------------------------------------------

    @staticmethod
    def _make_tray_image(color: str) -> Any:
        """Draw a round coloured icon with a J in it (no image files needed)."""
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((2, 2, size - 2, size - 2), fill=color, outline="white", width=3)
        font = None
        for loader in (lambda: ImageFont.truetype("arialbd.ttf", 38), lambda: ImageFont.load_default(38)):
            try:
                font = loader()
                break
            except Exception:
                continue
        if font is not None:
            box = d.textbbox((0, 0), "J", font=font)
            d.text(((size - (box[2] - box[0])) / 2 - box[0], (size - (box[3] - box[1])) / 2 - box[1]),
                   "J", fill="white", font=font)
        return img

    def _tray_image(self, key: str) -> Any:
        if key not in self._tray_images:
            self._tray_images[key] = self._make_tray_image(STATUS_STYLE[key][1])
        return self._tray_images[key]

    def _start_tray(self) -> None:
        try:
            menu = pystray.Menu(
                pystray.MenuItem("Show", lambda icon, item: self.post(self.show_window), default=True),
                pystray.MenuItem(lambda item: "Resume" if self.is_paused() else "Pause",
                                 lambda icon, item: self.post(self.toggle_pause),
                                 enabled=lambda item: self.tracker.phase == "running"),
                pystray.MenuItem("Quit", lambda icon, item: self.post(self.quit_app)),
            )
            self._tray = pystray.Icon("jarvis", self._tray_image("stopped"), f"{J.ASSISTANT_NAME} - Stopped", menu)
            self._tray.run_detached()  # runs in its own thread
            self._tray_key = "stopped"
        except Exception:
            record_crash("tray icon failed")
            self._tray = None
            self.chat("system", "No tray icon available; closing the window will minimize it instead.")

    def _update_tray(self, key: str, text: str) -> None:
        if self._tray is None or key == self._tray_key:
            return
        self._tray_key = key
        try:
            self._tray.icon = self._tray_image(key)
            self._tray.title = f"{J.ASSISTANT_NAME} - {text}"
        except Exception:
            record_crash("updating tray icon failed")

    def _update_tray_menu(self) -> None:
        if self._tray is not None:
            with contextlib.suppress(Exception):
                self._tray.update_menu()

    def show_window(self) -> None:
        self.deiconify()
        self.lift()
        self.attributes("-topmost", True)
        self.after(250, lambda: self.attributes("-topmost", False))
        self.focus_force()

    def hide_to_tray(self) -> None:
        """The window's X button: hide to the tray (or minimize if there is no tray)."""
        if self._tray is None:
            self.iconify()
            return
        self.withdraw()
        if not self._tray_hint_shown:
            self._tray_hint_shown = True
            with contextlib.suppress(Exception):
                self._tray.notify(f"{J.ASSISTANT_NAME} is still running here. Right-click for options.",
                                  J.ASSISTANT_NAME)

    def quit_app(self) -> None:
        """Really exit: stop listening, remove the tray icon, close the window."""
        if self._closing:
            return
        self._closing = True
        self._stop.set()
        if self.speaker is not None:
            self.speaker.stop()
        J.log.removeHandler(self._log_handler)
        if self._after_id is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._after_id)
        if self._tray is not None:
            with contextlib.suppress(Exception):
                self._tray.stop()
        self.destroy()


def main() -> int:
    try:
        app = JarvisApp()
        app.mainloop()
    except Exception as e:
        record_crash("fatal error")
        _fatal(f"Jarvis couldn't start: {type(e).__name__}: {e}\n\nDetails are in {_LOG_PATH.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
