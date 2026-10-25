"""Jarvis setup: a small window that installs everything, no command prompt needed.

Standard library only (nothing else is installed yet). Started by setup.bat with pythonw.
Safe to run again: it reuses .venv and just makes sure everything is installed.
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
LOG_PATH = APP_DIR / "setup.log"
VENV = APP_DIR / ".venv"
MIN_PY = (3, 10)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # hide every child console on Windows

BG, FG, MUTED, BLUE, BLUE_HOVER, RED = "#212121", "#ffffff", "#9ca3af", "#1f6aa5", "#144870", "#f87171"
_PROGRESS_RE = re.compile(r"^Progress (\d+) of (\d+)\s*$")  # pip >= 24.1 --progress-bar raw


def venv_python(windowless: bool = False) -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / ("pythonw.exe" if windowless else "python.exe")
    return VENV / "bin" / "python"


def has_nvidia_gpu() -> bool:
    """True if an NVIDIA driver is installed (nvidia-smi is on PATH or in System32). Windows only."""
    if os.name != "nt":
        return False
    import shutil
    if shutil.which("nvidia-smi"):
        return True
    return (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe").exists()


def console_python() -> str:
    """The python.exe next to this pythonw.exe (same install), for running hidden child processes."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


def fmt_bytes(n: float) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n / 1024:.0f} KB"


def pkg_name(text: str) -> str:
    name = re.split(r"[\s<>=!~\[;(]", text.strip(), maxsplit=1)[0]
    name = name.rsplit("/", 1)[-1].split("-")[0]  # wheel file name -> package name
    return (name[:30] + "...") if len(name) > 30 else name


class SetupError(Exception):
    pass


class Installer:
    """Does the work in a background thread. Reports through report(kind, *args):
    ("progress", fraction 0..1, status text), ("done",), ("error", message, last lines)."""

    def __init__(self, with_kokoro: bool, report) -> None:
        self.with_kokoro = with_kokoro
        self.report = report
        self.tail: list[str] = []
        self.log = None

    # --- helpers ---
    def _write(self, text: str) -> None:
        self.log.write(text)
        self.log.flush()

    def _run(self, cmd: list[str], lo: float, hi: float, label: str, env: dict | None = None) -> tuple[int, bool]:
        """Run one hidden command, moving the bar between lo and hi. Returns (exit code, raw_unsupported)."""
        self._write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        self.report("progress", lo, label)
        steps, current, last_post, last_logged, raw_bad = 0, "", 0.0, -1, False
        proc = subprocess.Popen(cmd, cwd=str(APP_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                bufsize=1, creationflags=NO_WINDOW, env=env)
        assert proc.stdout is not None
        span = hi - lo
        for line in proc.stdout:
            s = line.strip()
            m = _PROGRESS_RE.match(s)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                if total <= 0:
                    continue
                if done * 10 // total != last_logged:  # keep the log short: one line per 10%
                    last_logged = done * 10 // total
                    self._write(line)
                now = time.monotonic()
                if now - last_post >= 0.1 or done >= total:  # about 10 updates a second
                    last_post = now
                    frac = lo + span * min(0.95, steps / (steps + 8))
                    self.report("progress", frac,
                                f"Downloading {current}: {fmt_bytes(done)} of {fmt_bytes(total)}")
                continue
            self._write(line)
            if s:
                self.tail = (self.tail + [line.rstrip()])[-8:]
            if "invalid choice" in s and "raw" in s:
                raw_bad = True
            status = None
            if s.startswith("Collecting "):
                status = "Getting things ready..."
            elif s.startswith("Downloading ") and not s.split(" (")[0].endswith(".metadata"):
                current = pkg_name(s[12:])
                status = f"Downloading {current}..."
            elif s.startswith("Installing collected packages"):
                status = "Installing... this can take a few minutes"
            if status:
                steps += 1  # no total is known, so creep towards hi without ever reaching it
                self.report("progress", lo + span * min(0.95, steps / (steps + 8)), status)
        code = proc.wait()
        self._write(f"===== exit code {code}\n")
        return code, raw_bad

    def _pip(self, py: str, args: list[str], lo: float, hi: float, label: str,
             fail: str = "Some parts of Jarvis couldn't be downloaded.") -> None:
        base = [py, "-m", "pip", "install", "--disable-pip-version-check"]
        self.tail = []  # error details should be about this step only
        for mode in ("raw", "off"):  # old pip doesn't know "raw": retry once with "off"
            code, raw_bad = self._run(base + ["--progress-bar", mode] + args, lo, hi, label)
            if code == 0:
                return
            if not (raw_bad and mode == "raw"):
                break
        self._write(f"(step failed: {label} exit code {code})\n")
        raise SetupError(fail)

    # --- the steps ---
    def run(self) -> None:
        try:
            with open(LOG_PATH, "a", encoding="utf-8", errors="replace") as self.log:
                self._write(f"\n##### Jarvis setup {time.strftime('%Y-%m-%d %H:%M:%S')}, Python {sys.version}\n")
                self._steps()
                self._record_version()
            self.report("done")
        except SetupError as e:
            self.report("error", str(e), "\n".join(self.tail[-6:]))
        except Exception as e:  # never leave the window stuck
            if self.log and not self.log.closed:
                self.log.write(f"(unexpected error: {e!r})\n")
            self.report("error", "Something went wrong during setup.", "\n".join(self.tail[-6:]))

    def _steps(self) -> None:
        py = venv_python()
        if not py.exists():
            code, _ = self._run([console_python(), "-m", "venv", str(VENV)], 0.0, 0.05,
                                "Getting things ready...")
            if code != 0 or not py.exists():
                raise SetupError("Setup couldn't get started in this folder. Try moving the Jarvis folder to Documents.")
        self.report("progress", 0.05, "Getting things ready...")
        self._pip(str(py), ["--upgrade", "pip"], 0.05, 0.10, "Getting things ready...")
        gpu = has_nvidia_gpu() and (APP_DIR / "requirements-gpu.txt").exists()
        self._write(f"(NVIDIA graphics card found: {gpu})\n")
        top = 0.65 if self.with_kokoro else 0.95
        if gpu:
            top = 0.45 if self.with_kokoro else 0.6
        self._pip(str(py), ["-r", str(APP_DIR / "requirements.txt")], 0.10, top, "Downloading Jarvis...")
        if gpu:
            hi = 0.65 if self.with_kokoro else 0.95
            try:  # optional: without it speech recognition just runs on the CPU
                self._pip(str(py), ["-r", str(APP_DIR / "requirements-gpu.txt")], top, hi,
                          "Adding the graphics card speed-up (big download)...")
            except SetupError:
                self._write("(GPU speed-up not installed; Jarvis will use the CPU)\n")
        if self.with_kokoro:
            self._pip(str(py), ["-r", str(APP_DIR / "requirements-kokoro.txt")], 0.65, 0.95,
                      "Downloading the offline voice...", "The offline voice (Kokoro) couldn't be added. "
                      "Run setup again without it, or add it later in Jarvis: Settings > Voice.")
        self._shortcut()
        self.report("progress", 1.0, "Done!")

    def _record_version(self) -> None:
        """Remember which GitHub version is installed (.jarvis_version) so Jarvis can spot updates.
        Silent on any failure: Jarvis will then simply offer the latest version once."""
        try:
            sys.path.insert(0, str(APP_DIR))
            import jarvis_updater as U

            latest = U.latest_commit(timeout=6.0)
            if latest and U.write_installed_sha(latest[0], APP_DIR):
                self._write(f"(installed version {latest[0][:7]})\n")
        except Exception:
            pass

    def _shortcut(self) -> None:
        """Same plain PowerShell as tools\\create_desktop_shortcut.bat (WScript.Shell makes Desktop\\Jarvis.lnk)."""
        if os.name != "nt":
            self._write("(skipping desktop shortcut: not Windows)\n")
            return
        self.report("progress", 0.96, "Adding Desktop shortcut...")
        env = dict(os.environ, JV_PYW=str(venv_python(windowless=True)), JV_DIR=str(APP_DIR))
        ps = ("$lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Jarvis.lnk'; "
              "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk); $s.TargetPath = $env:JV_PYW; "
              "$s.Arguments = [char]34 + (Join-Path $env:JV_DIR 'jarvis_gui.py') + [char]34; "
              "$s.WorkingDirectory = $env:JV_DIR; $ico = Join-Path $env:JV_DIR 'jarvis.ico'; "
              "if (Test-Path $ico) { $s.IconLocation = $ico }; $s.Description = 'Jarvis voice assistant'; "
              "$s.Save(); Write-Host ('Created ' + $lnk)")
        code, _ = self._run(["powershell", "-NoProfile", "-Command", ps], 0.96, 0.99,
                            "Adding Desktop shortcut...", env=env)
        if code != 0:  # not fatal: Jarvis works without it
            self._write("(shortcut failed; run tools\\create_desktop_shortcut.bat later)\n")


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

def main() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    if os.name == "nt":  # own taskbar icon instead of the generic Python one
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Jarvis.Setup")
        except Exception:
            pass
    root = tk.Tk()
    root.title("Jarvis setup")
    root.configure(bg=BG)
    root.geometry("520x360")
    root.resizable(False, False)
    ico = APP_DIR / "jarvis.ico"
    if ico.exists():
        try:
            root.iconbitmap(str(ico))
        except tk.TclError:
            pass

    if sys.version_info < MIN_PY:
        root.withdraw()
        messagebox.showerror("Jarvis setup", "Jarvis needs a newer version of Python.\n\n"
                             "Install Python 3.12 from python.org, then run setup again.")
        return

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("J.Horizontal.TProgressbar", troughcolor="#2b2b2b", background=BLUE,
                    bordercolor=BG, lightcolor=BLUE, darkcolor=BLUE)
    title_font, body_font, small_font = ("Segoe UI", 16, "bold"), ("Segoe UI", 11), ("Segoe UI", 9)

    frame = tk.Frame(root, bg=BG)
    frame.pack(fill="both", expand=True, padx=24, pady=20)

    def clear() -> None:
        for w in frame.winfo_children():
            w.destroy()

    def label(text: str, font=body_font, fg=FG, pady=(0, 8), padx=0) -> tk.Label:
        lb = tk.Label(frame, text=text, font=font, fg=fg, bg=BG, justify="left", anchor="w",
                      wraplength=470 - padx)
        lb.pack(fill="x", pady=pady, padx=(padx, 0))
        return lb

    def button(parent, text: str, cmd, color=BLUE, hover=BLUE_HOVER) -> tk.Button:
        b = tk.Button(parent, text=text, command=cmd, bg=color, fg=FG, activebackground=hover,
                      activeforeground=FG, relief="flat", bd=0, padx=18, pady=6, font=("Segoe UI", 10, "bold"),
                      cursor="hand2")
        b.bind("<Enter>", lambda e: b.configure(bg=hover))
        b.bind("<Leave>", lambda e: b.configure(bg=color))
        return b

    kokoro_var = tk.BooleanVar(value=False)
    events: queue.Queue = queue.Queue()

    def welcome() -> None:
        clear()
        label("Set up Jarvis", title_font)
        label("Jarvis will be installed here:", pady=0)
        label(str(APP_DIR), ("Consolas", 9), MUTED)
        label("A Jarvis shortcut will be added to your Desktop.")
        label("Download: a few hundred MB (about 5-15 minutes).")
        tk.Checkbutton(frame, text="Also add the free offline voice (Kokoro), about 1-2 GB",
                       variable=kokoro_var, bg=BG, fg=FG, selectcolor="#2b2b2b", activebackground=BG,
                       activeforeground=FG, font=body_font, anchor="w").pack(fill="x", pady=(4, 0))
        label("You can add it later in Jarvis: Settings > Voice.", small_font, MUTED, padx=24)
        row = tk.Frame(frame, bg=BG)
        row.pack(side="bottom", fill="x")
        button(row, "Install", start).pack(side="right")
        button(row, "Cancel", root.destroy, "#3a3a3a", "#4a4a4a").pack(side="right", padx=(0, 8))

    def progress_screen() -> dict:
        clear()
        label("Installing Jarvis...", title_font)
        label("Please keep this window open.", fg=MUTED)
        bar_row = tk.Frame(frame, bg=BG)
        bar_row.pack(fill="x", pady=(10, 4))
        bar = ttk.Progressbar(bar_row, style="J.Horizontal.TProgressbar", maximum=100, length=410)
        bar.pack(side="left")
        pct = tk.Label(bar_row, text="0%", font=body_font, fg=FG, bg=BG, width=5, anchor="e")
        pct.pack(side="right")
        status = label("Starting...", small_font, MUTED)
        return {"bar": bar, "pct": pct, "status": status}

    ui: dict = {}

    def start() -> None:
        ui.clear()
        ui.update(progress_screen())
        ui["value"] = 0.0
        inst = Installer(kokoro_var.get(), lambda *a: events.put(a))
        threading.Thread(target=inst.run, daemon=True).start()
        root.after(100, pump)

    def pump() -> None:
        """Apply the worker's updates on the window thread."""
        try:
            while True:
                ev = events.get_nowait()
                if ev[0] == "progress":
                    ui["value"] = max(ui["value"], ev[1])  # never move backwards
                    ui["bar"]["value"] = ui["value"] * 100
                    ui["pct"].configure(text=f"{round(ui['value'] * 100)}%")
                    ui["status"].configure(text=ev[2])
                elif ev[0] == "done":
                    return done_screen()
                elif ev[0] == "error":
                    return error_screen(ev[1], ev[2])
        except queue.Empty:
            pass
        root.after(100, pump)

    def done_screen() -> None:
        clear()
        label("Jarvis is ready!", title_font)
        label("Open it from the Jarvis shortcut on your Desktop.")
        label("To uninstall later: Jarvis > Settings > Account > Uninstall.", small_font, MUTED)
        row = tk.Frame(frame, bg=BG)
        row.pack(side="bottom", fill="x")
        button(row, "Start Jarvis", launch).pack(side="right")
        button(row, "Close", root.destroy, "#3a3a3a", "#4a4a4a").pack(side="right", padx=(0, 8))

    def error_screen(msg: str, tail: str) -> None:
        clear()
        label("Setup didn't finish", title_font, RED)
        label(msg)
        label("Check your internet connection and press Retry.")
        label("Details were saved to:", pady=0)
        label(str(LOG_PATH), ("Consolas", 9), MUTED)
        row = tk.Frame(frame, bg=BG)
        row.pack(side="bottom", fill="x")
        button(row, "Retry", start).pack(side="right")
        button(row, "Close", root.destroy, "#3a3a3a", "#4a4a4a").pack(side="right", padx=(0, 8))

    def launch() -> None:
        try:
            subprocess.Popen([str(venv_python(windowless=True)), str(APP_DIR / "jarvis_gui.py")],
                             cwd=str(APP_DIR), creationflags=NO_WINDOW, close_fds=True)
        except OSError as e:
            messagebox.showerror("Jarvis setup", f"Couldn't start Jarvis: {e}\nUse the Desktop icon instead.")
            return
        root.destroy()

    welcome()
    root.mainloop()


def find_pythonw() -> str:
    """For setup.bat: this Python's pythonw.exe (or python.exe) if the version is new enough, else nothing."""
    if sys.version_info < MIN_PY:
        return ""
    p = Path(sys.executable).with_name("pythonw.exe")
    return str(p) if p.exists() else sys.executable  # e.g. Microsoft Store Python: use python.exe instead


if __name__ == "__main__":
    if "--find-pythonw" in sys.argv:
        print(find_pythonw())
    else:
        main()
