"""In-app updater for ZIP installs (no git needed). Pure standard library, no PowerShell.

Flow: latest_sha() asks GitHub for the newest commit on main -> compare with .jarvis_version ->
download_zip() -> install_zip() copies the files over this folder, never touching user data
(.env, .venv, .cache, logs ...) and never deleting local files -> pip install if requirements.txt changed
-> write the new SHA. Every step logs to update.log.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

REPO = "Cam5Hero/jarvis_assistant"
BRANCH = "main"
API_URL = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"
ZIP_URL = f"https://github.com/{REPO}/archive/refs/heads/{BRANCH}.zip"
APP_DIR = Path(__file__).resolve().parent
VERSION_FILE = ".jarvis_version"
LOG_NAME = "update.log"
USER_AGENT = "Jarvis-Assistant-Updater"
GPU_REQ = "requirements-gpu.txt"

# Top-level names (or patterns) that belong to the user and are never written by an update.
PROTECTED = (".env", ".env.bak", ".venv", "venv", ".cache", ".git", "__pycache__", VERSION_FILE,
             "*.log", "*.pyc")

Progress = Callable[[float, str], None]


def log(msg: str, app_dir: Path = APP_DIR) -> None:
    try:
        with open(app_dir / LOG_NAME, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def _get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def latest_commit(timeout: float = 6.0, fetch: Callable[[str, float], bytes] = _get) -> tuple[str, str] | None:
    """(sha, first line of the commit message), or None if offline / rate limited / anything odd."""
    try:
        data = json.loads(fetch(API_URL, timeout).decode("utf-8"))
        sha = str(data.get("sha") or "").strip()
        msg = str(((data.get("commit") or {}).get("message") or "")).strip().splitlines()
        if len(sha) >= 7:
            return sha, (msg[0] if msg else "")
    except Exception:
        pass
    return None


def installed_sha(app_dir: Path = APP_DIR) -> str:
    try:
        return (app_dir / VERSION_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def write_installed_sha(sha: str, app_dir: Path = APP_DIR) -> bool:
    try:
        (app_dir / VERSION_FILE).write_text(sha.strip() + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def update_available(latest: str, installed: str) -> bool:
    """Unknown installed version (no file) counts as out of date, so old ZIP installs get offered it."""
    latest, installed = (latest or "").strip().lower(), (installed or "").strip().lower()
    if not latest:
        return False
    return latest != installed


def is_protected(rel: str) -> bool:
    parts = Path(rel).parts
    if not parts:
        return True
    for p in parts:  # any path component: .venv/..., sub/.cache/...
        if p in (".venv", "venv", ".cache", ".git", "__pycache__"):
            return True
    top, name = parts[0], parts[-1]
    for pat in PROTECTED:
        if fnmatch.fnmatch(top, pat) or fnmatch.fnmatch(name, pat):
            return True
    return False


def download_zip(dest: Path, progress: Progress | None = None, timeout: float = 30.0, url: str = ZIP_URL) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)  # GitHub often omits it for archives
        got = 0
        while True:
            chunk = r.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if progress:
                frac = min(got / total, 1.0) if total else -1.0
                progress(frac, f"Downloading... {got / 1e6:.1f} MB")
    return dest


def _sha256(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return ""


def install_zip(zip_path: Path, app_dir: Path = APP_DIR, progress: Progress | None = None) -> dict:
    """Copy the zip's files over app_dir. Returns {'copied': [...], 'skipped': [...], 'requirements_changed': bool}.
    Never deletes anything and never writes protected paths."""
    before = _sha256(app_dir / "requirements.txt")
    gpu_before = _sha256(app_dir / GPU_REQ)
    copied, skipped = [], []
    with tempfile.TemporaryDirectory(prefix="jarvis_update_") as tmp:
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():  # zip-slip guard
                target = (Path(tmp) / info.filename).resolve()
                if not str(target).startswith(str(Path(tmp).resolve())):
                    raise ValueError(f"Unsafe path in update zip: {info.filename}")
            z.extractall(tmp)
        roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
        src = roots[0] if len(roots) == 1 and not any(p.is_file() for p in Path(tmp).iterdir()) else Path(tmp)
        if not (src / "jarvis_assistant.py").exists():
            raise ValueError("The download doesn't look like Jarvis (jarvis_assistant.py missing).")
        files = [p for p in src.rglob("*") if p.is_file()]
        for i, f in enumerate(files):
            rel = f.relative_to(src).as_posix()
            if is_protected(rel):
                skipped.append(rel)
                continue
            dst = app_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp_dst = dst.with_name(dst.name + ".updtmp")
            shutil.copy2(f, tmp_dst)
            os.replace(tmp_dst, dst)  # atomic per file; fine for running .py files on Windows
            copied.append(rel)
            if progress:
                progress((i + 1) / max(len(files), 1), f"Copying files... ({i + 1}/{len(files)})")
    return {"copied": copied, "skipped": skipped,
            "requirements_changed": before != _sha256(app_dir / "requirements.txt"),
            "gpu_requirements_changed": gpu_before != _sha256(app_dir / GPU_REQ)}


def venv_python(app_dir: Path = APP_DIR) -> Path:
    if os.name == "nt":
        return app_dir / ".venv" / "Scripts" / "python.exe"
    return app_dir / ".venv" / "bin" / "python"


def has_nvidia_gpu() -> bool:
    if os.name != "nt":
        return False
    if shutil.which("nvidia-smi"):
        return True
    return (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe").exists()


def pip_install(app_dir: Path = APP_DIR, progress: Progress | None = None, req: str = "requirements.txt") -> bool:
    py = venv_python(app_dir)
    if not py.exists():
        py = Path(sys.executable)
    cmd = [str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(app_dir / req)]
    log("running: " + " ".join(cmd), app_dir)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, cwd=str(app_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True, errors="replace", creationflags=flags)
    with open(app_dir / LOG_NAME, "a", encoding="utf-8") as lf:
        for line in proc.stdout:  # type: ignore[union-attr]
            lf.write(line)
            s = line.strip()
            if progress and s:
                progress(-1.0, "Updating add-ons: " + s[:70])
    return proc.wait() == 0


def run_update(sha: str, app_dir: Path = APP_DIR, progress: Progress | None = None,
               downloader: Callable[..., Path] = download_zip) -> dict:
    """Whole update. Raises on failure (caller shows a friendly message); details go to update.log."""
    log(f"update to {sha or '?'} started (installed: {installed_sha(app_dir) or 'unknown'})", app_dir)
    with tempfile.TemporaryDirectory(prefix="jarvis_dl_") as tmp:
        zp = downloader(Path(tmp) / "jarvis.zip", progress)
        res = install_zip(zp, app_dir, progress)
    log(f"copied {len(res['copied'])} files, skipped {res['skipped']}", app_dir)
    res["pip_ok"] = True
    if res["requirements_changed"]:
        if progress:
            progress(-1.0, "Updating add-ons (this can take a few minutes)...")
        res["pip_ok"] = pip_install(app_dir, progress)
        log(f"pip install ok={res['pip_ok']}", app_dir)
    if res.get("gpu_requirements_changed") and (app_dir / GPU_REQ).exists() and has_nvidia_gpu():
        if progress:
            progress(-1.0, "Adding the graphics card speed-up (big download, can take a while)...")
        gpu_ok = pip_install(app_dir, progress, GPU_REQ)  # optional: CPU still works if this fails
        log(f"gpu pip install ok={gpu_ok}", app_dir)
    if sha:
        write_installed_sha(sha, app_dir)
    log("update finished", app_dir)
    return res
