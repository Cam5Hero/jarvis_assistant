# Jarvis Assistant (v1 – "ask Jarvis anything")

Wake word "hey Jarvis" → speak → local Whisper transcribes → local LLM (LM Studio) answers / runs a skill → Jarvis speaks back.

Skills: time, open website, web search, open app (notepad / calculator / spotify), timers.

## Setup (Windows, Python 3.12)

1. Install **LM Studio** (https://lmstudio.ai).
2. In LM Studio, search and download **Qwen3 4B** (e.g. `qwen/qwen3-4b`, Q4_K_M GGUF).
3. Open the **Developer** tab, load the model, and **Start Server** (default `http://localhost:1234`).
   Check the model identifier shown there and put it in `LLM_MODEL` if it isn't `qwen3-4b`.
4. In this folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env
   ```
   Edit `.env` (ElevenLabs key/voice optional – otherwise Windows voices via pyttsx3).
5. Run:
   ```
   python jarvis_assistant.py          # voice: say "hey Jarvis", then your request
   python jarvis_assistant.py --text   # type instead (no mic needed)
   ```
   Extra flags: `--mute` (print only), `-v` (verbose logs).

First run downloads the openwakeword models and the Whisper model (one-time).

## Opening apps (opt-in app discovery)
Built-in apps always work: notepad, calculator, spotify, steam, netflix, prime video.
To open anything else ("open Discord", "launch Elden Ring"), Jarvis must scan your PC. **This is opt-in**:
on first run, after "Jarvis online", it asks *"Would you like me to scan your PC so I can open your apps for you?"*
(answer by voice, no wake word needed; in `--text` mode type yes/no). No reply counts as no.
Your answer is saved in `.cache/settings.json` (`app_scan_consent`).

- "scan my apps" / "rescan my apps" turns scanning on and rescans; Jarvis says how many apps it found.
- "stop scanning my apps" / "forget my apps" turns it off and deletes the cached index.

What gets scanned (read-only): Start Menu and Desktop shortcuts (yours and all users'), installed Steam games
(found through `libraryfolders.vdf`, launched with `steam://rungameid/<id>`), and Store apps (`Get-StartApps`).
Uninstall/help/readme/website links are skipped. The index is cached in `.cache/app_index.json` and rebuilt in the
background when it's more than 24 h old (`[apps] indexed N apps`). Names are matched loosely; if nothing matches well
Jarvis suggests up to 3 close names. Only indexed or built-in apps are ever launched; it never runs commands from the model.

## Closing apps
"Close Spotify", "close Notepad", "close Netflix". Jarvis finds the running app by name, using built-in aliases,
your scanned shortcuts' target programs, process names and window titles. It asks it to close politely, the same as
clicking X, and Steam gets the official `steam://exit`. Netflix and Prime Video only close the browser window
showing them. If the app hasn't closed after about 3 s (e.g. an unsaved document), Jarvis says so; say
**"force close <app>"** to end it (`taskkill /F`). Force is only used when you actually say "force".
Protected and never closed: Explorer, Jarvis/Python itself, LM Studio, Windows system processes and antivirus
(Norton, Defender). If several apps match, Jarvis asks which one.

## Follow-up conversation
After Jarvis answers, it keeps listening for `JARVIS_FOLLOWUP_SECONDS` (default 6) without the wake word
(`[listening for follow-up]`). Just keep talking to continue. Silence → soft cue and back to wake-word mode.
Saying "thanks", "that's all", "never mind", "goodbye" (see `JARVIS_FOLLOWUP_END_PHRASES`) ends it.
Set `JARVIS_FOLLOWUP_SECONDS=0` to disable.

## Pause / stop listening
Say "hey Jarvis, stop listening" (or "go to sleep", "mute yourself", "pause listening" – see `JARVIS_PAUSE_PHRASES`).
Jarvis prints `[paused]` and stays quiet. Saying "hey Jarvis" again resumes immediately ("At your service, sir.")
and goes straight into listening for your request.
Pressing **p** in the console window toggles pause/resume too (Windows only).

## Start with Windows
- `start_jarvis.bat` starts Jarvis in a minimized console window titled **Jarvis**. Restore it from the
  taskbar to see logs or errors, or to press **p** (pause/resume). It uses `.venv` in this folder if there is one.
- `install_startup.bat` adds a `Jarvis.lnk` shortcut to your Startup folder
  (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`), so Jarvis starts when you log in. No admin needed.
- `remove_startup.bat` removes that shortcut.

**LM Studio has to be running too.** Jarvis needs LM Studio's local server running, with the model loaded, before it can answer.
Check LM Studio's settings for options to start at login and to start the local server automatically,
and turn them on. If Jarvis starts first, it will say it can't reach LM Studio until the server is up. No restart is needed.

## Game-mode tip
While gaming, free the GPU: load **Qwen3 1.7B** in LM Studio with GPU offload set to 0 (CPU only),
set `LLM_MODEL=qwen3-1.7b`, and `WHISPER_DEVICE=cpu`, `WHISPER_MODEL=tiny` or `base`.

## OBS Studio control
Jarvis can drive OBS Studio 28+ through its built-in WebSocket server (`obsws-python`, installed via requirements.txt).

Setup:
1. In OBS open **Tools > WebSocket Server Settings** and tick **Enable WebSocket server**. Leave the port at 4455.
2. Leave **Enable Authentication** on, click **Show Connect Info**, and copy the **Server Password**.
3. Paste it into `.env` as `OBS_PASSWORD=...`. If authentication is off, leave it blank.
4. Set `OBS_MIC_INPUT` to your mic's name in the OBS Audio Mixer. The default is `Mic/Aux`.

Things to say: "start recording", "stop recording", "go live" / "start streaming", "stop streaming",
"switch to scene gameplay", "list scenes", "mute my mic" / "unmute my mic", "clip that" / "save replay"
(the Replay Buffer must be started in OBS), and "am I live?" / "am I recording?".
Jarvis only starts streaming when you clearly say "go live" or "start streaming". Scene names don't need to match exactly.
If OBS isn't running or WebSocket is off, Jarvis tells you so instead of failing.

## Troubleshooting
- "Can't reach my brain" → LM Studio server isn't running / wrong `LLM_BASE_URL`.
- Wake word too sensitive / deaf → tune `JARVIS_WAKE_THRESHOLD` (0.3–0.7).
- Recording never stops in a noisy room → raise `JARVIS_VAD_RMS`.
- openwakeword needs `onnxruntime` on Windows (included in requirements.txt; tflite isn't available there).
- CUDA Whisper needs cuBLAS 12 / cuDNN 9 (e.g. `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` and add their `bin` dirs to PATH).
  A 1 s warm-up at load detects missing DLLs and falls back to CPU int8; a CUDA error during transcription also
  triggers a one-time reload on CPU. Set `WHISPER_DEVICE=cpu` to skip CUDA entirely.

## Renaming your assistant

The name the assistant uses for itself is just a setting. In your `.env`:

```
ASSISTANT_NAME=Friday     # used in the system prompt ("You are Friday..."), greeting ("Friday online."), console and the app-scan question
USER_TITLE=boss           # what it calls you; leave blank (USER_TITLE=) for no title at all
JARVIS_WAKE_MODEL=hey_jarvis
```

With a blank `USER_TITLE`, lines like "Done, sir." become "Done." and the model is told not to use titles.
Settings keep the `JARVIS_` prefix, and the file stays `jarvis_assistant.py`, whatever name you pick.

**The wake word is a separate thing.** Changing `ASSISTANT_NAME` does not change what you say to wake it up.
The wake word comes from an openWakeWord model, and `JARVIS_WAKE_MODEL` accepts:

- a built-in model: `hey_jarvis` (default), `alexa`, `hey_mycroft` or `hey_rhasspy`
- a path to a custom-trained `.onnx` model, e.g. `JARVIS_WAKE_MODEL=models/hey_friday.onnx` (relative paths are from this folder)

The console prompt and the pause message use a friendly name taken from the model ("hey_mycroft" becomes "Hey Mycroft"; `hey_friday_v0.1.onnx` becomes "Hey Friday"). Pausing still works the same way: say the wake word to resume.

To get a wake word like "Hey Friday", you need to train a custom model. The openWakeWord project has a notebook for training custom models, linked in its README: https://github.com/dscripka/openWakeWord
