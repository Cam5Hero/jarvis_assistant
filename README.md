# Jarvis Assistant (v1 – "ask Jarvis anything")

Wake word "hey Jarvis" → speak → local Whisper transcribes → local LLM (LM Studio) answers / runs a skill → Jarvis speaks back.

## Features
- Wake word ("hey Jarvis") with a short two-tone chime, then local Whisper speech-to-text (`small` model by default)
- Local LLM (LM Studio) with tools: time, open website, web search, open/close apps and games, timers
- **Media and volume control**: play/pause, next/previous track, volume up/down, mute, set volume to a percent
- OBS Studio control (record, stream, scenes, mic, replay clips, status)
- Follow-up conversation, pause mode, **barge-in** (say the wake word to interrupt it)
- ElevenLabs voice or free built-in Windows voices
- First-run **setup wizard** (`--setup`)

## Quick start
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python jarvis_assistant.py --setup     # pick mic, names, voice, LM Studio model, OBS password
python jarvis_assistant.py
```
(Start LM Studio's server with a model loaded first so the wizard can list it; see Setup below.)

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
   python jarvis_assistant.py --setup
   ```
   The wizard writes `.env` (see "Setup wizard" below). You can also `copy .env.example .env` and edit it by hand
   (ElevenLabs key/voice optional – otherwise Windows voices via pyttsx3).
5. Run:
   ```
   python jarvis_assistant.py          # voice: say "hey Jarvis", then your request
   python jarvis_assistant.py --text   # type instead (no mic needed)
   ```
   Extra flags: `--mute` (print only), `-v` (verbose logs).

First run downloads the openwakeword models and the Whisper model (one-time).

Whisper uses the `small` model by default (`WHISPER_MODEL=small`) for better accuracy. On a weak PC set
`WHISPER_MODEL=base` (faster, a bit less accurate). On CPU it always runs `int8`; if CUDA fails it falls back to CPU.

## Setup wizard
`python jarvis_assistant.py --setup` (also offered automatically when there's no `.env`) asks, in the console:
1. which **microphone** to use (lists input devices; blank = Windows default),
2. the **assistant name** and what it calls you (**user title**; `none` for no title),
3. an optional **ElevenLabs API key** and **voice ID**. It suggests George (`JBFqnCBsd6RMkjVDRZzb`). Free ElevenLabs
   accounts can't use Voice Library voices through the API, so pick a default voice. The key is never printed in full.
4. it checks **LM Studio** at `LLM_BASE_URL/models`, lists the loaded models and lets you pick `LLM_MODEL`,
5. an optional **OBS WebSocket password**.

It writes `.env` from `.env.example` with your values filled in (comments kept). An existing `.env` is first copied
to `.env.bak`. Press Enter at any question to keep the value shown in brackets. Restart the assistant afterwards.

## Media and volume
Things to say: "pause the music", "play", "next song", "previous track", "turn it up", "turn the volume down a lot",
"mute" / "unmute", "set the volume to 40 percent".
These send Windows media keys, so they control whatever app has the media session (Spotify, a browser video, etc.)
and the system volume. Each volume step is 2%; "up"/"down" moves 5 steps (10%) unless you say a little/a lot.
**Set volume to a percent is approximate**: Windows has no media key for an exact level, so it presses volume-down
50 times (to 0%) and then volume-up percent/2 times (odd numbers round down). The volume overlay will flicker briefly.
Off Windows these tools just reply that media keys only work on Windows.

## Wake chime and barge-in
When the wake word is heard, a short two-tone chime plays before recording starts (a quieter one marks follow-up
listening). Turn it off with `JARVIS_WAKE_SOUND=0`, or change `JARVIS_WAKE_SOUND_VOLUME` (default 0.3).

**Barge-in**: while it's talking, say the wake word to stop it immediately; it chimes and listens for your new request.
It uses a stricter `JARVIS_BARGE_IN_THRESHOLD` (default 0.7) so its own voice from the speakers doesn't trigger it.
**Headphones work best**; with loud speakers, raise the threshold or set `JARVIS_BARGE_IN=0`.
ElevenLabs speech stops instantly. The built-in Windows voice (pyttsx3) is stopped from another thread, which usually
cuts it within a word or so but is best effort; some voices finish the current sentence.

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

## Desktop app
`jarvis_gui.py` is a dark-themed window around the same assistant (built with CustomTkinter). The console version
(`jarvis_assistant.py`) is unchanged and still works exactly as before.

```
pip install -r requirements.txt       # adds customtkinter, pystray and pillow
python jarvis_gui.py                  # or double-click start_jarvis_gui.bat (no console window)
```
Use **either** the console version **or** the desktop app, not both at once (they would both answer you).

**Assistant tab**
- A big status box: **Listening** (green), **Thinking** (amber), **Speaking** (blue) or **Paused** (grey). It also shows
  **Stopped** and **Starting...** (the first start loads Whisper, which can take a while).
- **Start** begins the voice loop (say the wake word as usual), **Pause** / **Resume** works like saying "stop listening"
  (saying the wake word also resumes), and **Stop** ends the loop. Starting again afterwards is instant.
- The chat log shows what you said and what Jarvis replied, plus warnings (for example if Whisper falls back to CPU).
- Type in the box and press Enter to make a request without speaking. Untick "Speak replies out loud" for text-only
  answers. This works even before you press Start.
- The first time, a Yes/No pop-up asks about the opt-in app scan (see "Opening apps").

**Settings tab**
- Dropdowns: **microphone** (from sounddevice), **Windows voice** (installed voices), **wake sound** (with a **Preview**
  button) and **LM Studio model** (from `/v1/models`; press **Refresh** after starting LM Studio, or type a name).
- **Wake sound volume** slider (0-100%, default 30%). It changes as you slide, so use **Preview** to hear it; press
  Apply to keep it (`JARVIS_WAKE_SOUND_VOLUME`).
- Fields: assistant name, what it calls you, ElevenLabs key (hidden) and voice ID, OBS password (hidden) and OBS mic
  name. Tick "Show keys and passwords" to see them. Under the OBS password there is a **How to set this up** drop-down
  with the steps, and a **Find in OBS** button: with OBS open, it lists your OBS microphones so you can pick yours
  (no need to edit any files).
- Each setting has its own **Apply** button that appears when you change it. Apply saves just that one setting to
  `.env` (every other line and comment stays as it is; the first save of a session copies your old `.env` to `.env.bak`).
  A banner at the top of the Settings tab (it stays put while you scroll) says **Settings saved** for a few seconds.
  If a setting is only read at start-up, the banner stays amber ("Restart Jarvis to use the new ...") with a
  **Restart Jarvis** button that closes the app and opens it again. Errors show in red in the same place.
- **Applied immediately, no restart:** the **wake sound** (as soon as you pick it in the dropdown; it's stored in
  `.cache/settings.json`, which overrides `.env`), **what it calls you** (when you press Apply; the chat history is
  cleared so earlier replies don't keep using the old title) and the **OBS password and mic name** (when you press Apply).
- **Need a restart** (the banner offers a **Restart Jarvis** button): microphone, Windows voice, model, assistant
  name and ElevenLabs key/voice ID. Use the tray icon > **Quit**, then start the app again.
- Microphone numbers can change when devices are plugged in or out (virtual mics like Voicemod re-number them).
  If your mic stops working, re-pick it here and save.

**System tray**
- Closing the window (the X) hides it to the tray; Jarvis keeps running. The icon changes colour with the status.
- Left-click the icon, or right-click > **Show**, to bring the window back. Right-click also has **Pause** / **Resume**
  and **Quit**. Use **Quit** to really exit.

**Troubleshooting the app**
- `start_jarvis_gui.bat` hides the console. To see errors, run `python jarvis_gui.py` in a terminal. Crashes are also
  written to `jarvis_gui.log` in this folder.
- Everything the app does runs in a background thread; the window only updates through a queue polled with `after()`,
  so it never freezes while Jarvis is thinking or speaking.

## Game-mode tip
While gaming, free the GPU: load **Qwen3 1.7B** in LM Studio with GPU offload set to 0 (CPU only),
set `LLM_MODEL=qwen3-1.7b`, and `WHISPER_DEVICE=cpu`, `WHISPER_MODEL=tiny` or `base`.

## Gaming mode (for streamers and older graphics cards)
Gaming mode keeps Jarvis out of your game's and stream's way (handy on a GTX 1070 that's also streaming).
Turn it on with the **Gaming mode** switch on the Assistant tab, by saying "*Jarvis, gaming mode on*"
(or "turn off gaming mode"), or with `JARVIS_GAMING_MODE=1` in `.env`. While it's on:
- Speech recognition (Whisper) and the Kokoro voice run on the **CPU** (int8), leaving the graphics card alone.
- Whisper uses the smaller **base** model (change it with `JARVIS_GAMING_WHISPER_MODEL`, e.g. `tiny`).
- Jarvis runs at **below-normal priority**, so the game always wins.
- **Barge-in is off** (it keeps listening for the wake word while talking, which costs a bit of CPU).

Priority and barge-in change straight away. Moving Whisper/Kokoro to the CPU needs a **restart** of Jarvis.
The switch/voice choice is remembered in `.cache/settings.json` and overrides `.env`.
For the language model itself, see the tip above (LM Studio GPU offload).

## Desktop shortcut
Double-click **`create_desktop_shortcut.bat`** (or press **Create desktop shortcut** in Settings). It puts a
**Jarvis** icon on your Desktop (also works with a OneDrive Desktop) that opens the app with no black console window.
No admin rights needed and nothing is downloaded. Run it again any time, e.g. after moving the folder.
The icon is `jarvis.ico` (redraw it with `python make_icon.py`).

## OBS Studio control
Jarvis can drive OBS Studio 28+ through its built-in WebSocket server (`obsws-python`, installed via requirements.txt).

Setup:
1. In OBS open **Tools > WebSocket Server Settings** and tick **Enable WebSocket server**. Leave the port at 4455.
2. Leave **Enable Authentication** on, click **Show Connect Info**, and copy the **Server Password**.
3. Paste it into `.env` as `OBS_PASSWORD=...`. If authentication is off, leave it blank.
4. Set `OBS_MIC_INPUT` to your mic's name in the OBS Audio Mixer. The default is `Mic/Aux`.

(In the desktop app you can do steps 3 and 4 in the Settings tab instead: paste the password, then click **Find in OBS**.)

Things to say: "start recording", "stop recording", "go live" / "start streaming", "stop streaming",
"switch to scene gameplay", "list scenes", "mute my mic" / "unmute my mic", "clip that" / "save replay"
(the Replay Buffer must be started in OBS), and "am I live?" / "am I recording?".
Jarvis only starts streaming when you clearly say "go live" or "start streaming". Scene names don't need to match exactly.
If OBS isn't running or WebSocket is off, Jarvis tells you so instead of failing.

## Free offline voice (Kokoro)

[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) is a free voice that runs on your own PC (no account,
no internet after the first download). It sounds much more natural than the built-in Windows voice. It's optional.

1. Open a terminal in the Jarvis folder (the same place you ran `pip install -r requirements.txt`) and run:
   ```
   pip install -r requirements-kokoro.txt
   ```
   This is a big download (it includes PyTorch), so give it a few minutes. Kokoro needs Python 3.10–3.12 (3.12 is fine).
2. Start Jarvis. In the desktop app go to **Settings**, set **Voice engine** to **Kokoro (free, offline)**,
   pick a **Kokoro voice** (`bm_george` is the default British butler voice; also `bm_lewis`, `bm_daniel`, `bm_fable`,
   or female `bf_emma`, `bf_isabella`, `bf_alice`, `bf_lily`) and press **Preview** to hear it. Then press **Apply** next to each one
   and restart Jarvis. (Or set `JARVIS_TTS_ENGINE=kokoro` and `KOKORO_VOICE=bm_george` in `.env`.)
3. The **first time** it speaks it downloads the voice model (about 300 MB) and a small English language pack,
   so the first sentence can take a minute. After that it starts quickly.

Notes:
- If you leave Voice engine on **Automatic**, Jarvis uses ElevenLabs when you have a key, otherwise Kokoro if it's
  installed, otherwise the Windows voice. If ElevenLabs fails (no internet, out of credits) it falls back to Kokoro, then the Windows voice.
  If Kokoro isn't installed it is simply skipped. Barge-in (saying the wake word to cut it off) works with Kokoro too.
- `KOKORO_DEVICE=auto|cpu|cuda` (default `auto`). Kokoro is small and runs fine on the CPU. The normal
  `pip install` gives you CPU-only PyTorch on Windows, which is fine. To use an NVIDIA card, install a CUDA build of
  PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) (RTX 50-series cards like the 5070 Ti need the
  CUDA 12.8 or newer build). If the GPU doesn't work it falls back to the CPU automatically.
- Streaming on an older card (e.g. GTX 1070)? Set `KOKORO_DEVICE=cpu` so the voice doesn't compete with your game/stream for the GPU.
- **espeak-ng:** you normally do **not** need to install it. The `kokoro` package pulls in `espeakng-loader`, which
  ships its own copy of espeak-ng (used only for unusual words that aren't in Kokoro's dictionary). If you ever see an
  espeak error in the log, install espeak-ng from its [releases page](https://github.com/espeak-ng/espeak-ng/releases)
  (download the `.msi`, run it) and restart Jarvis.

## Troubleshooting
- "Can't reach my brain" → LM Studio server isn't running / wrong `LLM_BASE_URL`.
- Wake word too sensitive / deaf → tune `JARVIS_WAKE_THRESHOLD` (0.3–0.7).
- It interrupts itself → raise `JARVIS_BARGE_IN_THRESHOLD` (e.g. 0.85), use headphones, or set `JARVIS_BARGE_IN=0`.
- Transcription slow → `WHISPER_MODEL=base` (or `tiny`).
- Feels slow after you finish talking → lower `JARVIS_SILENCE_SECONDS` (default `0.75`). Raise it (e.g. `1.2`) if it cuts you off mid-sentence.
- Recording never stops in a noisy room → raise `JARVIS_VAD_RMS`.
- ElevenLabs speech starts as soon as the first audio arrives (streamed), instead of waiting for the whole reply.
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

## Choosing a wake sound
There are 39 built-in wake sounds plus `random` (a different one each wake). Pick one any of these ways:

- **By voice** (instant, no LLM needed):
  - "Change your sound to piano", "set the wake sound to steel drum", "switch the sound to music box"
  - "Use the harp sound", "use the random sound"
  - "Play the sitar sound" / "preview the synth sound" (plays it once without changing it)
  - "List wake sounds" / "what wake sounds do you have?"
  - Spoken forms work: "steel drum", "music box", "synth", "guitar", "drum" (taiko), "sax", and near-misses are fuzzy-matched.
- **From the console:** `python jarvis_assistant.py --sounds` shows a numbered list. Type a number to preview it, `set N` to choose it, `list` to reprint, or `q` to quit.
- **In the setup wizard** (`--setup`): there's a wake-sound step; type a number, or `p N` to preview first. It writes `JARVIS_WAKE_SOUND_STYLE` in `.env`.

A choice made by voice or `--sounds` is saved in `.cache/settings.json` (`"wake_sound"`) and takes effect immediately. It overrides `JARVIS_WAKE_SOUND_STYLE` in `.env`; the default is `bell`. Unknown names fall back to `bell` (with a log warning).

| Category | Sounds |
|---|---|
| Classic | `bell` (Bell), `chime3` (Three-note chime), `beep` (Two-tone beep) |
| Space | `warp` (Warp), `comms` (Starship comms), `cosmos` (Cosmos), `sonar` (Sonar ping), `blip` (Sci-fi blip) |
| Instruments | `piano` (Piano), `acoustic_guitar` (Acoustic guitar), `harp` (Harp), `violin` (Violin), `cello` (Cello), `trumpet` (Trumpet), `sax` (Saxophone), `flute` (Flute), `clarinet` (Clarinet), `marimba` (Marimba), `xylophone` (Xylophone), `vibraphone` (Vibraphone), `steel_drum` (Steel drum), `kalimba` (Kalimba), `music_box` (Music box), `banjo` (Banjo), `sitar` (Sitar), `synth_lead` (Synth lead), `organ` (Organ), `taiko` (Taiko drum) |
| Fun | `chug` (Metal palm-mute chug), `powerup` (Power-up), `chirp` (Bird chirp) |
| Radio | `radio_click` (Radio click, push-to-talk), `roger_beep` (Roger beep, the "over" beep), `talkabout` (Walkie talkie call tone), `police_radio` (Police radio chirp), `cb_radio` (CB radio kerchunk), `ham_radio` (Ham radio courtesy tone), `military_radio` (Military radio double click), `squelch_tail` (Squelch tail, "over and out") |
| Other | `random` (a different one each wake) |

The radio sounds are made up by the program in the style of each kind of handheld radio; they are not recordings of real radios.

**Can't hear a short sound?** Some USB and Bluetooth headphones swallow the first moment of any new sound while they wake up, so very short sounds (like `beep` or `blip`) can vanish. Jarvis plays a short silent lead-in first (`JARVIS_WAKE_SOUND_PREROLL`, default `0.2` seconds). Raise it (for example `0.4`) if a sound still seems missing, or set `0` to turn it off.

## Wake sound and quiet song controls
- `JARVIS_WAKE_SOUND_STYLE` picks the wake sound: `bell` (default), `warp`, `comms`, `cosmos` (space sounds), `chug` (metal palm-mute), `chime3`, `sonar`, `powerup`, `chirp`, `blip`, or `beep`; or an instrument: `piano`, `acoustic_guitar`, `harp`, `violin`, `cello`, `trumpet`, `sax`, `flute`, `clarinet`, `marimba`, `xylophone`, `vibraphone`, `steel_drum`, `kalimba`, `music_box`, `banjo`, `sitar`, `synth_lead`, `organ`, `taiko`; a radio: `radio_click`, `roger_beep`, `talkabout`, `police_radio`, `cb_radio`, `ham_radio`, `military_radio`, `squelch_tail`; or `random`. See [Choosing a wake sound](#choosing-a-wake-sound).
- `JARVIS_WAKE_SOUND_FILE` plays your own short `.wav` instead (up to 3 seconds).
- Song commands like "next song", "skip this track", "previous song", and "pause the music" run instantly and silently. Set `JARVIS_QUIET_MEDIA=0` if you want him to confirm out loud.
