# Jarvis setup guide (no coding needed)

This takes about 20–30 minutes, mostly waiting for downloads. You only do it once.

## 1. Python (automatic)
Jarvis runs on Python. **You don't need to install it yourself**: if it's missing, `setup.bat` (step 4) asks
"Install Python 3.12 now?". Click **Yes** and Python's own installer runs by itself (about 25 MB, a minute or two,
no admin needed). If you already have Python 3.10 or newer, this step is skipped.

Prefer doing it by hand? Download the Python 3.12 "Windows installer (64-bit)" from
https://www.python.org/downloads/windows/ and on the first screen tick **"Add python.exe to PATH"**.

## 2. Install LM Studio (Jarvis's "brain")
Jarvis thinks using a free AI model that runs on your own PC through LM Studio.
1. Download and install LM Studio from https://lmstudio.ai
2. In LM Studio, search for **Qwen3 4B** and download it (the `Q4_K_M` version fits a GTX 1070 well).
3. Open the **Developer** tab, load the model, and click **Start Server**.
   LM Studio must be running with the server started whenever you use Jarvis.

## 3. Download Jarvis
1. Open https://github.com/Cam5Hero/jarvis_assistant
2. Click the green **Code** button, then **Download ZIP**.
3. Right-click the ZIP, choose **Extract All...**, and put it somewhere simple like `Documents\jarvis_assistant`.
   (Don't run it from inside the ZIP.)

## 4. Run setup
1. Open the extracted folder and double-click **`setup.bat`**.
   If Windows shows "Windows protected your PC", click **More info** then **Run anyway**.
   A black window flashes for a moment, then the **Jarvis setup** window opens.
2. Tick **"Also install the free offline Kokoro voice?"** if you want it (big download; you can also add it later
   from Settings > Voice), then click **Install**.
3. Wait while the progress bar fills (5-15 minutes). Setup also puts a **Jarvis** shortcut on your Desktop.
4. Click **Start Jarvis**. From now on, start it with the **Jarvis** Desktop icon (or double-click `run_jarvis.bat`).
   If something fails, setup shows what went wrong and a **Retry** button. Details are in `setup.log`.

The first start downloads the wake-word and speech models once, so give it a minute.

## 5. Settings (keys and options)
Open Jarvis, then open **Settings**. Everything you type there is saved on your PC in a file called `.env`.
Never share that file or screenshots of your keys.

- **Brain tab:** click **Refresh** and pick the model you loaded in LM Studio. No key needed.
- **Voice tab:** pick your microphone and a voice engine (Automatic, ElevenLabs, Kokoro, or Windows voice).
- **Account tab – ElevenLabs (optional, nicest voice):**
  1. Make a free account at https://elevenlabs.io/sign-up
  2. Go to https://elevenlabs.io/app/settings/api-keys and create an API key. Turn on the permissions
     **Text to Speech**, **Speech to Text** and **Voices (Read)**.
  3. Paste the key into Settings and press **Apply**.
  4. **Free plan:** you can only use ElevenLabs' **default (premade) voices**. Voices from the Voice Library
     need a paid plan. Free credits are limited each month; when they run out Jarvis switches to Kokoro or the Windows voice automatically.
  - No ElevenLabs? That's fine. Jarvis uses Kokoro (if installed) or the built-in Windows voice.
- **OBS tab:** see the next step.

## 6. OBS controls (optional, for streaming)
Lets you say things like "start recording", "go live", "switch to scene gameplay", "mute my mic", "clip that".
1. In OBS Studio (version 28 or newer) open **Tools > WebSocket Server Settings**.
2. Tick **Enable WebSocket server**. Leave the port at **4455**.
3. Keep **Enable Authentication** on, click **Show Connect Info**, and copy the **Server Password**.
4. In Jarvis Settings > **OBS**, paste the password, press **Apply**, then click **Find in OBS** to pick your mic.
5. For "clip that", start the **Replay Buffer** in OBS first.

Jarvis only goes live when you clearly say "go live" or "start streaming".

## Using Jarvis
Say **"Hey Jarvis"**, wait for the chime, then ask your question. Headphones work best.

## Troubleshooting
- **Python couldn't be installed automatically:** install Python 3.12 by hand (step 1) with "Add python.exe to PATH"
  ticked, then **restart your PC** and run `setup.bat` again.
- **Package install failed:** check your internet and press **Retry** (or run `setup.bat` again; it's safe to re-run). Details are in `setup.log`.
  If you still see red errors, take a screenshot and send it to whoever helps you.
- **Jarvis says it can't reach LM Studio:** open LM Studio, load the model, click **Start Server** in the Developer tab, then press **Refresh** in Settings > Brain.
- **Jarvis doesn't hear "Hey Jarvis":** check the microphone in Settings > Voice and that Windows allows mic access
  (Windows Settings > Privacy & security > Microphone).
- **Jarvis stops listening before you finish talking (or doesn't hear you):** raise **Mic sensitivity** in
  Settings > Voice > Listening time (1-10, default 5). Lower it if background noise keeps him listening.
- **ElevenLabs key refused:** recreate the key with the three permissions above, or pick a default voice.
- **Games feel slower while Jarvis runs:** see "Gaming mode" in the README (e.g. use a smaller Whisper model).
- **OBS commands don't work:** make sure OBS is open, the WebSocket server is enabled, and the password matches.
- **Jarvis window doesn't open:** open the folder, click the address bar, type `cmd`, press Enter, and run
  `.venv\Scripts\python jarvis_gui.py` to see the error message.
- **Moved the folder?** Run `setup.bat` again (it recreates the Desktop shortcut).
- **Updating:** if you downloaded the ZIP, download the new ZIP, extract it, copy your old `.env` file into the new folder, and run `setup.bat`.

## Updating Jarvis
You don't need to download the ZIP again. When Jarvis starts it quietly checks for a new version. If one is ready, a window says **"A new version of Jarvis is ready"** — click **Update**, wait for it to finish, then click **Restart Jarvis now**. Your settings, API keys (`.env`) and downloaded voices/models are kept. You can also check any time in **Settings > Account > Check for updates**. If something goes wrong, details are in `update.log` in the Jarvis folder.

## Uninstall
1. Close Jarvis (right-click the tray icon > Quit), then double-click **`uninstall.bat`** in the Jarvis folder.
2. Read the list, type **Y** and press Enter. It removes:
   - the **Jarvis** shortcuts on your Desktop and in Startup (only ones pointing to this folder)
   - the `.venv` folder (everything setup.bat installed), `.cache`, `__pycache__` and `jarvis_gui.log`
   - the downloaded Kokoro and Whisper models (only Jarvis's ones, in `%USERPROFILE%\.cache\huggingface\hub`)
3. It then asks whether to delete `.env` (your settings and API keys). Press **N** to keep it (recommended if you might reinstall).
4. At the end a pop-up asks **"Delete the Jarvis folder too?"**. Click **Yes** to move the whole folder (including `.env` if you kept it) to the **Recycle Bin** — you can still restore it from there. Click **No** to keep it.
5. **Python and LM Studio are not removed.** Uninstall them from **Windows Settings > Apps > Installed apps** if you don't need them.
   Remove the models you downloaded in LM Studio first if you want that disk space back.
