# Run & Test — Plivo + Pipecat voice demo

Stack: **Sarvam AI** (STT + TTS) · **OpenRouter** (LLM) · **Plivo** (telephony) · Pipecat 1.4.0.

See [readme.md](readme.md) for the full build brief. This file is just run/test steps.

## Files
| file | role |
|---|---|
| [services.py](services.py) | STT/LLM/TTS factory — the ONE place to swap providers |
| [bot.py](bot.py) | Pipecat pipeline; runs under local-mic **or** Plivo transport |
| [server.py](server.py) | FastAPI: `/answer` (Plivo XML) + `/ws` (Pipecat websocket) |
| [make_call.py](make_call.py) | places the outbound Plivo test call |
| [.env](.env) | your secrets (fill this in) |

## One-time setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # already installed in ./venv
cp .env.example .env                    # then fill in keys
```
> macOS: the local-mic test needs portaudio — `brew install portaudio` (already done).

Fill `.env`:
- **Test A** needs only `SARVAM_API_KEY` + `OPENROUTER_API_KEY`.
- **Test B** additionally needs `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`, `PLIVO_FROM_NUMBER`, `TEST_TO_NUMBER`, `NGROK_HOST`.

---

## Test A — local mic (no Plivo, no server)
Proves STT → LLM → TTS end-to-end with zero telephony.
```bash
source venv/bin/activate
python bot.py
```
- The bot greets you first; speak after the greeting.
- Success = you talk, it transcribes, replies, and speaks back with decent turn-taking.
- Iterate on `SYSTEM_PROMPT` in [bot.py](bot.py). `Ctrl+C` to stop.
- Pick the mic/speaker via your OS default input/output device. Use headphones to avoid the bot hearing itself.

Tune in `.env`: `OPENROUTER_MODEL`, `SARVAM_TTS_VOICE`, `SARVAM_LANGUAGE` (e.g. `hi-IN`), `LOCAL_SAMPLE_RATE`.

---

## Test B — real Plivo phone call

### B0. Provision an India (+91) caller-ID number (one-time, KYC)
This account is **India-region** (only +91 numbers are rentable; US/UK/CA return none), and India numbers are **business-only + KYC-gated**. No compliance application exists yet.

**Documents to upload** (confirmed via Plivo's compliance API): **Certificate of Incorporation** (or Udyam), **Business PAN** and/or **GST Certificate**, plus **Customer Use Case** = `Direct`.

**Steps:**
1. Console → **Phone Numbers → Regulatory Compliance** → create a **Business End User** → create a **Compliance Application** for **India / Local** → upload the docs above → submit.
2. Wait for approval — typically **~15 min to 1 business day** for 022/080 local numbers.
3. Console → **Phone Numbers → Buy Numbers** → Country **India**, type **Local**, capability **Voice** → **Buy**. Put it in `.env` as `PLIVO_FROM_NUMBER` (E.164, e.g. `+91xxxxxxxxxx`).
4. Production-only (not needed for a single test to your own phone): **DLT/TCCCPR** registration + correct number series for commercial outbound voice at scale.

### B1. Place the call
1. **Start the server**
   ```bash
   source venv/bin/activate
   uvicorn server:app --port 8000
   ```
2. **Expose it with ngrok** (India region keeps media in-country):
   ```bash
   ngrok http --region in 8000
   ```
   Copy the **host** (no `https://`) into `.env` → `NGROK_HOST`, e.g. `abc123.ngrok-free.app`. Restart uvicorn after editing `.env`.
   Sanity check: open `https://<NGROK_HOST>/health` → should show your `wss://…/ws` URL.
3. Set `TEST_TO_NUMBER` to your mobile (E.164, `+91...`). If outbound is rejected as "not verified" (trial-limited), add it under Console → **Phone Numbers → Sandbox Numbers → Add Sandbox Number** (OTP).
4. **Place the call**:
   ```bash
   python make_call.py
   ```
   Your phone rings → answer → bot greets you → speak → it replies in voice. Watch the uvicorn logs for the call lifecycle (`/answer` hit → WebSocket accepted → `Telephony=plivo` → greeting → clean hangup).

> ngrok is installed (v3.39.8). First-time only: `ngrok config add-authtoken <token>` (from your ngrok dashboard).
> India domestic calls require in-country media — keep `ngrok http --region in 8000` and run uvicorn on your India-based laptop.

---

## Swapping providers later (e.g. Deepgram/Cartesia, or back to Sarvam)
Edit **only** [services.py](services.py) — `create_stt` / `create_llm` / `create_tts`. The pipeline, transport, and telephony code never reference a vendor SDK.

## Troubleshooting
- **`KeyError: 'SARVAM_API_KEY'`** — `.env` not filled / not loaded; ensure you ran from the project dir.
- **No audio on the phone / call drops** — check `NGROK_HOST` has no `https://` and no trailing slash; confirm ngrok is in the India region.
- **Mic test silent** — wrong default input device, or the bot is hearing its own output (use headphones).
- **Plivo "number not verified"** — verify `TEST_TO_NUMBER` in the console (trial restriction).
