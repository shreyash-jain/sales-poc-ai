# Build: Plivo + Pipecat Voice-AI Demo (runs locally, no cloud deploy)

> Paste this whole file into your IDE's AI agent as the build brief.

## Objective
Build a minimal but working AI voice-agent demo that runs **entirely on my laptop** and can place/receive a real phone call through **Plivo**. The bot answers a call, greets the caller, transcribes speech (STT), generates a reply (LLM), and speaks it back (TTS) — full duplex. The goal is to prove the end-to-end audio loop on Plivo's free trial **before** any production work. **Do NOT deploy to a cloud server** — run locally and expose via ngrok.

## START HERE — adapt the official example, don't write from scratch
Base the telephony integration on the **official Pipecat Plivo example and guide**, and adapt it. Pull the exact, current API (serializer class, transport params, audio framing, Answer XML) from these — they are authoritative and more current than any snippet below:
- Plivo × Pipecat integration guide: https://www.plivo.com/docs/voice-agents/audio-streaming/integration-guides/pipecat/overview
- Pipecat examples repo — use the **`plivo-chatbot`** example: https://github.com/pipecat-ai/pipecat-examples
- Pipecat `FastAPIWebsocketTransport`: https://docs.pipecat.ai/server/services/transport/fastapi-websocket
- Pipecat quickstart (for the local mic-test bot): https://docs.pipecat.ai
- ngrok: https://ngrok.com/docs

## Architecture
```
Phone ↔ Plivo ↔ (WebSocket audio stream) ↔ my FastAPI app ↔ Pipecat pipeline ↔ [STT → LLM → TTS]
```
- Plivo routes the call and streams real-time audio over a WebSocket to my app.
- A FastAPI app exposes (1) an **Answer URL** returning Plivo XML that opens a `<Stream>` to my WebSocket, and (2) the **WebSocket endpoint** Pipecat consumes.
- Pipecat (`FastAPIWebsocketTransport` + Plivo frame serializer + Silero VAD) runs STT → LLM → TTS.
- **ngrok** exposes my local app to Plivo with a public https/wss URL.

## Stack
- Python 3.11+, FastAPI + uvicorn
- `pipecat-ai` with the **Plivo serializer** + `FastAPIWebsocketTransport`; include Silero VAD
- `plivo` (Python SDK) — to place the outbound test call
- **STT:** Deepgram · **LLM:** OpenAI `gpt-4o-mini` (or Gemini Flash) · **TTS:** Cartesia (or ElevenLabs) — these are the demo defaults from the official guide
- `python-dotenv`, ngrok
- **Keep STT/TTS isolated in one module.** For India production we will later swap in **Sarvam** STT/TTS via a custom Pipecat service — structure the code so that swap is a one-file change and does not touch transport/telephony code.

## Project structure
```
plivo-voice-demo/
├── .env.example
├── requirements.txt
├── services.py          # STT/LLM/TTS factory — the ONE place to swap providers (Deepgram→Sarvam later)
├── bot.py               # Pipecat pipeline; runnable with TWO transports (local mic + Plivo websocket)
├── server.py            # FastAPI: /answer (Plivo XML) + /ws (WebSocket for Pipecat)
├── make_call.py         # Places an outbound test call via Plivo REST API
└── README.md            # run + test steps
```

## Components to build
1. **services.py** — factory functions returning the STT, LLM, TTS service instances from env config. This isolation is mandatory (Sarvam swap later).
2. **bot.py** — the Pipecat pipeline (transport → STT → LLM → TTS). Accept the transport as a parameter so the same pipeline runs under:
   - a **local transport** (mic/browser) for Test A, and
   - `FastAPIWebsocketTransport(serializer=<PlivoSerializer>, vad=Silero)` for Test B.
   - System prompt: a short, friendly **first-touch qualification bot for an education brand (Vacademy)**. Replies must be 1–2 sentences max (low latency, natural turn-taking). Bot speaks a greeting first on connect.
3. **server.py** —
   - `GET/POST /answer` → returns Plivo Answer XML with a **bidirectional `<Stream>`** pointing at `wss://<NGROK_HOST>/ws` (exact XML from the Plivo guide).
   - `WS /ws` → hand the socket to the Pipecat transport and start the pipeline.
   - Read the public host from env `NGROK_HOST` so the XML uses the correct wss URL.
4. **make_call.py** — use the Plivo SDK + `PLIVO_AUTH_ID/TOKEN` to place an outbound call FROM `PLIVO_FROM_NUMBER` TO `TEST_TO_NUMBER` (my verified mobile), with `answer_url = https://<NGROK_HOST>/answer`.

## .env.example
```
PLIVO_AUTH_ID=
PLIVO_AUTH_TOKEN=
NGROK_HOST=                 # e.g. abc123.ngrok-free.app  (host only, no https://)
DEEPGRAM_API_KEY=
OPENAI_API_KEY=            # or GEMINI_API_KEY
CARTESIA_API_KEY=         # or ELEVENLABS_API_KEY
PLIVO_FROM_NUMBER=        # Plivo caller-ID number (India number needs KYC — see notes)
TEST_TO_NUMBER=           # my verified mobile, E.164 (+91...)
```

## How to test (two phases — do A first)

### Test A — local pipeline, NO Plivo, NO server exposure (works today)
Proves STT→LLM→TTS with zero telephony dependencies.
1. `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
2. Run `bot.py` with the **local transport** and talk to the bot through the browser/mic.
3. Success = I speak, it transcribes, replies, and speaks back. Iterate on the prompt here.

### Test B — real Plivo phone call (needs a Plivo caller-ID number)
1. Start the app: `uvicorn server:app --port 8000`
2. Start ngrok **in the India region** (keeps media in-country for Plivo India anchoring):
   `ngrok http --region in 8000` → copy the https host into `.env` `NGROK_HOST`.
3. Plivo console → **Verify** → add my mobile as a verified number (trial allows outbound only to verified numbers).
4. Plivo console → **Applications** → create an app with Answer URL `https://<NGROK_HOST>/answer` and assign it — OR pass `answer_url` directly in `make_call.py`.
5. `python make_call.py` → my phone rings → I answer → bot greets me → I speak → it replies in voice.

## Constraints / gotchas (Plivo free trial + India) — honor these
- **Trial: outbound only to VERIFIED numbers.** Verify my mobile before Test B.
- **Outbound CPS = 2**, monthly cap ₹80,000 — fine for testing.
- **India caller-ID number needs KYC.** Placing an outbound *India* call requires an India number as caller ID, which requires Plivo KYC to clear. So **Test A needs nothing; Test B may be blocked until a number is provisioned.** Do not block on this — ship Test A first.
- **Media anchoring (India):** my laptop is in India + ngrok is in the India region, so call media stays in-country. In production, host in **ap-south-1**. Never route media through a non-India endpoint or the call drops with a media-anchoring error.
- **Audio framing:** Plivo streams 8 kHz audio; the serializer/transport audio format + framing must match what Plivo sends. The official example handles this — **do not hand-roll the audio format**; reuse the example's serializer config.
- **No real India DID yet:** we have no 140/160-series numbers until DLT clears; this demo only proves the tech loop, not compliant dialing.

## Acceptance criteria
- **Test A:** local mic conversation works — speak, transcribe, reply, speak back, good turn-taking.
- **Test B:** `make_call.py` rings my verified phone; on answer I hear a greeting; I speak and the bot replies audibly both directions; server logs show the call lifecycle and a clean hangup.
- STT/LLM/TTS are isolated in `services.py` so Sarvam can be swapped in later without touching `server.py`/transport code.
- The Pipecat pipeline in `bot.py` runs under both the local transport and the Plivo websocket transport from the same code.