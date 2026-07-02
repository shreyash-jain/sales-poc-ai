"""FastAPI app for Test B (real Plivo phone call).

  GET/POST /answer  -> Plivo Answer XML opening a bidirectional <Stream> to /ws
  WS       /ws       -> hands the socket to the Pipecat Plivo transport + bot

All telephony/transport wiring lives here; provider services live in services.py
(swap providers there without touching this file). Run with:

    uvicorn server:app --port 8000
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, WebSocket
from loguru import logger

from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from bot import run_bot

load_dotenv()

# Plivo streams 8 kHz mu-law; keep the whole phone leg at 8 kHz.
PLIVO_SAMPLE_RATE = 8000

app = FastAPI()


def _ws_url() -> str:
    host = os.getenv("NGROK_HOST", "").strip().rstrip("/")
    if not host:
        logger.error("NGROK_HOST is not set — the <Stream> URL will be invalid.")
    if host.startswith("http"):
        host = host.split("://", 1)[1]
    return f"wss://{host}/ws"


def _answer_xml() -> str:
    # Exact element shape from the official Plivo x Pipecat example:
    # bidirectional stream, 8 kHz mu-law, wss URL as the element TEXT.
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        '  <Stream bidirectional="true" keepCallAlive="true" '
        f'contentType="audio/x-mulaw;rate=8000">{_ws_url()}</Stream>\n'
        "</Response>"
    )


@app.get("/health")
async def health():
    return {"status": "ok", "ws_url": _ws_url()}


@app.api_route("/answer", methods=["GET", "POST"])
async def answer(request: Request):
    logger.info(f"/answer {request.method} from {request.client.host if request.client else '?'} -> {_ws_url()}")
    return Response(content=_answer_xml(), media_type="application/xml")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket accepted; reading Plivo start message...")

    # Reads the first messages off the socket, detects the provider, and pulls
    # stream_id / call_id out of Plivo's "start" event.
    transport_type, call_data = await parse_telephony_websocket(websocket)
    logger.info(f"Telephony={transport_type} call_data={call_data}")

    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
    # auto_hang_up needs Plivo creds; disable it if they're absent so /ws still runs.
    serializer_params = None
    if not (auth_id and auth_token):
        logger.warning("PLIVO_AUTH_ID/TOKEN missing — disabling serializer auto_hang_up.")
        serializer_params = PlivoFrameSerializer.InputParams(auto_hang_up=False)

    serializer = PlivoFrameSerializer(
        stream_id=call_data["stream_id"],
        call_id=call_data.get("call_id"),
        auth_id=auth_id,
        auth_token=auth_token,
        params=serializer_params,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,  # required for raw telephony audio
            serializer=serializer,
        ),
    )

    await run_bot(
        transport,
        handle_sigint=False,  # uvicorn owns signal handling
        greet_on_event=True,
        sample_rate=PLIVO_SAMPLE_RATE,
    )
    logger.info("Pipeline finished; call ended.")
