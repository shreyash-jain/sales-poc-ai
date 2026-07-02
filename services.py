"""Provider factory — the ONE place to swap STT / LLM / TTS.

Current stack: Sarvam AI (STT + TTS) + OpenRouter (LLM).

This isolation is deliberate: bot.py / server.py / make_call.py never import a
vendor SDK. To swap a provider later (e.g. Deepgram STT, Cartesia TTS, or a new
vendor) change ONLY this file — the pipeline, transport and telephony code stay
untouched.
"""

import os

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language


def _language(code: str | None = None) -> Language:
    """Map a BCP-47 code (e.g. 'en-IN', 'hi-IN') to a Pipecat Language enum.
    Falls back to the SARVAM_LANGUAGE env, then English."""
    code = (code or os.getenv("SARVAM_LANGUAGE", "en-IN")).strip()
    try:
        return Language(code)  # enum value IS the BCP-47 code
    except ValueError:
        return Language.EN_IN


def create_stt(sample_rate: int, language: str | None = None) -> SarvamSTTService:
    """Sarvam streaming STT. Accepts 8000 (telephony) or 16000 (local mic)."""
    model = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
    settings_kwargs = {"model": model}
    # saaras:v2.5 auto-detects language and rejects an explicit language param.
    if model != "saaras:v2.5":
        settings_kwargs["language"] = _language(language)
    return SarvamSTTService(
        api_key=os.environ["SARVAM_API_KEY"],
        sample_rate=sample_rate,
        settings=SarvamSTTService.Settings(**settings_kwargs),
        # Biggest latency lever: the turn-stop strategy holds the user turn for
        # max(0, ttfs_p99 - vad_stop_secs) before handing to the LLM, because
        # Sarvam never flags transcripts finalized=True. Default 1.17 => ~0.97s
        # dead air every turn. 0.5 => ~0.30s. Tune 0.4-0.6 for your line quality.
        ttfs_p99_latency=float(os.getenv("SARVAM_TTFS_P99", "0.5")),
    )


def create_llm():
    """LLM factory. LLM_PROVIDER selects the backend:
      - "sarvam"  (default): Sarvam-30b, India-hosted -> ~232ms TTFB (~6x faster
        than OpenRouter from India). Reasoning model: we MUST disable "thinking"
        (reasoning_effort=null) or it streams reasoning instead of the reply.
      - "openrouter": google/gemini-3.1-flash-lite fallback (~850ms+ from India).
    Both are OpenAI-compatible and authenticate with their respective keys.
    """
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "150"))
    # Temperature 0.6 (not 0.3): low temp made replies near-deterministic, so the
    # bot repeated the same line verbatim across turns. Higher = more varied/natural.
    temp = float(os.getenv("LLM_TEMPERATURE", "0.6"))
    if os.getenv("LLM_PROVIDER", "sarvam").lower() == "sarvam":
        return OpenAILLMService(
            api_key=os.environ["SARVAM_API_KEY"],  # same key as Sarvam STT/TTS
            base_url="https://api.sarvam.ai/v1",
            settings=OpenAILLMService.Settings(
                model=os.getenv("SARVAM_LLM_MODEL", "sarvam-105b"),
                temperature=temp,
                max_tokens=max_tokens,
                # reasoning_effort=null disables Sarvam's hybrid "thinking" so it
                # streams the spoken reply directly. Passed via extra_body so the
                # literal null reaches the request body (the SDK drops None kwargs).
                extra={"extra_body": {"reasoning_effort": None}},
            ),
        )
    return OpenRouterLLMService(
        api_key=os.environ["OPENROUTER_API_KEY"],
        settings=OpenRouterLLMService.Settings(
            model=os.getenv("OPENROUTER_MODEL", "google/gemini-3.1-flash-lite"),
            temperature=temp,
            max_tokens=max_tokens,
        ),
    )


def create_tts(sample_rate: int, language: str | None = None) -> SarvamTTSService:
    """Sarvam streaming TTS (bulbul). Matches the pipeline sample rate."""
    return SarvamTTSService(
        api_key=os.environ["SARVAM_API_KEY"],
        sample_rate=sample_rate,
        settings=SarvamTTSService.Settings(
            model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
            voice=os.getenv("SARVAM_TTS_VOICE", "priya"),
            language=_language(language),
            # Warmth: slightly slower + a touch of expressiveness. On bulbul:v3
            # only pace + temperature apply (pitch/loudness are ignored). pace<1
            # sounds friendlier; temperature ~0.5 keeps it natural but stable.
            pace=float(os.getenv("SARVAM_TTS_PACE", "0.95")),
            temperature=float(os.getenv("SARVAM_TTS_TEMP", "0.5")),
            enable_preprocessing=True,
            # NOTE: do NOT lower min_buffer_size below ~50 — Sarvam's TTS WS server
            # rejects small values ("Input parameters has to be a valid dictionary").
        ),
    )
