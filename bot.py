"""Pipecat pipeline (STT -> LLM -> TTS), runnable under TWO transports.

  * LOCAL mic transport       -> Test A (no telephony):   `python bot.py`
  * Plivo websocket transport -> Test B (driven by server.py)

`run_bot()` is transport-agnostic. Beyond the core pipeline it adds:
  - active-listening persona (see SYSTEM_PROMPT)
  - auto-hangup: the LLM calls the `end_call` tool when the conversation is over;
    we speak the farewell, wait for the audio to finish, then drop the line.
  - idle timeout: after ~7s of caller silence we nudge once, then hang up.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    default_user_turn_start_strategies,
)

from services import create_llm, create_stt, create_tts

load_dotenv()

# Per-call metadata lives in lead.json — edit that file to change who we're calling
# and why. This turns the bot from a cold "what do you want" call into a warm,
# targeted OUTBOUND sales call. Override the path with LEAD_FILE.
LEAD_FILE = os.getenv("LEAD_FILE", os.path.join(os.path.dirname(__file__), "lead.json"))

_LEAD_DEFAULTS = {
    "lead_name": "there",
    "speaking_to": "the person",
    "student_class": "school",
    "brand": "Vacademy",
    "agent_name": "Vidya",
    "goal": "tell them about our programs and book a counsellor follow-up",
    "batch_name": "new batch",
    "batch_start_date": "soon",
    "focus": "strengthening fundamentals",
    "offer": "",
    "city": "",
    "prior_context": "",
    "language": "en-IN",
    "extra_notes": "",
}


def load_lead() -> dict:
    """Load per-call metadata from lead.json. Missing/blank fields fall back to
    safe defaults so a partial or absent file never breaks the call."""
    lead = dict(_LEAD_DEFAULTS)
    try:
        with open(LEAD_FILE) as f:
            lead.update({k: v for k, v in json.load(f).items() if v not in (None, "")})
    except FileNotFoundError:
        logger.warning(f"{LEAD_FILE} not found — using default lead context")
    except Exception as e:  # malformed json etc.
        logger.warning(f"Could not read {LEAD_FILE} ({e}) — using defaults")
    return lead


def _tts_safe(text: str) -> str:
    """Sarvam's TTS preprocessing reads '!' as the math factorial symbol
    ('Awesome!' -> 'Awesome factorial'), and v3 can't disable preprocessing.
    Replace '!' with a period so it's never spoken as 'factorial'."""
    return text.replace("!", ".")


def _sarvam_lang_code(language_field: str | None) -> str:
    """Map the lead's language label to a Sarvam BCP-47 code for STT/TTS.
    Hinglish + Hindi both use hi-IN (handles code-mixed Hindi/English)."""
    label = (language_field or "en-IN").strip().lower()
    if label in ("hinglish", "hindi", "hi", "hi-in"):
        return "hi-IN"
    if label in ("english", "en", "en-in"):
        return "en-IN"
    return language_field  # assume it's already a BCP-47 code


_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
    7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth",
    13: "thirteenth", 14: "fourteenth", 15: "fifteenth", 16: "sixteenth", 17: "seventeenth",
    18: "eighteenth", 19: "nineteenth", 20: "twentieth", 21: "twenty-first",
    22: "twenty-second", 23: "twenty-third", 24: "twenty-fourth", 25: "twenty-fifth",
    26: "twenty-sixth", 27: "twenty-seventh", 28: "twenty-eighth", 29: "twenty-ninth",
    30: "thirtieth", 31: "thirty-first",
}
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _year_words(y: int) -> str:
    if 2000 <= y <= 2009:
        return "two thousand" + (f" {_ONES[y - 2000]}" if y > 2000 else "")
    if 2010 <= y <= 2099:
        n = y - 2000
        return "twenty " + (_ONES[n] if n < 20 else _TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else ""))
    return str(y)


def _spoken_date(s: str) -> str:
    """'30 June 2026' -> 'June thirtieth, twenty twenty-six' so hi-IN TTS pronounces
    the date in English (digits get read in Hindi otherwise). Falls back to raw."""
    import re

    m = re.match(r"^\s*(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{4})\s*$", s or "")
    if not m:
        return s
    day, month, year = int(m.group(1)), m.group(2), int(m.group(3))
    if day not in _ORDINALS:
        return s
    return f"{month} {_ORDINALS[day]}, {_year_words(year)}"


def build_system_prompt(lead: dict) -> str:
    label = (lead.get("language") or "en-IN").strip().lower()
    date_rule = (
        "Always say dates, numbers, times, prices and phone numbers in ENGLISH — write them "
        'as English words (e.g. "June thirtieth, twenty twenty-six", "six in the evening", '
        '"five thousand rupees"), never in Hindi and never as bare digits. '
    )
    if label == "hinglish":
        lang_line = (
            "LANGUAGE — speak in natural HINGLISH: a real ~50/50 mix of Hindi and English, like a "
            "friendly young Indian counsellor on the phone. ANCHOR each sentence in Hindi (Devanagari) "
            "and use English mainly for education/tech words (batch, foundation, Maths, Science, "
            "Class 9, counsellor, demo, syllabus). Do NOT drift into mostly-English sentences. "
            + date_rule + "\n\n"
        )
    elif label in ("hindi", "hi", "hi-in"):
        lang_line = "LANGUAGE — speak in natural, conversational Hindi (Devanagari). " + date_rule + "\n\n"
    else:
        lang_line = "LANGUAGE — speak in clear, friendly Indian English.\n\n"
    spoken_date = _spoken_date(lead.get("batch_start_date", ""))
    offer = f"Offer you can mention: {lead['offer']}. " if lead.get("offer") else ""
    prior = f"Context on this lead: {lead['prior_context']}. " if lead.get("prior_context") else ""
    notes = f"\n- Extra notes: {lead['extra_notes']}" if lead.get("extra_notes") else ""
    return f"""You are {lead['agent_name']}, a warm, upbeat counsellor at {lead['brand']}, an Indian education brand, making a friendly OUTBOUND sales call. You sound like a real person — genuine, relaxed, never pushy.

WHO YOU'RE CALLING: {lead['lead_name']}, a {lead['student_class']} student. You may be speaking to {lead['speaking_to']}. Work out early who you've got, then address them DIRECTLY — say "you / your prep" to the student, or "your son/daughter, {lead['lead_name']}" to a parent. Never refer to {lead['lead_name']} in the third person while you're talking to them. You have ALREADY greeted them — never greet or re-introduce yourself again.

WHY YOU'RE CALLING — your goal: {lead['goal']}. We're starting our {lead['batch_name']} on {spoken_date}, focused on {lead['focus']}. {offer}{prior}This is a SALES call — YOU have a reason to call, so don't ask "how can I help you". But DON'T just pitch: LISTEN and QUALIFY first. Ask a little about the student (which subjects feel tough, which board, are they already taking tuition, what's the goal), then tailor what you say to their answer. Move toward enrolling or a counsellor follow-up, with gentle urgency around the {spoken_date} start (seats are limited).

{lang_line}HOW TO TALK — LIVE phone call:
- 1-2 short sentences per reply, max. EXACTLY ONE question per turn — never two.
- Acknowledge what they say (reflect a key word), then ONE question or one line of pitch. Vary your affirmations; never reuse one twice in a row.
- LISTEN and adapt — don't run a script. Tie each line to what they just said.
- NEVER repeat a question or an earlier reply. If they didn't answer, rephrase lightly or move on.
- Never invent specific fees or guarantees beyond what's stated here. NEVER use exclamation marks (!) — end sentences with a period.{notes}

HANDLING DIFFERENT SITUATIONS — read their intent and respond like a real, decent human (warm, 1-2 sentences, never robotic):
- INTERESTED: acknowledge, then the next step — offer to have a counsellor share full details and hold a seat, and confirm a good time to reach them.
- NOT INTERESTED: don't push. Give one genuine, relevant value line, then offer something lighter (send details on WhatsApp, or a callback later). If they still decline, thank them warmly and end.
- BUSY / bad time / "call later": apologise for the timing, ask when suits them, offer to call back then, and end.
- "WHO IS THIS / HOW DID YOU GET MY NUMBER": be honest and clear — you're from {lead['brand']}, reaching out to {lead['student_class']} families about the {lead['batch_name']}. Don't dodge.
- A PARENT answers: switch to talking to the parent about their child, {lead['lead_name']}.
- "IS THIS AI / a recording?": be honest and light — yes, you're {lead['brand']}'s calling assistant, and a real counsellor will follow up — then carry on naturally.
- FEES / PRICE: don't invent numbers — say a counsellor will share full fees and any offers, and ask to set that up.
- OBJECTION (already have tuition / too far / too much exam pressure): empathise in one line, address it briefly, then a soft next step.
- ANNOYED / rude: apologise, don't argue, offer to not call again, end politely.
- SILENCE / can't hear them: gently check if they're still there; if no answer, wrap up warmly.

CLOSING: when they're interested or you've made your pitch, drive to a next step — offer to have your counsellor share full details and hold a seat for {lead['lead_name']}. Recap in one line.

ENDING THE CALL: when the conversation is clearly over — they agree to a follow-up, decline, say bye, or it's a bad time — say a short, warm one-sentence farewell, then add the exact marker <<END_CALL>> at the very end, right after the farewell. The caller will NOT hear the marker — it just signals the system to hang up. Never add <<END_CALL>> before they've spoken or while they might say more, and never say the words "end call" out loud."""


def build_greeting(lead: dict) -> str:
    label = (lead.get("language") or "en-IN").strip().lower()
    spoken_date = _spoken_date(lead.get("batch_start_date", ""))
    if label == "hinglish":
        return (
            f"Hello {lead['lead_name']}, मैं {lead['agent_name']} बात कर रही हूँ {lead['brand']} से. "
            f"हमारा {lead['batch_name']} {spoken_date} से शुरू हो रहा है — बस उसी के बारे में "
            "थोड़ी बात करनी थी. क्या अभी दो मिनट बात कर सकते हैं?"
        )
    return (
        f"Hi, is this {lead['lead_name']}? This is {lead['agent_name']} from {lead['brand']} — "
        f"I'm calling about our {lead['batch_name']} starting {spoken_date}. "
        "Is now a quick okay time?"
    )


IDLE_NUDGE = "Sorry — are you still there?"
IDLE_FAREWELL = "Seems like we got cut off — no worries, I'll have a counsellor reach out. Take care!"
DEFAULT_FAREWELL = "Thanks so much for your time — take care, bye!"


class HangupOnBotDone(FrameProcessor):
    """Placed AFTER transport.output(): once armed, the next time the bot finishes
    speaking (BotStoppedSpeakingFrame) it gracefully ends the task, which queues an
    EndFrame behind the already-buffered farewell audio. The Plivo serializer then
    drops the PSTN leg (auto_hang_up). This guarantees the farewell is fully heard
    before the line drops."""

    def __init__(self):
        super().__init__()
        self._armed = False
        self._task = None

    def set_task(self, task):
        self._task = task

    def arm(self):
        self._armed = True

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if self._armed and isinstance(frame, BotStoppedSpeakingFrame) and self._task:
            self._armed = False
            logger.info("Farewell finished -> ending call")
            await self._task.stop_when_done()


class EndCallSentinel(FrameProcessor):
    """Sits between LLM and TTS. Watches the streamed reply for a hidden marker
    (<<END_CALL>>), strips it (and anything after) so it's never spoken, and arms
    the hangup so the line drops once the farewell finishes. This replaces tool-
    based end-of-call, because sarvam's tool calls sometimes leak as plain text."""

    MARKER = "<<END_CALL>>"

    def __init__(self, hangup: "HangupOnBotDone"):
        super().__init__()
        self._hangup = hangup
        self._buf = ""
        self._seen = False
        self._emitted = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buf = ""
            self._seen = False
            self._emitted = 0
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            if self._seen:
                return  # drop everything after the marker
            self._buf += frame.text
            idx = self._buf.find(self.MARKER)
            if idx != -1:
                self._seen = True
                head = self._buf[:idx]
                self._buf = ""
                if head:
                    self._emitted += len(head)
                    await self.push_frame(LLMTextFrame(_tts_safe(head)), direction)
                return
            # Hold back a possible partial marker at the tail; emit the rest.
            keep = len(self.MARKER) - 1
            if len(self._buf) > keep:
                emit, self._buf = self._buf[:-keep], self._buf[-keep:]
                self._emitted += len(emit)
                await self.push_frame(LLMTextFrame(_tts_safe(emit)), direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._buf and not self._seen:
                self._emitted += len(self._buf)
                await self.push_frame(LLMTextFrame(_tts_safe(self._buf)), direction)
            self._buf = ""
            if self._seen:
                # Ensure there's audio (so a BotStoppedSpeaking fires the hangup).
                if self._emitted == 0:
                    await self.push_frame(TTSSpeakFrame(_tts_safe(DEFAULT_FAREWELL)), direction)
                self._hangup.arm()
                logger.info("end-call marker seen -> will hang up after farewell")
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)


async def run_bot(
    transport,
    *,
    handle_sigint: bool,
    greet_on_event: bool,
    sample_rate: int,
) -> None:
    """Build and run the pipeline against the given transport."""
    # End-of-call is handled by the <<END_CALL>> text sentinel (EndCallSentinel),
    # not a tool — sarvam's tool calls are unreliable and sometimes leak as speech.
    lead = load_lead()
    sarvam_lang = _sarvam_lang_code(lead.get("language"))
    logger.info(
        f"Lead: {lead['lead_name']} ({lead['student_class']}, {lead.get('language')}/{sarvam_lang}) -> "
        f"{lead['batch_name']} @ {lead['batch_start_date']}"
    )
    stt = create_stt(sample_rate, language=sarvam_lang)
    llm = create_llm()
    tts = create_tts(sample_rate, language=sarvam_lang)
    greeting = build_greeting(lead)
    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt(lead)}],
    )

    idle_timeout = float(os.getenv("IDLE_TIMEOUT_SECS", "7"))
    smart_turn_stop = float(os.getenv("SMART_TURN_STOP_SECS", "1.5"))
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # VAD stop_secs 0.15 (vs 0.2 default); Smart Turn stop_secs 1.5 (vs 3.0)
            # — both trimmed for snappier turns. Re-supply default start strategies.
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=0.15, start_secs=0.2, confidence=0.7)
            ),
            user_turn_strategies=UserTurnStrategies(
                start=default_user_turn_start_strategies(),
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(
                            params=SmartTurnParams(stop_secs=smart_turn_stop)
                        ),
                    )
                ],
            ),
            # Fires on_user_turn_idle after this many seconds of caller silence.
            user_idle_timeout=idle_timeout,
        ),
    )
    user_agg = aggregators.user()
    assistant_agg = aggregators.assistant()

    hangup = HangupOnBotDone()
    endcall = EndCallSentinel(hangup)  # strips <<END_CALL>> before TTS + arms hangup

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_agg,
            llm,
            endcall,  # between LLM and TTS: filter the end-call marker out of speech
            tts,
            transport.output(),
            hangup,  # must sit AFTER output to catch the downstream BotStoppedSpeaking
            assistant_agg,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=sample_rate,
            audio_out_sample_rate=sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    hangup.set_task(task)

    idle_state = {"nudged": False, "ending": False}

    @user_agg.event_handler("on_user_turn_idle")
    async def _on_idle(_agg, *_args):
        if idle_state["ending"]:
            return
        if not idle_state["nudged"]:
            idle_state["nudged"] = True
            logger.info("Caller idle -> nudge")
            await task.queue_frame(TTSSpeakFrame(_tts_safe(IDLE_NUDGE)))
        else:
            idle_state["ending"] = True
            logger.info("Caller still idle -> ending call")
            hangup.arm()
            await task.queue_frame(TTSSpeakFrame(_tts_safe(IDLE_FAREWELL)))

    @user_agg.event_handler("on_user_turn_stopped")
    async def _reset_idle(_agg, *_args):
        idle_state["nudged"] = False   # caller spoke; refresh the nudge budget

    async def _greet():
        # Fixed opener: speak it directly (no LLM run) so the model + end_call tool
        # isn't invoked on an empty conversation. Do NOT also add it to context here —
        # the assistant aggregator captures the spoken line, and the prompt forbids
        # re-greeting. (Adding it manually caused duplicate greetings in context.)
        await task.queue_frames([TTSSpeakFrame(_tts_safe(greeting))])

    if greet_on_event:

        @transport.event_handler("on_client_connected")
        async def _on_connected(_transport, _client):
            logger.info("Client connected -> fixed greeting")
            await _greet()

        @transport.event_handler("on_client_disconnected")
        async def _on_disconnected(_transport, _client):
            logger.info("Client disconnected -> cancelling pipeline")
            await task.cancel()

    else:
        # Local audio transport emits no connect event; greet now.
        await _greet()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


async def _run_local() -> None:
    """Test A entrypoint: talk to the bot through your laptop mic + speakers."""
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )

    sample_rate = int(os.getenv("LOCAL_SAMPLE_RATE", "16000"))
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )
    logger.info(
        f"LOCAL mic test @ {sample_rate} Hz. Wait for the greeting, then talk. Ctrl+C to stop."
    )
    await run_bot(
        transport,
        handle_sigint=True,
        greet_on_event=False,
        sample_rate=sample_rate,
    )


if __name__ == "__main__":
    asyncio.run(_run_local())
