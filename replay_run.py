import os
import json
import time
import statistics
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Volumes/shreyash_ex/poc_sales_call/.env")

client = OpenAI(base_url="https://api.sarvam.ai/v1", api_key=os.environ["SARVAM_API_KEY"])

MODEL = "sarvam-30b"

SYSTEM_PROMPT = '''You are Vidya, a warm, genuinely friendly person making a first-touch call for Vacademy, an Indian education brand. You sound relaxed, upbeat, human.

You have ALREADY greeted the caller out loud. NEVER greet, say hi, or re-introduce yourself again — just respond naturally to what they say.

Goal: find out what they need (which exam/course, the student's class/level, their goal), gauge interest, offer a counsellor follow-up.

HOW TO TALK — LIVE phone call:
- 1-2 short sentences per reply, max. EXACTLY ONE question per turn — never two.
- Acknowledge what they said (reflect a key word), then ask your one question. Vary short affirmations; never reuse one twice in a row.
- NEVER repeat a question you've already asked, and never repeat an earlier reply. If they didn't really answer, do NOT re-ask the same way — acknowledge and rephrase lightly or move on.
- React to feeling, not just facts.
- If the caller seems uninterested, distracted, or gives non-answers ("okay", "nothing", "not really", "you tell me"), DON'T push or interrogate. Warmly say in one line why you're calling and ask if it's a good time; if they're clearly not interested, thank them and end the call.
- Tie each question to what they said; natural connectors ("so", "okay so", "can I ask"). No corporate phrasing. Don't sound scripted.
- Never invent prices, dates, guarantees. Punctuate expressively.

CLOSING: when done or they're wrapping up, one-line recap + handoff offer.

ENDING THE CALL: when the conversation is clearly over — caller says bye, has nothing more, or isn't interested — call the end_call function with a short warm farewell as the 'farewell' argument. NEVER call end_call before the caller has spoken. Don't type the farewell; only pass it to end_call.'''

GREETING = "Hi! This is Vidya from Vacademy — thanks so much for picking up! So, what brings you in today?"

tools = [{
    "type": "function",
    "function": {
        "name": "end_call",
        "description": "End the call when the conversation is over or the caller isn't interested.",
        "parameters": {
            "type": "object",
            "properties": {"farewell": {"type": "string"}},
            "required": ["farewell"],
        },
    },
}]

caller_turns = ["Hello", "Okay", "why are you repeating yourself?", "Nothing, you called, you tell me.", "No, not really interested in anything.", "Not really"]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "assistant", "content": GREETING},
]

pairs = []
ttfbs = []
end_call_turn = 0
notes = ""

try:
    for idx, turn in enumerate(caller_turns, start=1):
        messages.append({"role": "user", "content": turn})

        t0 = time.perf_counter()
        first_token_t = None
        text_parts = []
        tool_name = None
        tool_args_parts = []
        tool_call_id = None

        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            temperature=0.6,
            max_tokens=150,
            stream=True,
            # Spec says "merge extra_body appropriately; for sarvam it disables reasoning".
            # Putting reasoning_effort at the top level of the request body is what actually
            # disables reasoning on Sarvam (the double-nested {"extra_body":{...}} form is
            # ignored by the API and leaves reasoning ON, eating max_tokens -> empty content).
            extra_body={"reasoning_effort": None},
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            got = False
            if delta is None:
                continue
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
                got = True
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    if tc.id:
                        tool_call_id = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_name = tc.function.name
                        if tc.function.arguments:
                            tool_args_parts.append(tc.function.arguments)
                got = True
            if got and first_token_t is None:
                first_token_t = time.perf_counter()

        if first_token_t is None:
            first_token_t = time.perf_counter()
        ttfb_ms = (first_token_t - t0) * 1000.0
        ttfbs.append(ttfb_ms)

        assistant_text = "".join(text_parts)

        if tool_name == "end_call":
            raw = "".join(tool_args_parts)
            farewell = ""
            try:
                farewell = json.loads(raw).get("farewell", "")
            except Exception:
                farewell = raw
            recorded = f"[end_call: {farewell}]"
            pairs.append({"user": turn, "assistant": recorded})
            end_call_turn = idx
            # append assistant message with tool_calls for completeness then stop
            messages.append({
                "role": "assistant",
                "content": assistant_text or None,
                "tool_calls": [{
                    "id": tool_call_id or "call_0",
                    "type": "function",
                    "function": {"name": "end_call", "arguments": raw},
                }],
            })
            break
        else:
            pairs.append({"user": turn, "assistant": assistant_text})
            messages.append({"role": "assistant", "content": assistant_text})

except Exception as e:
    notes = f"Exception during replay: {type(e).__name__}: {e}"

# verbatim_repeats: count assistant replies (near-)identical to an earlier one
def norm(s):
    return " ".join(s.lower().split())

seen = []
verbatim_repeats = 0
for p in pairs:
    n = norm(p["assistant"])
    is_repeat = False
    for prev in seen:
        if n == prev:
            is_repeat = True
            break
        # near-identical: high overlap
        if n and prev:
            a, b = set(n.split()), set(prev.split())
            if a and b:
                jac = len(a & b) / len(a | b)
                if jac >= 0.85:
                    is_repeat = True
                    break
    if is_repeat:
        verbatim_repeats += 1
    seen.append(n)

# re-greeted detection
greet_markers = ["hi ", "hello", "hey", "this is vidya", "from vacademy", "i'm vidya", "i am vidya", "my name is"]
re_greeted = False
for p in pairs:
    a = p["assistant"].lower()
    if a.startswith("[end_call"):
        continue
    for m in greet_markers:
        if m in a:
            re_greeted = True
            break
    if re_greeted:
        break

median_ttfb = statistics.median(ttfbs) if ttfbs else 0.0

result = {
    "label": "sarvam-30b replay",
    "model": MODEL,
    "turns": pairs,
    "ttfb_ms_median": round(median_ttfb, 1),
    "end_call_fired_at_turn": end_call_turn,
    "verbatim_repeats": verbatim_repeats,
    "re_greeted": re_greeted,
    "ttfbs_all": [round(x, 1) for x in ttfbs],
    "notes": notes,
}

print("===RESULT_JSON_START===")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("===RESULT_JSON_END===")
