import os, time, json, statistics
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Volumes/shreyash_ex/poc_sales_call/.env")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

MODEL = "google/gemini-3.1-flash-lite"

PROMPT = '''You are Vidya, a warm, genuinely friendly person making a first-touch call for Vacademy, an Indian education brand. You sound relaxed, upbeat, human.

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

tools = [{"type": "function", "function": {
    "name": "end_call",
    "description": "End the call when the conversation is over or the caller isn't interested.",
    "parameters": {"type": "object", "properties": {"farewell": {"type": "string"}}, "required": ["farewell"]}
}}]

messages = [
    {"role": "system", "content": PROMPT},
    {"role": "assistant", "content": "Hi! This is Vidya from Vacademy — thanks so much for picking up! So, what brings you in today?"},
]

caller_turns = ["Hello", "Okay", "why are you repeating yourself?", "Nothing, you called, you tell me.", "No, not really interested in anything.", "Not really"]

results = []
ttfbs = []
end_call_turn = 0
notes_extra = ""

for idx, turn in enumerate(caller_turns, start=1):
    messages.append({"role": "user", "content": turn})
    start = time.time()
    ttfb = None
    text_parts = []
    tool_name = None
    tool_args_str = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools,
            temperature=0.5, max_tokens=150, stream=True, extra_body={},
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            got_token = False
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
                got_token = True
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    if tc.function:
                        if tc.function.name:
                            tool_name = tc.function.name
                        if tc.function.arguments:
                            tool_args_str += tc.function.arguments
                            got_token = True
            if got_token and ttfb is None:
                ttfb = (time.time() - start) * 1000.0
    except Exception as e:
        notes_extra += f" ERROR turn {idx}: {type(e).__name__}: {e}."
        results.append((turn, f"[ERROR: {e}]"))
        break

    if ttfb is None:
        ttfb = (time.time() - start) * 1000.0
    ttfbs.append(ttfb)

    text = "".join(text_parts).strip()

    if tool_name == "end_call":
        farewell = ""
        try:
            farewell = json.loads(tool_args_str).get("farewell", "") if tool_args_str else ""
        except Exception:
            farewell = tool_args_str
        assistant_record = f"[end_call: {farewell}]"
        results.append((turn, assistant_record))
        # append assistant message with tool_calls
        messages.append({
            "role": "assistant", "content": text or None,
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "end_call", "arguments": tool_args_str or "{}"}}],
        })
        end_call_turn = idx
        break
    else:
        results.append((turn, text))
        messages.append({"role": "assistant", "content": text})

# verbatim repeats: count assistant replies (near-)identical to an earlier one
def norm(s):
    return " ".join(s.lower().split())

seen = []
verbatim_repeats = 0
for _, a in results:
    na = norm(a)
    is_rep = False
    for prev in seen:
        if na == prev:
            is_rep = True
            break
        # near-identical: high overlap
        if na and prev and (na in prev or prev in na):
            is_rep = True
            break
    if is_rep:
        verbatim_repeats += 1
    seen.append(na)

# re-greet detection
greet_markers = ["hi ", "hello", "hey ", "vidya", "vacademy", "this is", "my name"]
re_greeted = False
for _, a in results:
    la = a.lower()
    if any(m in la for m in greet_markers):
        re_greeted = True
        break

median_ttfb = round(statistics.median(ttfbs), 2) if ttfbs else 0

out = {
    "label": "Vacademy first-touch replay",
    "model": MODEL,
    "turns": [{"user": u, "assistant": a} for (u, a) in results],
    "ttfb_ms_median": median_ttfb,
    "end_call_fired_at_turn": end_call_turn,
    "verbatim_repeats": verbatim_repeats,
    "re_greeted": re_greeted,
    "ttfbs": [round(x, 2) for x in ttfbs],
    "notes_extra": notes_extra,
}
print(json.dumps(out, indent=2, ensure_ascii=False))
