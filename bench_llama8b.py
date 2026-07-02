import json
import time
import statistics
import sys
from dotenv import load_dotenv
import os

load_dotenv("/Volumes/shreyash_ex/poc_sales_call/.env")

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=20.0,
)

LABEL = "llama-3.1-8b @ Cerebras/Groq"
MODEL = "meta-llama/llama-3.1-8b-instruct"
PROVIDER_ROUTING = {
    "provider": {
        "order": ["Cerebras", "Groq"],
        "allow_fallbacks": True,
        "sort": "latency",
    }
}

SYSTEM = "You are a terse phone assistant. Reply in ONE short sentence."
USER = "Hi, I'm interested in IIT JEE coaching for class 11."

N_SAMPLES = 5

result = {
    "label": LABEL,
    "model": MODEL,
    "provider_pref": 'order=[Cerebras,Groq], allow_fallbacks=true, sort=latency',
    "available": False,
    "ttfb_ms_median": 0.0,
    "ttfb_ms_min": 0.0,
    "ttfb_ms_max": 0.0,
    "total_ms_median": 0.0,
    "reply_sample": "",
    "error": "",
    "notes": "",
    "served_provider": "",
}


def one_call():
    """Returns (ttfb_ms, total_ms, full_text, served_provider) or raises."""
    start = time.perf_counter()
    ttfb = None
    chunks = []
    served = None
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        max_tokens=60,
        stream=True,
        extra_body=PROVIDER_ROUTING,
        extra_query={"include_usage": "true"},
    )
    for chunk in stream:
        # capture provider if present on chunk
        prov = getattr(chunk, "provider", None)
        if prov:
            served = prov
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                if ttfb is None:
                    ttfb = time.perf_counter() - start
                chunks.append(content)
    total = time.perf_counter() - start
    if ttfb is None:
        ttfb = total
    return ttfb * 1000.0, total * 1000.0, "".join(chunks), served


# Non-streaming probe to capture provider reliably
def probe_provider():
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
            max_tokens=60,
            extra_body=PROVIDER_ROUTING,
        )
        return getattr(resp, "provider", None)
    except Exception as e:
        return None


ttfbs = []
totals = []
last_text = ""
served_provider = None

try:
    for i in range(N_SAMPLES):
        try:
            ttfb_ms, total_ms, text, served = one_call()
        except Exception as e:
            # First failure -> record and stop
            result["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            print(json.dumps(result), file=sys.stderr)
            print("CALL_FAILED", i, repr(e), file=sys.stderr)
            raise
        if served and not served_provider:
            served_provider = served
        if i == 0:
            # warm-up, discard
            last_text = text
            continue
        ttfbs.append(ttfb_ms)
        totals.append(total_ms)
        last_text = text or last_text

    # try a non-streaming probe for provider if not captured
    if not served_provider:
        served_provider = probe_provider()

    result["available"] = True
    result["ttfb_ms_median"] = round(statistics.median(ttfbs), 1)
    result["ttfb_ms_min"] = round(min(ttfbs), 1)
    result["ttfb_ms_max"] = round(max(ttfbs), 1)
    result["total_ms_median"] = round(statistics.median(totals), 1)
    result["reply_sample"] = last_text.strip()
    result["served_provider"] = served_provider or "unknown"
except Exception as e:
    result["available"] = False
    if not result["error"]:
        result["error"] = f"{type(e).__name__}: {str(e)[:500]}"

print("RESULT_JSON_START")
print(json.dumps(result, indent=2))
print("RESULT_JSON_END")
