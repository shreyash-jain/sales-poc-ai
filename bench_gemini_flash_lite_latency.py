import os
import time
import json
import statistics
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Volumes/shreyash_ex/poc_sales_call/.env")

LABEL = "gemini-2.5-flash-lite + sort=latency"
MODEL = "google/gemini-2.5-flash-lite"
EXTRA_BODY = {"provider": {"sort": "latency"}}

SYSTEM = "You are a terse phone assistant. Reply in ONE short sentence."
USER = "Hi, I'm interested in IIT JEE coaching for class 11."
MAX_TOKENS = 60
N_SAMPLES = 5
TIMEOUT = 20.0

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=TIMEOUT,
)

result = {
    "label": LABEL,
    "model": MODEL,
    "provider_pref": "sort=latency",
    "available": False,
    "ttfb_ms_median": None,
    "ttfb_ms_min": None,
    "ttfb_ms_max": None,
    "total_ms_median": None,
    "reply_sample": "",
    "error": "",
    "notes": "",
    "provider_served": None,
}

ttfbs = []
totals = []
replies = []
provider_served = None
last_error = ""

for i in range(N_SAMPLES):
    try:
        start = time.perf_counter()
        first_token_t = None
        chunks_text = []
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
            max_tokens=MAX_TOKENS,
            stream=True,
            extra_body=EXTRA_BODY,
        )
        for chunk in stream:
            # capture provider if exposed
            prov = getattr(chunk, "provider", None)
            if prov and provider_served is None:
                provider_served = prov
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                    chunks_text.append(content)
        end = time.perf_counter()
        if first_token_t is None:
            raise RuntimeError("no content tokens received")
        ttfb_ms = (first_token_t - start) * 1000.0
        total_ms = (end - start) * 1000.0
        reply = "".join(chunks_text).strip()
        print(f"sample {i}: ttfb={ttfb_ms:.0f}ms total={total_ms:.0f}ms provider={provider_served} reply={reply!r}")
        if i == 0:
            # warm-up, discard
            continue
        ttfbs.append(ttfb_ms)
        totals.append(total_ms)
        replies.append(reply)
    except Exception as e:
        last_error = f"{type(e).__name__}: {e}"
        print(f"sample {i}: ERROR {last_error}")

if ttfbs:
    result["available"] = True
    result["ttfb_ms_median"] = round(statistics.median(ttfbs), 1)
    result["ttfb_ms_min"] = round(min(ttfbs), 1)
    result["ttfb_ms_max"] = round(max(ttfbs), 1)
    result["total_ms_median"] = round(statistics.median(totals), 1)
    result["reply_sample"] = replies[-1] if replies else ""
    result["provider_served"] = provider_served
    result["error"] = ""
else:
    result["available"] = False
    result["error"] = last_error or "all samples failed"

print("RESULT_JSON_START")
print(json.dumps(result))
print("RESULT_JSON_END")
