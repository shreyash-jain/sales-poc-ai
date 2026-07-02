import json
import time
import statistics
from dotenv import load_dotenv
import os

load_dotenv("/Volumes/shreyash_ex/poc_sales_call/.env")

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

LABEL = "gemini-flash-lite + sort=latency"
MODEL = "google/gemini-3.1-flash-lite"
EXTRA_BODY = {"provider": {"sort": "latency"}}

SYSTEM = "You are a terse phone assistant. Reply in ONE short sentence."
USER = "Hi, I'm interested in IIT JEE coaching for class 11."

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
last_reply = ""
served_provider = None
err = ""

N = 5
for i in range(N):
    try:
        start = time.perf_counter()
        first_token_t = None
        chunks = []
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
            max_tokens=60,
            stream=True,
            timeout=20,
            extra_body=EXTRA_BODY,
        )
        for chunk in stream:
            # capture provider if present
            prov = getattr(chunk, "provider", None)
            if prov:
                served_provider = prov
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                    chunks.append(content)
        end = time.perf_counter()
        if first_token_t is None:
            # no content tokens streamed
            raise RuntimeError("no content tokens streamed")
        ttfb_ms = (first_token_t - start) * 1000.0
        total_ms = (end - start) * 1000.0
        reply = "".join(chunks)
        print(f"sample {i}: ttfb={ttfb_ms:.0f}ms total={total_ms:.0f}ms provider={served_provider} reply={reply!r}")
        if i == 0:
            # warm-up, discard
            last_reply = reply
            continue
        ttfbs.append(ttfb_ms)
        totals.append(total_ms)
        last_reply = reply
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"sample {i}: ERROR {err}")
        # if first call errors hard (404/routing), bail
        if i == 0:
            break

if ttfbs:
    result["available"] = True
    result["ttfb_ms_median"] = round(statistics.median(ttfbs), 1)
    result["ttfb_ms_min"] = round(min(ttfbs), 1)
    result["ttfb_ms_max"] = round(max(ttfbs), 1)
    result["total_ms_median"] = round(statistics.median(totals), 1)
    result["reply_sample"] = last_reply
    result["provider_served"] = served_provider
    result["error"] = err  # may hold a later transient error
else:
    result["available"] = False
    result["reply_sample"] = last_reply
    result["error"] = err or "no successful samples"
    result["provider_served"] = served_provider

print("RESULT_JSON_START")
print(json.dumps(result))
print("RESULT_JSON_END")
