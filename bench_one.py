import os, time, json, statistics, sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Volumes/shreyash_ex/poc_sales_call/.env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

LABEL = "llama-3.3-70b @ Cerebras/Groq"
MODEL = "meta-llama/llama-3.3-70b-instruct"
PROVIDER = {"order": ["Cerebras", "Groq"], "allow_fallbacks": True, "sort": "latency"}

SYSTEM = "You are a terse phone assistant. Reply in ONE short sentence."
USER = "Hi, I'm interested in IIT JEE coaching for class 11."

result = {
    "label": LABEL,
    "model": MODEL,
    "provider_pref": "order=[Cerebras,Groq], allow_fallbacks=true, sort=latency",
    "available": False,
    "ttfb_ms_median": None,
    "ttfb_ms_min": None,
    "ttfb_ms_max": None,
    "total_ms_median": None,
    "reply_sample": "",
    "error": "",
    "notes": "",
    "serving_provider": None,
}

def one_call():
    start = time.time()
    ttfb = None
    text_parts = []
    served = None
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": USER}],
        max_tokens=60,
        stream=True,
        timeout=20,
        extra_body={"provider": PROVIDER, "usage": {"include": True}},
    )
    for chunk in stream:
        # capture provider if present
        p = getattr(chunk, "provider", None)
        if p:
            served = p
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                if ttfb is None:
                    ttfb = time.time() - start
                text_parts.append(content)
    total = time.time() - start
    return ttfb, total, "".join(text_parts), served

ttfbs = []
totals = []
last_reply = ""
served_provider = None
errs = []

for i in range(5):
    try:
        ttfb, total, text, served = one_call()
        if served:
            served_provider = served
        if i == 0:
            # warm-up, discard
            if text:
                last_reply = text
            continue
        if ttfb is None:
            # no content streamed
            errs.append(f"sample{i}: no content streamed")
            continue
        ttfbs.append(ttfb * 1000.0)
        totals.append(total * 1000.0)
        last_reply = text
    except Exception as e:
        errs.append(f"sample{i}: {type(e).__name__}: {e}")

# non-streaming probe for provider if not captured
if served_provider is None:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": USER}],
            max_tokens=60,
            timeout=20,
            extra_body={"provider": PROVIDER},
        )
        served_provider = getattr(resp, "provider", None)
    except Exception as e:
        errs.append(f"probe: {type(e).__name__}: {e}")

if ttfbs:
    result["available"] = True
    result["ttfb_ms_median"] = round(statistics.median(ttfbs), 1)
    result["ttfb_ms_min"] = round(min(ttfbs), 1)
    result["ttfb_ms_max"] = round(max(ttfbs), 1)
    result["total_ms_median"] = round(statistics.median(totals), 1)
    result["reply_sample"] = last_reply.strip()
else:
    result["available"] = False

result["serving_provider"] = served_provider
if errs:
    result["error"] = " | ".join(errs)

print(json.dumps(result, indent=2))
