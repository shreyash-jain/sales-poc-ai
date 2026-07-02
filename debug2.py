import os, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Volumes/shreyash_ex/poc_sales_call/.env")
client = OpenAI(base_url="https://api.sarvam.ai/v1", api_key=os.environ["SARVAM_API_KEY"])

msgs = [
    {"role":"system","content":"You are Vidya, a warm first-touch caller for Vacademy. Already greeted; never re-greet. One short reply, one question."},
    {"role":"assistant","content":"Hi! This is Vidya from Vacademy — what brings you in today?"},
    {"role":"user","content":"Hello"},
]

def has_reasoning(resp):
    m = resp.choices[0].message
    rc = getattr(m, "reasoning_content", None)
    return bool(rc), (m.content or "")[:120]

print("A) extra_body={'extra_body':{'reasoning_effort':None}} (spec literal):")
try:
    r = client.chat.completions.create(model="sarvam-30b", messages=msgs, temperature=0.6, max_tokens=150, extra_body={"extra_body":{"reasoning_effort":None}})
    print("  reasoning?", has_reasoning(r))
except Exception as e:
    print("  ERR", type(e).__name__, e)

print("B) extra_body={'reasoning_effort':None} (top-level):")
try:
    r = client.chat.completions.create(model="sarvam-30b", messages=msgs, temperature=0.6, max_tokens=150, extra_body={"reasoning_effort":None})
    print("  reasoning?", has_reasoning(r))
except Exception as e:
    print("  ERR", type(e).__name__, e)

print("C) reasoning_effort='low' top-level:")
try:
    r = client.chat.completions.create(model="sarvam-30b", messages=msgs, temperature=0.6, max_tokens=150, extra_body={"reasoning_effort":"low"})
    print("  reasoning?", has_reasoning(r))
except Exception as e:
    print("  ERR", type(e).__name__, e)

print("D) extra_body={'reasoning_effort':'none'} string:")
try:
    r = client.chat.completions.create(model="sarvam-30b", messages=msgs, temperature=0.6, max_tokens=150, extra_body={"reasoning_effort":"none"})
    print("  reasoning?", has_reasoning(r))
except Exception as e:
    print("  ERR", type(e).__name__, e)
