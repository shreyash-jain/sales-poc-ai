import os, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Volumes/shreyash_ex/poc_sales_call/.env")
client = OpenAI(base_url="https://api.sarvam.ai/v1", api_key=os.environ["SARVAM_API_KEY"])

SYSTEM_PROMPT = "You are Vidya, a warm friendly first-touch caller for Vacademy. You ALREADY greeted. Never re-greet. One short reply, exactly one question."
GREETING = "Hi! This is Vidya from Vacademy — thanks so much for picking up! So, what brings you in today?"

tools = [{"type":"function","function":{"name":"end_call","description":"End the call.","parameters":{"type":"object","properties":{"farewell":{"type":"string"}},"required":["farewell"]}}}]

messages = [
    {"role":"system","content":SYSTEM_PROMPT},
    {"role":"assistant","content":GREETING},
    {"role":"user","content":"Hello"},
]

# Non-streaming to inspect full message object
resp = client.chat.completions.create(model="sarvam-30b", messages=messages, tools=tools, temperature=0.6, max_tokens=150, extra_body={"extra_body":{"reasoning_effort":None}})
print("NON-STREAM full choice message:")
print(resp.choices[0].message.model_dump_json(indent=2))
print("finish_reason:", resp.choices[0].finish_reason)
print()

print("STREAM raw deltas:")
stream = client.chat.completions.create(model="sarvam-30b", messages=messages, tools=tools, temperature=0.6, max_tokens=150, stream=True, extra_body={"extra_body":{"reasoning_effort":None}})
for i, chunk in enumerate(stream):
    if chunk.choices:
        print(i, chunk.choices[0].delta.model_dump_json())
