"""Place an outbound test call via the Plivo REST API (Test B).

Rings TEST_TO_NUMBER from PLIVO_FROM_NUMBER and points Plivo at our /answer
endpoint, which returns the <Stream> XML that bridges call audio to the bot.

Prereqs (see RUN.md):
  - `uvicorn server:app --port 8000` is running
  - ngrok is exposing it and NGROK_HOST is set in .env
  - TEST_TO_NUMBER is verified in the Plivo console (trial restriction)

Usage: python make_call.py
"""

import os
import sys

import plivo
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    auth_id = os.getenv("PLIVO_AUTH_ID")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN")
    from_number = os.getenv("PLIVO_FROM_NUMBER")
    to_number = os.getenv("TEST_TO_NUMBER")
    ngrok_host = os.getenv("NGROK_HOST", "").strip().rstrip("/")

    missing = [
        name
        for name, val in {
            "PLIVO_AUTH_ID": auth_id,
            "PLIVO_AUTH_TOKEN": auth_token,
            "PLIVO_FROM_NUMBER": from_number,
            "TEST_TO_NUMBER": to_number,
            "NGROK_HOST": ngrok_host,
        }.items()
        if not val
    ]
    if missing:
        sys.exit(f"Missing required env vars in .env: {', '.join(missing)}")

    if ngrok_host.startswith("http"):
        sys.exit("NGROK_HOST must be the host only (no https://). e.g. abc123.ngrok-free.app")

    answer_url = f"https://{ngrok_host}/answer"

    client = plivo.RestClient(auth_id, auth_token)
    print(f"Calling {to_number} from {from_number}")
    print(f"answer_url = {answer_url}")

    response = client.calls.create(
        from_=from_number,
        to_=to_number,
        answer_url=answer_url,
        answer_method="POST",
    )
    # response carries the call request_uuid / api_id
    print("Plivo accepted the call request:")
    print(response)


if __name__ == "__main__":
    main()
