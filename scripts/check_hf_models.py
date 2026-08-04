"""
Verifies the HF_API_TOKEN and checks each configured model responds on
the HF Inference API. Run manually before deployment:

    python scripts/check_hf_models.py

Exits non-zero if any required model is unavailable, so it can also be
wired into CI/CD as a pre-deploy gate.
"""
import os
import sys
import time

import requests

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_API_BASE = "https://api-inference.huggingface.co/models"

MODELS = {
    "PRIMARY (JD/question JSON)": os.environ.get(
        "HF_CHAT_MODEL_PRIMARY", "Qwen/Qwen2.5-Coder-32B-Instruct"
    ),
    "INTERVIEW (conversation)": os.environ.get(
        "HF_CHAT_MODEL_INTERVIEW", "meta-llama/Llama-3.1-8B-Instruct"
    ),
    "SCORING (answer evaluation)": os.environ.get(
        "HF_CHAT_MODEL_SCORING", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    ),
}

TIMEOUT_SECONDS = 30
COLD_START_RETRY_WAIT = 10


def check_token() -> bool:
    if not HF_API_TOKEN:
        print("FAIL: HF_API_TOKEN is not set in the environment.")
        return False
    print("OK: HF_API_TOKEN is set.")
    return True


def check_model(label: str, model_id: str) -> bool:
    url = f"{HF_API_BASE}/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": "ping", "parameters": {"max_new_tokens": 1}}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as e:
        print(f"FAIL [{label}] {model_id}: network error — {e}")
        return False

    if resp.status_code == 200:
        print(f"OK [{label}] {model_id}: available.")
        return True

    if resp.status_code == 503:
        # Model is cold-starting on HF infra — retry once.
        print(f"WARN [{label}] {model_id}: cold-starting, retrying in {COLD_START_RETRY_WAIT}s...")
        time.sleep(COLD_START_RETRY_WAIT)
        try:
            resp2 = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as e:
            print(f"FAIL [{label}] {model_id}: network error on retry — {e}")
            return False
        if resp2.status_code == 200:
            print(f"OK [{label}] {model_id}: available after cold start.")
            return True
        print(f"FAIL [{label}] {model_id}: still unavailable after retry — HTTP {resp2.status_code}")
        return False

    if resp.status_code == 404:
        print(f"FAIL [{label}] {model_id}: not found for this provider (HTTP 404). Check model id.")
        return False

    if resp.status_code == 429:
        print(f"FAIL [{label}] {model_id}: rate-limited (HTTP 429). Check token tier/quota.")
        return False

    print(f"FAIL [{label}] {model_id}: unexpected HTTP {resp.status_code} — {resp.text[:200]}")
    return False


def main() -> int:
    all_ok = check_token()
    if not all_ok:
        return 1

    for label, model_id in MODELS.items():
        ok = check_model(label, model_id)
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll models available.")
        return 0

    print("\nOne or more models failed the check — do not deploy until resolved.")
    return 1


if __name__ == "__main__":
    sys.exit(main())