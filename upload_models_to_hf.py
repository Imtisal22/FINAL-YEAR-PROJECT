"""
Upload model files to Hugging Face Hub.
Run AFTER:  ~/imtisal_tf_venv/Scripts/python -m huggingface_cli login

Usage:  ~/imtisal_tf_venv/Scripts/python upload_models_to_hf.py
"""

import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "Alim-7"
REPO_NAME   = "imtisal-maize-models"
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")

MODEL_FILES = [
    "arima_model.pkl",
    "rf_model.pkl",
    "svm_model.pkl",
    "lstm_model.keras",
    "lstm_scaler.pkl",
]

api = HfApi()

# Create repo (private=False so Render can download without a token)
print(f"[hf] Creating repo: {REPO_ID} ...")
create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)
print(f"[hf] Repo ready: https://huggingface.co/{REPO_ID}")

# Upload each file
for fname in MODEL_FILES:
    local_path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(local_path):
        print(f"[skip] {fname} not found locally")
        continue
    size_kb = os.path.getsize(local_path) // 1024
    print(f"[upload] {fname}  ({size_kb} KB) ...", end=" ", flush=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=fname,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print("done")

print(f"\n[hf] All files uploaded.")
print(f"[hf] Repo URL: https://huggingface.co/{REPO_ID}")
print(f"[hf] Files can now be downloaded at startup on Render.")
