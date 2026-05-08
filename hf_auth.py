"""
hf_auth.py — HuggingFace authentication + model download.
Reads HF_TOKEN from .env in the same directory.
"""

import os
import sys
from pathlib import Path


def load_env(env_path=None):
    """Parse .env and inject into os.environ (no third-party deps needed)."""
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    if not Path(env_path).exists():
        print(f"[WARN] .env not found at {env_path}. Create one: HF_TOKEN=hf_your_token")
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"\'\' '))


def login_huggingface():
    """Login to HF Hub using HF_TOKEN from .env / environment."""
    load_env()
    token = os.environ.get("HF_TOKEN", "")
    if not token or token == "hf_your_token_here":
        print("[ERROR] HF_TOKEN not set. Edit .env: HF_TOKEN=hf_your_real_token")
        sys.exit(1)
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("[INFO] HuggingFace login successful.")
    except Exception as e:
        print(f"[ERROR] HF login failed: {e}")
        sys.exit(1)


def ensure_model(model_id: str):
    """Download model to HF cache on first run; instant on subsequent runs."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
    print(f"[INFO] Verifying model: {model_id}")
    try:
        snapshot_download(repo_id=model_id, repo_type="model",
                          ignore_patterns=["*.pt", "original/*"])
        print(f"[INFO] Model ready: {model_id}")
    except GatedRepoError:
        print(f"[ERROR] Gated model. Accept license at https://huggingface.co/{model_id}")
        sys.exit(1)
    except RepositoryNotFoundError:
        print(f"[ERROR] Model not found: {model_id}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)
