import os
from pathlib import Path
from dotenv import load_dotenv


def load_secrets(env_path: str | Path = ".env") -> dict:
    """Load .env (silently no-op if missing) and return relevant vars."""
    load_dotenv(env_path, override=False)
    return {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "HF_USERNAME": os.environ.get("HF_USERNAME", "twanghcmut"),
        "HF_PRIVATE": os.environ.get("HF_PRIVATE", "0") == "1",
    }


def login_hf_and_wandb(secrets: dict) -> None:
    if secrets.get("HF_TOKEN"):
        from huggingface_hub import login as hf_login
        hf_login(token=secrets["HF_TOKEN"], add_to_git_credential=False)
    if secrets.get("WANDB_API_KEY"):
        import wandb
        wandb.login(key=secrets["WANDB_API_KEY"])
