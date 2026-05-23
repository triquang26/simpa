import os
from pathlib import Path
from dotenv import load_dotenv


def load_secrets(env_path: str | Path = ".env") -> dict:
    """Load .env (silently no-op if missing) and return relevant vars."""
    load_dotenv(env_path, override=False)
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        try:
            from huggingface_hub.utils import get_token
            hf_token = get_token() or ""
        except Exception:
            pass
    return {
        "HF_TOKEN": hf_token,
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


def preflight_check(secrets: dict) -> None:
    """Verify HF and W&B credentials before training. Raises RuntimeError on failure."""
    from huggingface_hub import HfApi
    token = secrets.get("HF_TOKEN") or None
    try:
        user = HfApi().whoami(token=token)
        print(f"[preflight] HF OK — logged in as: {user['name']}")
    except Exception as e:
        raise RuntimeError(
            f"[preflight] HF auth failed: {e!r}\n"
            "Fix: set HF_TOKEN in .env or run `huggingface-cli login`"
        ) from e

    wandb_key = secrets.get("WANDB_API_KEY")
    if wandb_key:
        try:
            import wandb
            result = wandb.login(key=wandb_key, verify=True)
            if result:
                print("[preflight] W&B OK")
            else:
                print("[preflight] W&B warning: login returned False — check API key")
        except Exception as e:
            print(f"[preflight] W&B warning: {e!r}")
