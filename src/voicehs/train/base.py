'''Functions to download F5-TTS checkpoints from HF.'''

from huggingface_hub import hf_hub_download
from pathlib import Path

def download_base_weights(project_dir: str = "ckpts", symlink_name: str = "pretrained_rus.safetensors"):
    """
    Download only the v2 checkpoint from HF to the default cache,
    then create a symlink in the project folder.
    """
    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)

    repo_id = "Misha24-10/F5-TTS_RUSSIAN"
    filename = "F5TTS_v1_Base_v2/model_last_inference.safetensors"

    print(f"Downloading {filename} from {repo_id} to HF cache...")
    cached_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision="main",
    )

    # Create symlink in project folder
    symlink_path = project_path / symlink_name
    if symlink_path.exists():
        symlink_path.unlink()  # remove old file/link if it exists
    symlink_path.symlink_to(cached_file)

    print(f"Symlink created at {symlink_path} → {cached_file}")

if __name__ == "__main__":
    download_base_weights()
