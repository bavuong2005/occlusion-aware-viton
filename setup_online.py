import os
import re
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STABLEVITON_DIR = BASE_DIR / "StableVITON"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "stablevton" / "ckpts"
RUNTIME_DIR = BASE_DIR / "runtime"

def run_cmd(cmd, cwd=None):
    print(">>", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def clone_repo():
    if STABLEVITON_DIR.exists():
        print("StableVITON already exists:", STABLEVITON_DIR)
        return
    run_cmd(["git", "clone", "https://github.com/rlawjdghek/StableVITON.git", str(STABLEVITON_DIR)])

def patch_attention():
    attn_file = STABLEVITON_DIR / "ldm" / "modules" / "attention.py"
    if not attn_file.exists():
        raise FileNotFoundError(f"Cannot find attention file: {attn_file}")

    with open(attn_file, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    content = re.sub(
        r"def forward\(self, x, context=None\):",
        "def forward(self, x, context=None, hint=None):",
        content
    )

    content = re.sub(
        r",\s*hint=hint",
        "",
        content
    )

    content = re.sub(
        r"self\.attn1\((.*?),\s*context=(.*?),\s*hint=hint\)",
        r"self.attn1(\1, context=\2)",
        content
    )

    if content != original:
        with open(attn_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("Patched attention.py")
    else:
        print("attention.py already patched or no change needed")

def download_checkpoint():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = CHECKPOINT_DIR / "VITONHD.ckpt"
    if ckpt_path.exists():
        print("Checkpoint already exists:", ckpt_path)
        return

    code = r'''
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="rlawjdghek/StableVITON",
    filename="ckpts/VITONHD.ckpt",
    local_dir="checkpoints/stablevton",
    local_dir_use_symlinks=False
)
print("Checkpoint downloaded")
'''
    run_cmd(["python", "-c", code], cwd=str(BASE_DIR))

def prepare_dirs():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print("Runtime and checkpoint directories ready")

if __name__ == "__main__":
    prepare_dirs()
    clone_repo()
    patch_attention()
    download_checkpoint()
    print("Setup completed successfully.")