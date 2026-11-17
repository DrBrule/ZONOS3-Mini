"""Stage and deploy the ZONOS3 Mini ZeroGPU Space.

    uv run --locked python servers/space/deploy.py --lexicon DIR [--repo-id Zyphra/ZONOS3-Mini] [--stage-only DIR]

Deploying creates the private Space, stores the local HF login token as its HF_TOKEN secret, requests
ZeroGPU hardware and uploads app.py + the space card + the zonos3 package, steering/, default_voices/ and laci_heads/
(from assets/), FuseAlign's fusealign_hub package, the studio's build (studio/dist, without the in-browser engine) as
studio/ and the gold pronunciation lexicon as lexicon/. The lexicon (lexicons/gold_phonemes_v1: manifest.json +
data.mdb) is in no model repo; --lexicon defaults to $PHONEME_LEXICON.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware, get_token

SPACE_ID = "Zyphra/ZONOS3-Mini"
REPO = Path(__file__).resolve().parents[2]
SPACE_FILES = ("app.py", "README.md", "requirements.txt", "packages.txt")
# Tracked repo paths and where the Space has them (app.py reads them relative to itself).
TRACKED = {
    "model/src/zonos3": "zonos3",
    "assets/steering": "steering",
    "assets/default_voices": "default_voices",
    "assets/laci_heads": "laci_heads",
}
# FuseAlign (speech editing's forced aligner), a submodule: its package as tracked there, with its license.
FUSEALIGN = REPO / "third_party" / "fusealign-multilingual"


def git(*args: str, repo: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def stage(dest: Path, lexicon: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in SPACE_FILES:
        shutil.copy2(REPO / "servers" / "space" / name, dest / name)
    for path in filter(None, git("ls-files", "-z", "--", *TRACKED).split("\0")):
        source = next(s for s in TRACKED if path.startswith(f"{s}/"))
        target = dest / TRACKED[source] / path.removeprefix(f"{source}/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / path, target)
    for path in filter(None, git("ls-files", "-z", "--", "fusealign_hub", repo=FUSEALIGN).split("\0")):
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FUSEALIGN / path, target)
    shutil.copy2(FUSEALIGN / "LICENSE", dest / "fusealign_hub" / "LICENSE")
    # The build as it is on disk (a fresh `pnpm --dir studio build` has untracked hashed assets until committed).
    shutil.copytree(REPO / "studio" / "dist", dest / "studio", ignore=shutil.ignore_patterns("web-engine"))
    (dest / "lexicon").mkdir()
    for name in ("manifest.json", "data.mdb"):
        shutil.copy2(lexicon / name, dest / "lexicon" / name)


def deploy(repo_id: str, lexicon: Path) -> None:
    token = get_token()
    if token is None:
        sys.exit("run 'hf auth login' first")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    api.add_space_secret(repo_id, "HF_TOKEN", token)
    api.request_space_hardware(repo_id, SpaceHardware.ZERO_A10G)
    describe = git("describe", "--always", "--dirty").strip()
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp), lexicon)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=tmp,
            commit_message=f"Deploy ZONOS3 {describe}",
            delete_patterns=[
                "zonos3/**",
                "steering/**",
                "default_voices/**",
                "laci_heads/**",
                "studio/**",
                "fusealign_hub/**",
                "lexicon/**",
            ],
        )
    print(f"https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default=SPACE_ID)
    parser.add_argument("--stage-only", type=Path, metavar="DIR")
    parser.add_argument(
        "--lexicon",
        type=Path,
        metavar="DIR",
        default=os.environ.get("PHONEME_LEXICON"),
        help="gold pronunciation lexicon directory (default: $PHONEME_LEXICON)",
    )
    args = parser.parse_args()
    if args.lexicon is None:
        parser.error("--lexicon (or PHONEME_LEXICON) is required")
    if args.stage_only is not None:
        if args.stage_only.exists() and any(args.stage_only.iterdir()):
            sys.exit(f"refusing to stage into non-empty {args.stage_only}")
        stage(args.stage_only, args.lexicon)
        print(args.stage_only)
        return
    deploy(args.repo_id, args.lexicon)


if __name__ == "__main__":
    main()
