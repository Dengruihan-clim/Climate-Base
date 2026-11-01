"""Regenerate the source manifest for this independently built package."""

import hashlib
from pathlib import Path


def update_manifest():
    root = Path(__file__).resolve().parents[1]
    files = [path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt"]
    for directory in ("src", "tests", "examples", "tools"):
        files.extend(path for path in (root / directory).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and not path.suffix == ".pyc"
                     and not any(part.endswith(".egg-info") for part in path.parts))
    entries = []
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
    destination = root / "SHA256SUMS.txt"
    destination.write_text("\n".join(entries) + "\n", encoding="utf-8", newline="\n")
    print(f"Recorded {len(entries)} files in {destination}")


if __name__ == "__main__":
    update_manifest()
