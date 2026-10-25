"""Generate synthetic inputs, never real model/reanalysis data."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="New directory for synthetic inputs and config.json")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lbm_background.demo import create_demo
    print(create_demo(args.directory))


if __name__ == "__main__":
    main()
