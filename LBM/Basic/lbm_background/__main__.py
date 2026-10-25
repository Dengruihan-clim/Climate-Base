"""Command line entry point; --help does not import scientific libraries."""
import argparse
import json

from . import __version__


def main():
    parser = argparse.ArgumentParser(
        description="Build a stationary T42L20 dry LBM background from monthly CF NetCDF data.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", required=True, help="JSON configuration file")
    parser.add_argument("--check", action="store_true",
                        help="Check configuration and model reference files only; no reanalysis read or output")
    args = parser.parse_args()
    try:
        from .config import load_config
        from .pipeline import build, preflight
        config = load_config(args.config)
        if args.check:
            print(json.dumps(preflight(config), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(build(config), ensure_ascii=False, indent=2))
    except ModuleNotFoundError as error:
        parser.exit(2, f"Missing dependency {error.name!r}. Install requirements.txt in your Python environment.\n")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
