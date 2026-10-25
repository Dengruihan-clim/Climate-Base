"""Configuration-driven entry points; help never submits jobs or runs LBM."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("diagnose", "convert"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--check", action="store_true", help="Preflight without writing outputs; see README for command-specific scope")
    args = parser.parse_args()
    try:
        from .config import load_config
        config = load_config(args.config, args.command)
        if args.command == "diagnose":
            from .pipeline import diagnose, check_diagnosis
            result = check_diagnosis(config) if args.check else diagnose(config)
        else:
            from .converter import convert, check_conversion
            result = check_conversion(config) if args.check else convert(config)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
