"""Module entry point; preserve original main/load_config import paths."""

from .cli import load_config, main

__all__ = ["load_config", "main"]

if __name__ == "__main__":
    main()
