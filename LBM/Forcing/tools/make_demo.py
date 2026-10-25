#!/usr/bin/env python3
"""Create a new synthetic diagnosis/conversion case for offline software QA."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True,
                        help='New directory; existing paths are never overwritten')
    parser.add_argument('--regression', action='store_true',
                        help='Generate twelve seasons and a raw-scale predictor/control CSV')
    parser.add_argument('--calendar', choices=('standard', '360_day'), default='standard')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from lbm_forcing.demo import create_demo
        paths = create_demo(args.directory, regression=args.regression, calendar=args.calendar)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f'ERROR: {error}\n')
    for command, path in paths.items():
        print(f'{command}: {path}')


if __name__ == '__main__':
    main()
