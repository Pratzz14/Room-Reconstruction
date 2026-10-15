"""Create a clearly marked analytic fixture for local viewer QA, never model evidence."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.fixtures import completed

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error("Choose a new output directory")
completed(args.output.resolve())
print(f"SYNTHETIC analytic fixture created: {args.output}")
