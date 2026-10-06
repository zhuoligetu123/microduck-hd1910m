#!/usr/bin/env python3
"""HD1910M backend for the 4-observation/1-action testbench, not XL330 BAM.

Run from this checkout with: uv run --extra hd1910 scripts/testbench_hd1910.py --help
The sibling feetech_hls package supplies the guarded protocol adapter.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from feetech_hls.rl_test import main

if __name__ == '__main__':
    raise SystemExit(main())
