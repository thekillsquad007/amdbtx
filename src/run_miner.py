import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from amdbtx_miner.__main__ import main
main()
