"""Convenience entry point: python train.py"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from spamdetector.train import main

if __name__ == "__main__":
    main()
