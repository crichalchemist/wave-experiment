"""Fixtures for forecasting tests — ensures Dignity-Model is on sys.path."""
import os
import sys

_dignity_path = os.path.join(os.path.dirname(__file__), "..", "..", "Dignity-Model")
_dignity_path = os.path.abspath(_dignity_path)
if _dignity_path not in sys.path:
    sys.path.insert(0, _dignity_path)
