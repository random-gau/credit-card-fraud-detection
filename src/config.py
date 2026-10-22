"""Paths and constants shared by every step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"

SEED = 42
TARGET = "Class"
