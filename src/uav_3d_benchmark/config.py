from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = str(_PROJECT_ROOT)
DATA_ROOT = str(_PROJECT_ROOT / "data")
OUTPUT_ROOT = str(_PROJECT_ROOT / "outputs")

EUROC_ROOT = str(_PROJECT_ROOT / "data" / "euroc")
USEGEO_ROOT = str(_PROJECT_ROOT / "data" / "usegeo")
