from pathlib import Path
import os
import runpy
import sys


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "NLP_Pipeline_Zynxis"
MAIN_FILE = PROJECT_DIR / "main.py"

if not MAIN_FILE.exists():
    raise FileNotFoundError(f"Could not find the project entry point at {MAIN_FILE}")

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(ROOT))
os.chdir(PROJECT_DIR)
runpy.run_path(str(MAIN_FILE), run_name="__main__")
