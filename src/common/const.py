import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
DATA_DIR = PROJECT_ROOT / "data"