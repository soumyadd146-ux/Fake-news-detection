import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MAX_INPUT_CHARACTERS = int(os.environ.get("MAX_INPUT_CHARACTERS", "20000"))
HISTORY_PAGE_SIZE = 10
