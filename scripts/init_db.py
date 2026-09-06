"""Create the local FatigueGuard SQLite database and its tables."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATABASE_PATH
from src.database import initialize_database


if __name__ == "__main__":
    path = initialize_database(DATABASE_PATH)
    print(f"Base SQLite inicializada: {path}")

