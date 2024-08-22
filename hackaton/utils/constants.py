from pathlib import Path
from typing import Final

import hackaton


DATA_FOLDER: Final[Path] = Path(hackaton.__file__).parent / "data"
DOTENV_FILE: Final[Path] = Path(hackaton.__file__).parent.parent / ".env"
CSS: Final[
    str
] = """.center {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
}"""
