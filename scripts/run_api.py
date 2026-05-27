import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("APP_ENV", "development") == "development"
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=reload_enabled)
