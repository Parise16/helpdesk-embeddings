from __future__ import annotations

import threading
import time
import webbrowser

import uvicorn

from app.config import APP_HOST, APP_PORT
from app.db_setup import initialize_database


def open_browser() -> None:
    time.sleep(1.2)
    webbrowser.open(f"http://{APP_HOST}:{APP_PORT}")


def main() -> None:
    initialize_database()
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app.web:app", host=APP_HOST, port=APP_PORT, reload=False)


if __name__ == "__main__":
    main()
