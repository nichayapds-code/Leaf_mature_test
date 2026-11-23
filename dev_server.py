"""Local dev server: the API + the mobile page on one port.

    pip install -r requirements-dev.txt
    python dev_server.py            # http://localhost:8000
    python dev_server.py --lan      # also reachable from a phone on the same Wi-Fi
"""
import os
import socket
import sys

import uvicorn
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api.index import app  # noqa: E402

# locally the page is served by StaticFiles, so drop the path-agnostic fallback route
# (on Vercel the CDN serves public/ and only /api/* reaches the function)
app.router.routes = [r for r in app.router.routes if getattr(r, "path", "") != "/{full_path:path}"]
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "public"), html=True),
          name="public")

if __name__ == "__main__":
    host = "0.0.0.0" if "--lan" in sys.argv else "127.0.0.1"
    if host == "0.0.0.0":
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            print(f"open on your phone: http://{s.getsockname()[0]}:8000")
        except OSError:
            pass
    uvicorn.run(app, host=host, port=8000)
