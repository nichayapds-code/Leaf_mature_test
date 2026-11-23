"""python -m pytest -q   (or: python tests/test_api.py)"""
import base64
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fastapi.testclient import TestClient  # noqa: E402
from api.index import app  # noqa: E402

client = TestClient(app)


def _payload(name, **kw):
    img = cv2.imread(os.path.join(ROOT, "examples", name))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return dict(image="data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(), **kw)


def test_health():
    assert client.get("/api/health").json()["ok"]


def test_pot_scale_matches_ruler():
    pot = client.post("/api/measure", json=_payload("kale1.jpg", pot_size=10)).json()
    ruler = client.post("/api/measure", json=_payload("kale1.jpg", use_ruler=True)).json()
    a, b = pot["scale"]["px_per_cm"], ruler["scale"]["px_per_cm"]
    assert pot["scale"]["source"] == "pot" and ruler["scale"]["source"] == "ruler"
    assert abs(a - b) / b < 0.02, (a, b)
    assert len(pot["leaves"]) == 7
    assert abs(pot["plants"][0]["length_cm"] - 14.5) < 0.5


def test_seeds_one_leaf_per_tap():
    taps = [[520, 760], [420, 900], [640, 1000]]
    r = client.post("/api/measure", json=_payload("kale3.jpg", pot_size=10, seeds=taps)).json()
    assert len(r["leaves"]) == 3


def test_no_scale_warns():
    r = client.post("/api/measure", json=_payload("kale3.jpg")).json()
    assert r["scale"]["px_per_cm"] is None and r["warnings"]


def test_bad_image():
    assert client.post("/api/measure", json={"image": "xx"}).status_code == 400


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
