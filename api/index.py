"""Vercel Python function: POST /api/measure  (JSON in, JSON out)

Request JSON
  image        : JPEG/PNG as base64 or data URL (resize on the phone first; Vercel body limit 4.5 MB)
  pot_size     : nominal pot size in inches, e.g. 8 or 10 (uses POT_RIM_CM table)  [optional]
  pot_rim_cm   : measured outer rim diameter in cm (overrides pot_size)               [optional]
  use_ruler    : true -> read scale from an orange-stripe ruler in the photo          [optional]
  px_per_cm    : known scale                                                          [optional]
  mode         : "plant" (default) | "leaf"
  seeds        : [[x, y], ...] one tap per leaf, in pixels of the uploaded image      [optional]
  seed_frac    : leaf-split sensitivity (default 0.12)                                [optional]

Response JSON
  ok, scale, plants[], leaves[], warnings[], log[], image (annotated JPEG data URL),
  csv_leaves, csv_plants, size
"""
import base64
import os
import sys
import time

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import leaf_measure as lm  # noqa: E402

app = FastAPI(title="Leaf & plant measurement API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_SIDE = 2400          # images larger than this are downscaled before processing


class MeasureRequest(BaseModel):
    image: str
    pot_size: float | None = None
    pot_rim_cm: float | None = None
    use_ruler: bool = False
    px_per_cm: float | None = None
    mode: str = "plant"
    seeds: list[list[float]] | None = None
    seed_frac: float | None = None


def _decode(data: str):
    if "," in data[:100] and data.startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception:
        raise HTTPException(400, "image is not valid base64")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image (send JPEG or PNG)")
    return img


def _jpeg_data_url(img, quality=82, max_side=1400):
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@app.get("/api/health")
def health(request: Request = None):
    return {"ok": True, "version": app.version, "pot_sizes_cm": lm.POT_RIM_CM,
            "path_seen": request.url.path if request else None}


@app.post("/api/measure")
def measure(req: MeasureRequest):
    t0 = time.time()
    img = _decode(req.image)
    h0, w0 = img.shape[:2]
    f = 1.0
    if max(h0, w0) > MAX_SIDE:
        f = MAX_SIDE / max(h0, w0)
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    if req.mode not in ("plant", "leaf"):
        raise HTTPException(400, "mode must be 'plant' or 'leaf'")
    opts = dict(mode=req.mode, pot_size=req.pot_size, pot_rim_cm=req.pot_rim_cm)
    if req.seed_frac:
        opts["seed_frac"] = req.seed_frac
    if req.px_per_cm:
        opts["px_per_cm"] = req.px_per_cm * f
    elif req.use_ruler:
        opts["px_per_cm"] = "auto"
    if req.seeds:
        opts["seeds"] = [(x * f, y * f) for x, y in req.seeds]

    try:
        r = lm.analyze(img, opts)
    except Exception as e:  # keep the phone UI informative instead of a bare 500
        raise HTTPException(500, f"analysis failed: {type(e).__name__}: {e}")

    # report pixel values in the coordinates of the image the client sent
    if f != 1.0:
        for d in r["leaves"] + r["plants"]:
            for k, v in list(d.items()):
                if not isinstance(v, (int, float)) or isinstance(v, bool) or v is None:
                    continue
                if k.endswith("_px2") or k == "area_px":
                    d[k] = v / f ** 2
                elif k.endswith("_px") or k in ("cx", "cy"):
                    d[k] = v / f
        if r["scale"]["px_per_cm"]:
            r["scale"]["px_per_cm"] /= f

    warnings = []
    sc = r["scale"]
    if not sc["px_per_cm"]:
        warnings.append("ไม่มีสเกล ผลเป็น pixel อย่างเดียว ให้เลือกขนาดกระถางหรือเปิดใช้ไม้บรรทัด")
    pot = sc.get("pot")
    if pot and not pot.get("found"):
        warnings.append("หาขอบกระถางไม่เจอ ถ่ายตรงจากด้านบนให้เห็นปากกระถางครบทั้งวง")
    elif pot and pot.get("confidence") == "LOW":
        warnings.append("ความมั่นใจของขอบกระถางต่ำ (ขอบอาจล้นภาพ) ค่า cm อาจคลาดได้ถึง ±8% ควรถ่ายใหม่ให้เห็นปากกระถางครบทั้งวงและอยู่กลางภาพ")
    if sc.get("ruler") and "LOW" in (sc["ruler"].get("note") or ""):
        warnings.append("ความมั่นใจในการอ่านไม้บรรทัดต่ำ ให้ตรวจผลอีกครั้ง")
    if not r["leaves"]:
        warnings.append("ไม่พบใบ ตรวจแสง และดูว่าต้นไม้อยู่ในภาพ")
    if pot and pot.get("diff_vs_ruler_pct") is not None and abs(pot["diff_vs_ruler_pct"]) > 4:
        warnings.append(f"สเกลจากกระถางกับไม้บรรทัดต่างกัน {pot['diff_vs_ruler_pct']:+.1f}% (ใช้ค่าจากไม้บรรทัด)")

    return {
        "ok": True,
        "size": [w0, h0],
        "scale": sc,
        "plants": r["plants"],
        "leaves": r["leaves"],
        "warnings": warnings,
        "log": r["log"],
        "image": _jpeg_data_url(r["vis"]),
        "csv_leaves": lm.to_csv(r["leaves"], lm.LEAF_COLS),
        "csv_plants": lm.to_csv(r["plants"], lm.PLANT_COLS) if r["plants"] else "",
        "seconds": round(time.time() - t0, 2),
    }


# --- path-agnostic fallback ---------------------------------------------------------------
# Depending on the Vercel runtime/rewrite, the function may see "/api/measure", "/api/index",
# "/measure" or "/". Any other POST is treated as a measure request, any other GET as health,
# so the page works however the route is forwarded.
@app.api_route("/{full_path:path}", methods=["GET", "POST"])
async def fallback(full_path: str, request: Request):
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "body must be JSON")
        try:
            req = MeasureRequest(**body)
        except Exception as e:
            raise HTTPException(422, f"invalid request: {e}")
        return measure(req)
    return health(request)
