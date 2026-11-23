# Leaf & Plant Measurement (Leaf Measure)

A web app for measuring **length, width and leaf area of each leaf and of the whole plant** from a top-view photo, in pixels and centimetres.
Take the photo on a phone → choose the pot size → upload → see the results, and download a CSV.

- The scale comes from the **pot rim** (choose the pot size) or from an **orange ruler** in the photo
- Leaves are measured with **PCA (eigenvectors)** of the leaf pixels, plus the Feret diameter for nearly round leaves
- Lobed leaves are supported; overlapping leaves are split with watershed, and the user can **tap each leaf** to help the split

```
leaf-measure-app/
├── api/index.py          # Vercel Python function (FastAPI) : POST /api/measure, GET /api/health
├── lib/leaf_measure.py   # all image-processing code (works both as the API and as a CLI)
├── public/               # mobile web page (static) + PWA manifest + icons
├── examples/             # 3 test photos (kale in a 10" pot)
├── tests/test_api.py
├── vercel.json           # function config + rewrite /api/* → api/index.py
├── requirements.txt      # libraries for the server (Vercel)
└── dev_server.py         # run locally: page + API on a single port
```

---

## 1. Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python dev_server.py            # open http://localhost:8000
python dev_server.py --lan      # open from a phone on the same Wi-Fi (prints the URL)
python tests/test_api.py        # tests
```

## 2. Push to GitHub

```bash
cd leaf-measure-app
git init
git add .
git commit -m "Leaf Measure v0.1"
git branch -M main
git remote add origin https://github.com/<user>/leaf-measure-app.git   # create the repo on GitHub first (private is fine)
git push -u origin main
git tag v0.1 && git push --tags
```

## 3. Deploy to Vercel

1. Go to https://vercel.com/new → **Import** the GitHub repo from step 2
2. Framework Preset: **Other**, and leave Build Command and Output Directory empty (they're already set in `vercel.json`)
3. Press **Deploy** and wait ~2–4 minutes (the first build installs OpenCV/SciPy)
4. Open `https://<project>.vercel.app/api/health` → you should see `{"ok": true, ...}`
5. Open `https://<project>.vercel.app` on the phone → **Add to Home Screen** and it can be used like an app

From then on, every `git push` to `main` deploys automatically. Branches/PRs get a separate Preview URL.

> Technical notes for Vercel
> - Python 3.12 (`.python-version`); the function bundle is ~350 MB (the limit for Python is 500 MB)
> - Request body limit is 4.5 MB → the web page shrinks the image to a 1600 px long side before sending (~0.7 MB)
> - Each image takes ~1–3 s on the server; `maxDuration` is set to 60 s
> - The function's first call after it has been idle a while may be a few seconds slower (cold start)

## 4. How to use (phone)

1. Press **Take photo**: shoot from directly above, show the whole pot rim, with the pot in the center
2. Choose the **pot size** (5", 5.5", 6", 8", 10", 11" or "Measured myself" in cm)
3. (Optional) tap 1 point on each leaf to help split overlapping leaves
4. Press **Measure** → you get the plant length × width, canopy diameter, leaf area, and a per-leaf table
5. Download the plant CSV / leaf CSV / annotated image (on a phone, this opens the Share sheet)

## 5. API

`POST /api/measure` (JSON)

| field | type | description |
|---|---|---|
| `image` | string | JPEG/PNG base64 or data URL (shrink to ≤ 1600–2400 px first) |
| `pot_size` | number | nominal pot size (inches) → converted to cm with the `POT_RIM_CM` table |
| `pot_rim_cm` | number | pot rim diameter you measured (cm); overrides `pot_size` |
| `use_ruler` | bool | read the scale from the orange ruler in the photo |
| `px_per_cm` | number | known scale (if you have one) |
| `mode` | `"plant"` \| `"leaf"` | whole plant in a pot / detached leaves |
| `seeds` | `[[x,y],...]` | 1 point per leaf (pixel coordinates of the uploaded image) |

The response has `scale`, `plants[]`, `leaves[]`, `warnings[]` (Thai), `image` (annotated image), `csv_leaves`, `csv_plants`.
All pixel values are in the coordinates of the image that was sent.

```bash
curl -s https://<project>.vercel.app/api/measure -H 'content-type: application/json' \
  -d "{\"pot_size\":10,\"image\":\"$(base64 -w0 examples/kale1.jpg)\"}" | head -c 600
```

## 6. Pot table (`lib/leaf_measure.py` → `POT_RIM_CM`)

The shop's size in inches ≠ the real rim diameter. The values come from seller specs (Kanok, SPRING, PNP):

| size | rim (cm) |
|---|---|
| 5" | 12.7 |
| 5.5" | 13.7 |
| 6" | 15.5 |
| 8" | 19.0 |
| 10" | 26.0 |
| 11" | 28.0 |

If you use one brand regularly, measure the rim with a tape measure once (outer edge to outer edge) and edit the table → the most accurate option.

## 7. Accuracy and limitations (from the tests)

| item | result |
|---|---|
| pot scale vs ruler (rim fully in frame) | +0.4%, −1.7% |
| pot rim going off the frame | can be off by up to ±8% → the app warns "low confidence" |
| ruler at 0.4×–3.6× resolution, blurred/dark/bright | ≤ 0.8% |
| whole-plant size across resolutions | ±0.6% |
| splitting individual leaves when they overlap | can shift → tap each leaf to help |

**What you must know:** cm values are computed at the **pot-rim level**, while leaves sit lower → values come out a little small (depends on camera height; at ~24 cm, about 4% per 1 cm of depth).
To compare over time (growth), use the same pot, the same soil level and the same camera height every time.

## 8. Using it as a CLI (no web)

```bash
pip install -r requirements-dev.txt
python lib/leaf_measure.py examples/kale1.jpg --pot-size 10
python lib/leaf_measure.py examples/kale1.jpg --px-per-cm auto --pot-size 10   # ruler + pot cross-check
python lib/leaf_measure.py leaves.jpg --mode leaf --pot-rim-cm 26
python lib/leaf_measure.py -h      # all options
```
It writes `_measured.png`, `_measure.csv`, `_plant.csv`, `_mask.png`, `_hist.png`, `_pot.png`, `_ruler.png` next to the input image.

## 9. How it works (summary)

1. **Separate the leaves**: computes 4 greenness indices (ExG, ExGR, Lab a\*, HSV), picks the one whose histogram has the deepest valley between "background" and "leaf", and thresholds at that valley
2. **Split the leaves**: seeds = peaks of the distance transform (or the user's taps) → priority-flood watershed on the Lab color gradient + distance
3. **Measure each leaf**: PCA of the pixel coordinates → eigenvectors v1 (length), v2 (width); project pixels onto the axes and measure the extent (Feret is used if √(λ2/λ1) > 0.8)
4. **Whole plant**: PCA + Feret + enclosing circle + convex hull of all the plant's pixels
5. **Pot scale**: radial scan from the center of the soil outward → RANSAC ellipse on the outer rim edge → major axis = diameter
6. **Ruler scale**: straighten the orange stripe → FFT of the tick period → identify mm / 1/16" units from period ratios
