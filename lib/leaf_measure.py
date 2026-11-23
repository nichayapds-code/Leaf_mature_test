#!/usr/bin/env python3
"""
Leaf length / width (pixels) from a top-view image  — supports lobed / palmate leaves.

1) Colour separation (histogram based)
   Several "greenness" indices are computed, each stretched to 0..255:
       exg   = 2G - R - B                       (Excess Green)
       exgr  = ExG - (1.4R - G)                 (Excess Green minus Excess Red)
       lab_a = -(a*) from CIELab                 (green = negative a*)
       hsv   = hue closeness to green * saturation
   For each index: Otsu threshold + separability  eta = sigma_between^2 / sigma_total^2.
   Index choice: in each histogram find the background-peak / leaf-peak pair with the
   deepest valley and best Fisher separation (score = depth * fisher); highest score wins
   (or pick with --index). Falls back to Otsu eta if no valid pair.
   The threshold is then refined to the valley (histogram minimum) between the
   two dominant peaks, which is more robust than Otsu when classes are unequal in size.
   A histogram figure (<name>_hist.png) shows all indices and the chosen threshold.

2) Mask cleaning — kept gentle so the notches between lobes are NOT closed:
   small open, fill holes inside each leaf, drop blobs < --min-area.
   Petiole is stripped by default (--keep-stem to keep it): opening with a kernel relative to leaf size removes the petiole.

3) Split plant into leaves (rosette / overlapping leaves)
   Seeds = distance-transform peaks (or --seeds), watershed on Lab colour gradient +
   inverted distance, so borders follow colour edges between leaves.
   --mode plant (default) splits; --mode leaf treats each blob as one lobed leaf.

4) Whole plant (canopy), --mode plant: blobs within 5% of the image diagonal are one plant.
   PCA length/width of all plant pixels, Feret max diameter + perpendicular width,
   min enclosing circle diameter, convex-hull (canopy) area, leaf area, cover = leaf/hull.
   -> <name>_plant.csv

5) Measurement per leaf
   PCA   : eigenvectors of pixel-coordinate covariance -> v1 (major), v2 (minor)
           length = extent of pixels projected on v1, width = extent on v2
   Feret : max caliper diameter on the convex hull (lobe tip to lobe tip) and the
           width perpendicular to it. Independent of PCA.
   anisotropy = sqrt(lambda2/lambda1). Near 1 (palmate / round leaves) means the PCA
           direction is ill-defined -> flagged, and Feret is reported as primary.
   lobes : peaks (tips) of the radial-distance profile centroid->contour whose
           prominence > --lobe-depth * Feret length (notches also drawn).

Usage
  python leaf_measure.py leaf.jpg
  python leaf_measure.py plant.jpg --px-per-cm auto     # read scale from ruler
  python leaf_measure.py plant.jpg --px-per-cm 46.4
  python leaf_measure.py leaves.jpg --mode leaf          # detached (lobed) leaves
  python leaf_measure.py leaf.jpg --mask my_mask.png
"""
import argparse
import csv
import os

import cv2
import numpy as np

# ===================================================================== colour indices
def _stretch(x, mask=None):
    """Robust stretch to 0..255 using 1st/99th percentile."""
    v = x[mask] if mask is not None else x.ravel()
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.9)
    if hi - lo < 1e-6:
        hi = lo + 1
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def color_indices(img_bgr):
    f = img_bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    s = b + g + r + 1e-6
    rn, gn, bn = r / s, g / s, b / s                 # chromatic coords -> less shadow sensitive
    exg = 2 * gn - rn - bn
    exr = 1.4 * rn - gn
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, sat = hsv[..., 0] * 2, hsv[..., 1] / 255      # hue in degrees
    hue_green = np.cos(np.radians(h - 110)).clip(0)  # peak at ~110° (green)
    return {
        "exg": _stretch(exg),
        "exgr": _stretch(exg - exr),
        "lab_a": _stretch(-a),
        "hsv": _stretch(hue_green * sat),
    }


# ===================================================================== threshold
def otsu_eta(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    i = np.arange(256)
    w0 = np.cumsum(p)
    mu = np.cumsum(p * i)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0))
    sb = np.nan_to_num(sb)
    t = int(np.argmax(sb))
    var_t = (p * (i - mu_t) ** 2).sum()
    return t, sb[t] / (var_t + 1e-12), hist


def valley_refine(hist, t_otsu, smooth=9):
    """Move threshold to the histogram minimum between the peaks on each side of Otsu."""
    k = np.ones(smooth) / smooth
    h = np.convolve(hist, k, mode="same")
    p_lo = int(np.argmax(h[:t_otsu + 1])) if t_otsu > 0 else 0
    p_hi = t_otsu + int(np.argmax(h[t_otsu:]))
    if p_hi - p_lo < 3:
        return t_otsu
    return p_lo + int(np.argmin(h[p_lo:p_hi + 1]))


def pair_score(hist, min_fg=0.003, max_fg=0.6):
    """Find the best (background-peak, leaf-peak) pair in a histogram.
    score = valley depth * Fisher separation of the two classes it creates.
    End bins (0/255) are ignored because the percentile stretch piles clipped pixels there."""
    from scipy.signal import find_peaks
    h = hist.astype(np.float64).copy()
    h[0], h[255] = h[1], h[254]
    hs = np.convolve(h, np.ones(11) / 11, "same")
    pk, _ = find_peaks(hs, prominence=hs.max() * 0.005, distance=15)
    pk = pk[(pk > 8) & (pk < 247)]                  # ignore clipping piles at the ends
    x = np.arange(256)
    best = None
    for i in range(len(pk)):
        for j in range(i + 1, len(pk)):
            a, b = pk[i], pk[j]
            v = a + int(np.argmin(hs[a:b + 1]))
            fg = h[v:].sum() / h.sum()
            if not (min_fg < fg < max_fg) or b - a < 25:
                continue
            depth = 1 - hs[v] / min(hs[a], hs[b])
            w0, w1 = h[:v], h[v:]
            m0, m1 = (x[:v] * w0).sum() / w0.sum(), (x[v:] * w1).sum() / w1.sum()
            s0 = np.sqrt(((x[:v] - m0) ** 2 * w0).sum() / w0.sum())
            s1 = np.sqrt(((x[v:] - m1) ** 2 * w1).sum() / w1.sum())
            fisher = (m1 - m0) / (s0 + s1 + 1e-9)
            sc = depth ** 2 * min(fisher, 4.0)       # a deep valley matters most
            if best is None or sc > best["score"]:
                best = dict(score=sc, peak_bg=int(a), peak_leaf=int(b), valley=int(v),
                            depth=depth, fisher=fisher, fg=fg)
    return best


def choose_index(indices, forced=None):
    stats = {}
    for name, g in indices.items():
        t, eta, hist = otsu_eta(g)
        ps = pair_score(hist)
        stats[name] = dict(otsu=t, eta=eta, hist=hist, pair=ps,
                           valley=ps["valley"] if ps else t,
                           score=ps["score"] if ps else 0.0)
    if forced:
        return forced, stats
    if any(st["score"] > 0 for st in stats.values()):
        best = max(stats, key=lambda n: stats[n]["score"])
    else:                                            # fallback: Otsu separability
        best = max(stats, key=lambda n: stats[n]["eta"])
    return best, stats


def save_histograms(indices, stats, best, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(indices)
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 6.5))
    for j, n in enumerate(names):
        st = stats[n]
        ax = axes[0, j]
        ax.bar(np.arange(256), st["hist"], width=1, color="#4a8f3c" if n == best else "#999")
        ax.axvline(st["otsu"], color="orange", ls="--", label=f"Otsu {st['otsu']}")
        ax.axvline(st["valley"], color="red", label=f"Valley {st['valley']}")
        ax.set_yscale("log")
        ax.set_title(f"{n}  score={st['score']:.2f}  η={st['eta']:.2f}" + ("  ← chosen" if n == best else ""),
                     fontweight="bold" if n == best else None)
        ax.legend(fontsize=8)
        axes[1, j].imshow(indices[n], cmap="gray")
        axes[1, j].contour(indices[n] > st["valley"], levels=[0.5], colors="r", linewidths=0.6)
        axes[1, j].axis("off")
    fig.suptitle("Greenness index histograms — score = valley depth × Fisher separation (higher = cleaner leaf/background split)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ===================================================================== ruler -> px/cm
def _strip_ruler(img, comp, pad=45):
    """Straighten the ruler: sample the image along the ruler's PCA axes."""
    ys, xs = np.nonzero(comp)
    P = np.c_[xs, ys].astype(float)
    c = P.mean(0)
    ev, evec = np.linalg.eigh(np.cov((P - c).T))
    u, v = evec[:, 1], evec[:, 0]
    pu, pv = (P - c) @ u, (P - c) @ v
    L, W = int(np.ptp(pu)), int(np.ptp(pv)) + 2 * pad
    X, Y = np.meshgrid(np.arange(L), np.arange(W))
    mx = (c[0] + (pu.min() + X) * u[0] + (pv.min() - pad + Y) * v[0]).astype(np.float32)
    my = (c[1] + (pu.min() + X) * u[1] + (pv.min() - pad + Y) * v[1]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR), (c, u, v, pu.min(), pu.max(), pv.min() - pad)


def _find_stripe(img, hsv_lo, hsv_hi):
    """Ruler stripe = elongated coloured blob(s). Collinear pieces (ruler partly covered by a
    leaf) are merged. Returns uint8 mask or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, l, st, _ = cv2.connectedComponentsWithStats(m, 8)
    pieces = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 300:
            continue
        cnt, _ = cv2.findContours((l == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (cx, cy), (w, h), ang = cv2.minAreaRect(cnt[0])
        long_, short = max(w, h), max(1.0, min(w, h))
        fill = st[i, cv2.CC_STAT_AREA] / (w * h + 1e-9)
        if long_ / short > 4 and fill > 0.55:                 # a straight bar, not soil/pot
            th = np.radians(ang if w >= h else ang + 90)
            pieces.append(dict(i=i, c=np.array([cx, cy]), d=np.array([np.cos(th), np.sin(th)]),
                               short=short, area=st[i, cv2.CC_STAT_AREA]))
    if not pieces:
        return None
    pieces.sort(key=lambda p: -p["area"])
    main = pieces[0]
    keep = [main]
    for p in pieces[1:]:
        par = abs(float(p["d"] @ main["d"])) > 0.995                   # same direction
        off = abs(float((p["c"] - main["c"]) @ np.array([-main["d"][1], main["d"][0]])))
        if par and off < main["short"] and abs(p["short"] - main["short"]) < 0.3 * main["short"]:
            keep.append(p)
    out = np.zeros(m.shape, np.uint8)
    for p in keep:
        out[l == p["i"]] = 1
    return out


def _band_peaks(prof, pmin, pmax, k=6, N=1 << 15):
    """Top spectral peaks (period px, sharpness) of a 1-D tick profile."""
    from scipy.signal import find_peaks
    prof = prof - cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 25).ravel()
    prof = prof * np.hanning(len(prof))
    F = np.abs(np.fft.rfft(prof, N))
    per = 1 / np.maximum(np.fft.rfftfreq(N), 1e-12)
    sel = (per > pmin) & (per < pmax)
    if not sel.any():
        return []
    base = np.median(F[sel]) + 1e-9
    pk, _ = find_peaks(np.where(sel, F, 0), distance=8)
    pk = pk[np.argsort(F[pk])[::-1][:k]]
    # parabolic interpolation of the peak for sub-bin precision
    out = []
    for i in pk:
        if 0 < i < len(F) - 1 and F[i] / base > 10:
            y0, y1, y2 = F[i - 1], F[i], F[i + 1]
            d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
            f = (i + d) / N
            out.append((1 / f, F[i] / base))
    return out


# tick spacings (cm) and the pairs whose period ratio identifies them
_UNIT_CM = {"1mm": 0.1, "5mm": 0.5, "1cm": 1.0, "1/16in": 2.54 / 16, "1/8in": 2.54 / 8,
            "1/4in": 2.54 / 4, "1/2in": 2.54 / 2, "1in": 2.54}
_PAIRS = [("1mm", "1/16in"), ("1/8in", "5mm"), ("1/4in", "1cm"), ("1/16in", "1/8in"),
          ("1mm", "5mm"), ("5mm", "1cm")]


def ruler_scale(img, hsv_lo=(5, 120, 120), hsv_hi=(22, 255, 255), debug_path=None,
                ruler_cm=(5, 40)):
    """px/cm from a ruler with a coloured stripe (default orange).
    1) find the stripe (elongated blob; collinear pieces split by a leaf are merged),
       straighten it, ignore columns hidden by leaves
    2) FFT of thin bands across the ruler -> tick periods (sub-pixel)
    3) identify units from period ratios: 1mm:1/16in = 1/1.5875, 1/8in:5mm, 1/4in:1cm, ...
       candidate px/cm must fit the visible ruler length (ruler_cm range) -> resolution independent
    4) refine with every peak that matches the chosen scale; report confidence"""
    comp = _find_stripe(img, hsv_lo, hsv_hi)
    if comp is None:
        return None, "no ruler stripe found"
    crop, _ = _strip_ruler(img, comp)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    stripe = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi)) > 0
    rows = np.where(stripe.mean(1) > 0.3)[0]
    if len(rows) < 3:
        return None, "ruler stripe too thin"
    vis = (stripe[rows].mean(0) > 0.5).astype(np.uint8).reshape(1, -1)
    vis = cv2.erode(cv2.dilate(vis, np.ones((1, 9), np.uint8)), np.ones((1, 25), np.uint8)).ravel() > 0
    Lvis = int(vis.sum())
    if Lvis < 150:
        return None, "visible ruler too short"
    lo_ppcm, hi_ppcm = crop.shape[1] / ruler_cm[1], crop.shape[1] / ruler_cm[0]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(float)
    band = max(4, int(len(rows) * 0.15))
    peaks = []
    for r0 in range(0, g.shape[0] - band, max(2, band // 2)):
        prof = g[r0:r0 + band].mean(0)
        prof[~vis] = prof[vis].mean()
        peaks += _band_peaks(prof, 2.3, min(Lvis / 6, hi_ppcm * 2.6))
    if not peaks:
        return None, "no tick pattern found"
    P = np.array(peaks)

    def support(ppcm, tol=0.012):
        """matched spectral energy minus energy of strong peaks the scale cannot explain
        (a tick unit or its 2nd/3rd harmonic)."""
        hit, miss, units = 0.0, 0.0, set()
        for q, e in P:
            ok = False
            for un, cm in _UNIT_CM.items():
                for h in (1, 2, 3):
                    if abs(q - ppcm * cm / h) / q < tol:
                        ok = True
                        if h == 1:
                            units.add(un)
                        break
                if ok:
                    break
            if ok:
                hit += e
            else:
                miss += e
        score = hit - 1.5 * miss
        # finest ticks that should be resolvable (>= 3 px) but were not seen -> wrong scale
        emax = P[:, 1].max()
        if ppcm * 0.1 >= 3.0 and "1mm" not in units and units & {"5mm", "1cm"}:
            score -= 3 * emax
        if ppcm * 2.54 / 16 >= 3.0 and "1/16in" not in units and units & {"1/8in", "1/4in"}:
            score -= 3 * emax
        return score, units

    best = None
    for i, (p1, e1) in enumerate(P):
        for p2, e2 in P[i + 1:]:
            a_, b_ = sorted((p1, p2))
            for u1, u2 in _PAIRS:
                r = _UNIT_CM[u2] / _UNIT_CM[u1]
                if abs(b_ / a_ - r) / r < 0.012:
                    ppcm = a_ / _UNIT_CM[u1]
                    if lo_ppcm < ppcm < hi_ppcm:
                        sup, units = support(ppcm)
                        if best is None or sup > best[0]:
                            best = (sup, ppcm, units)
    confident = best is not None
    if best is None:                       # single scale: assume the finest strong ticks are mm
        p = P[np.argmax(P[:, 1]), 0]
        cands = [p / cm for cm in (0.1, 0.5, 1.0) if lo_ppcm < p / cm < hi_ppcm]
        if not cands:
            return None, "tick periods do not fit any ruler unit"
        ppcm = cands[0]
        best = (0, ppcm, support(ppcm)[1])
    _, ppcm0, units = best
    est, w = [], []
    for q, e in P:
        for un, cm in _UNIT_CM.items():
            if abs(q - ppcm0 * cm) / q < 0.012:
                est.append(q / cm)
                w.append(e * q)
                break
    ppcm = float(np.average(est, weights=w))
    spread = float(np.sqrt(np.average((np.array(est) - ppcm) ** 2, weights=w)) / ppcm * 100)
    metric = units & {"1mm", "5mm", "1cm"}
    imper = units & {"1/16in", "1/8in", "1/4in", "1/2in", "1in"}
    conf = "high" if confident and metric and imper and spread < 1 else ("medium" if confident else "LOW")
    note = (f"{ppcm:.2f} px/cm, confidence {conf} (ticks: {', '.join(sorted(units))}; "
            f"spread {spread:.2f}%)")
    hidden = 1 - Lvis / crop.shape[1]
    if hidden > 0.05:
        note += f"; {hidden * 100:.0f}% of ruler hidden/ignored"
    if debug_path:
        out = crop.copy()
        out[:, ~vis] = (out[:, ~vis] * 0.4).astype(np.uint8)
        x = float(np.argmax(vis))
        while x < out.shape[1]:
            cv2.line(out, (int(round(x)), 0), (int(round(x)), 10), (0, 0, 255), 2)
            x += ppcm
        cv2.putText(out, f"{ppcm:.2f} px/cm  conf={conf}  (red = every 1 cm)", (10, out.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(debug_path, out)
    return ppcm, note


# ===================================================================== pot rim -> px/cm
# nominal pot size ("inch" as sold in Thailand) -> outer rim diameter in CM, from seller specs
# (Kanok 8-POT-5/6/8/10, SPRING and PNP at HomePro, checked 2026-11). Nominal inches are NOT
# rim inches: an "8 in" pot is ~19 cm (7.5 in), a "10 in" pot ~26 cm (10.2 in). Brands differ by
# ~1-3 %, so measure your own pots once and edit this table, or pass --pot-rim-in.
POT_RIM_CM = {5: 12.7, 5.5: 13.7, 6: 15.5, 8: 19.0, 10: 26.0, 11: 28.0}


def _fit_circle(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    c = x[:2]
    return c, float(np.sqrt(max(x[2] + c @ c, 1e-9)))


def _robust_circle(P, it=8):
    keep = np.ones(len(P), bool)
    for _ in range(it):
        c, r = _fit_circle(P[keep])
        res = np.abs(np.linalg.norm(P - c, axis=1) - r)
        mad = np.median(res[keep]) + 1e-6
        keep = res < max(3.0, 3 * 1.4826 * mad)
    return c, r, keep


def _rim_from_mask(pot, h, w):
    """Ring analysis on one candidate pot mask. Returns (score, result) or (0, reason)."""
    pot = cv2.morphologyEx(pot, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, l = cv2.connectedComponents((1 - pot).astype(np.uint8))
    cid = l[h // 2, w // 2]
    if cid == 0:
        return 0, "image centre is not inside a pot"
    disk = cv2.morphologyEx((l == cid).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    area = disk.sum()
    if area < 0.05 * h * w or area > 0.95 * h * w:
        return 0, "no pot opening around the image centre"
    ys, xs = np.nonzero(disk)
    c0 = np.array([xs.mean(), ys.mean()])
    r0 = np.sqrt(area / np.pi)
    outer = []
    rs = np.arange(int(r0 * 0.6), int(r0 * 2.0))
    for th in np.linspace(0, 2 * np.pi, 720, endpoint=False):
        d = np.array([np.cos(th), np.sin(th)])
        pts = c0 + rs[:, None] * d
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] < h - 1)
        vals = np.zeros(len(rs), bool)
        p = pts[inside].astype(int)
        vals[inside] = pot[p[:, 1], p[:, 0]] > 0
        dk = np.zeros(len(rs), bool)
        dk[inside] = disk[p[:, 1], p[:, 0]] > 0
        out_of_disk = np.where(~dk)[0]
        if not len(out_of_disk):
            continue
        on = np.where(vals & (np.arange(len(rs)) >= out_of_disk[0]))[0]   # rim starts at opening edge
        if not len(on):
            continue
        j, gap, last = on[0], 0, on[0]
        while j < len(rs) and inside[j]:
            if vals[j]:
                last, gap = j, 0
            else:
                gap += 1
                if gap > 10:
                    break
            j += 1
        if j >= len(rs) or not inside[j]:
            continue                                   # ran off the image: unreliable
        outer.append(c0 + rs[last] * d)
    if len(outer) < 60:
        return 0, "rim not visible enough"
    outer = np.array(outer)
    # RANSAC ellipse (a tilted camera makes the rim an ellipse; rays blocked by the ruler or
    # running into neighbouring pots are outliers)
    def resid(el):
        (ex, ey), (d1, d2), ea = el
        t = np.radians(ea)
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        q = (outer - [ex, ey]) @ R.T
        return np.abs(np.hypot(q[:, 0] / (d1 / 2 + 1e-9), q[:, 1] / (d2 / 2 + 1e-9)) - 1)
    rng = np.random.default_rng(0)
    P32 = outer.astype(np.float32)
    best_k, keep = 0, None
    for _ in range(400):
        idx = rng.choice(len(outer), 6, replace=False)
        try:
            el = cv2.fitEllipse(P32[idx])
        except cv2.error:
            continue
        d1, d2 = el[1]
        if min(d1, d2) < 1 or min(d1, d2) / max(d1, d2) < 0.75 or max(d1, d2) < 1.6 * r0:
            continue
        k = resid(el) < 0.012
        if k.sum() > best_k:
            best_k, keep = k.sum(), k
    if keep is None:
        return 0, "rim edge too irregular"
    for _ in range(3):                                   # refine on inliers
        el = cv2.fitEllipse(P32[keep])
        keep = resid(el) < 0.012
    if keep.sum() < 40:
        return 0, "rim edge too irregular"
    el = cv2.fitEllipse(outer[keep].astype(np.float32))
    d1, d2 = el[1]
    ratio = min(d1, d2) / max(d1, d2)
    if ratio < 0.75:
        return 0, "rim not round (pot not seen from the top?)"
    score = keep.mean() * len(outer) / 720 * ratio
    return score, dict(ellipse=el, outer=outer, keep=keep, coverage=len(outer) / 720,
                       inliers=float(keep.mean()))


def pot_rim(img, leaf_mask=None, debug_path=None):
    """Outer rim of a black/grey nursery pot seen from the top (pot centred in the photo).
    Candidate pot masks (grey pot, black pot) -> large low-chroma regions only -> soil opening =
    non-pot region at the image centre -> rays outward find the outer rim edge -> robust circle
    rejects rays running into neighbouring pots -> ellipse on inliers. The ellipse MAJOR axis is
    the true rim diameter (camera tilt only shortens the other axis).
    Returns (diameter_px, info) or (None, reason)."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.0), cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    lm = np.zeros((h, w), bool) if leaf_mask is None else leaf_mask > 0
    cands = {"grey pot": (S < 45) & (V > 50) & (V < 215) & ~lm,
             "black pot": (V < 75) & (S < 110) & ~lm}
    best, reasons = None, []
    for name, m in cands.items():
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        sc, res = _rim_from_mask(m, h, w)
        if sc and (best is None or sc > best[0]):
            best = (sc, name, res)
        elif not sc:
            reasons.append(f"{name}: {res}")
    if best is None:
        return None, "; ".join(reasons)
    sc, name, res = best
    (ex, ey), (d1, d2), ea = res["ellipse"]
    D = float(max(d1, d2))
    tilt = float(np.degrees(np.arccos(min(1.0, min(d1, d2) / D))))
    info = dict(kind=name, diameter_px=D, center=(ex, ey), axes=(d1, d2), tilt_deg=tilt,
                inliers=res["inliers"], coverage=res["coverage"],
                # < 50 % inliers = rim largely off-frame/occluded: diameter can swing +-8 %
                confidence="high" if res["inliers"] > 0.6 and res["coverage"] > 0.55 else
                ("medium" if res["inliers"] >= 0.5 else "LOW"))
    if debug_path:
        vis = img.copy()
        for q in res["outer"][res["keep"]]:
            cv2.circle(vis, tuple(q.astype(int)), 2, (0, 255, 0), -1)
        for q in res["outer"][~res["keep"]]:
            cv2.circle(vis, tuple(q.astype(int)), 2, (0, 0, 255), -1)
        cv2.ellipse(vis, res["ellipse"], (0, 255, 255), 2)
        _text(vis, f"rim D={D:.0f}px  {name}  inliers {res['inliers'] * 100:.0f}%  tilt~{tilt:.0f}deg",
              (15, 35), 0.7, (0, 255, 255))
        cv2.imwrite(debug_path, vis)
    return D, info


# ===================================================================== mask
def fill_holes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


def clean_mask(mask, min_area, open_k=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)   # tiny close only – keeps lobe notches
    mask = fill_holes(mask)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return keep


def remove_stem(leaf, frac=0.06):
    """Opening with kernel ~ frac * sqrt(area) removes thin petiole; keep largest part."""
    area = cv2.countNonZero(leaf)
    ks = max(3, int(frac * np.sqrt(area)) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    op = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(op, 8)
    if n <= 1:
        return leaf
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    body = np.where(lab == big, 255, 0).astype(np.uint8)
    # restore lobe edges eroded by opening: dilate back, but stay inside original leaf
    body = cv2.bitwise_and(cv2.dilate(body, k), leaf)
    return body


# ===================================================================== split plant -> leaves
def _peak_local_max(img, min_distance, threshold_rel, mask):
    """Local maxima at least min_distance apart (Chebyshev), above threshold_rel*max, inside mask,
    not within min_distance of the image border. Same idea as skimage.feature.peak_local_max."""
    from scipy import ndimage as ndi
    size = 2 * min_distance + 1
    mx = ndi.maximum_filter(img, size=size, mode="constant", cval=-np.inf)
    cand = (img == mx) & mask & (img > threshold_rel * img[mask].max())
    b = min_distance
    cand[:b, :] = cand[-b:, :] = False
    cand[:, :b] = cand[:, -b:] = False
    ys, xs = np.nonzero(cand)
    order = np.argsort(-img[ys, xs], kind="stable")
    keep = []
    for i in order:
        y, x = ys[i], xs[i]
        if all(max(abs(y - ky), abs(x - kx)) > min_distance for ky, kx in keep):
            keep.append((y, x))
    return np.array(keep, dtype=int).reshape(-1, 2)


def _watershed(elev, markers, mask):
    """Marker-controlled priority-flood watershed (same algorithm as skimage.segmentation.watershed:
    pixels are flooded in order of elevation, ties by insertion age, 4-connectivity)."""
    import heapq
    h, w = elev.shape
    lab = markers.astype(np.int32).copy()
    lab[~mask] = 0
    E = elev.ravel()
    L = lab.ravel()
    M = mask.ravel()
    heap = []
    age = 0
    for p in np.flatnonzero(L > 0):
        heapq.heappush(heap, (float(E[p]), age, int(p)))
        age += 1
    while heap:
        _, _, p = heapq.heappop(heap)
        y, x = divmod(p, w)
        lp = L[p]
        for q, ok in ((p - w, y > 0), (p + w, y < h - 1), (p - 1, x > 0), (p + 1, x < w - 1)):
            if ok and M[q] and L[q] == 0:
                L[q] = lp
                heapq.heappush(heap, (float(E[q]), age, q))
                age += 1
    return L.reshape(h, w)


def split_leaves(img_bgr, comp, seed_frac=0.12, grad_w=0.7, seeds=None):
    """Watershed a connected plant blob into individual leaves.
    seeds    : distance-transform peaks (one per leaf blade), min spacing = seed_frac*sqrt(area),
               or user-given [(x,y), ...]
    elevation: grad_w * Lab colour gradient + (1-grad_w) * inverted distance,
               so borders follow colour edges between overlapping leaves and narrow necks."""
    from scipy import ndimage as ndi
    m = comp > 0
    dist = ndi.distance_transform_edt(m)
    if seeds is None:
        md = max(10, int(seed_frac * np.sqrt(m.sum())))
        pk = _peak_local_max(cv2.GaussianBlur(dist, (0, 0), 4), md, 0.35, m)
    else:
        pk = np.array([(y, x) for x, y in seeds if m[y, x]])
    if len(pk) <= 1:
        return [comp]
    lab = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (0, 0), 2), cv2.COLOR_BGR2LAB).astype(np.float32)
    grad = sum(np.hypot(cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0),
                        cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1)) for c in range(3))
    grad = cv2.GaussianBlur(grad, (0, 0), 2)
    grad /= np.percentile(grad[m], 95) + 1e-9
    elev = grad_w * grad + (1 - grad_w) * (1 - dist / dist.max())
    mk = np.zeros(m.shape, np.int32)
    if seeds is not None:
        # user taps: impose a small basin around each tap so a tap on a vein / leaf overlap
        # still owns a region (otherwise neighbouring floods can surround and starve it)
        rr = max(3, int(0.012 * max(m.shape)))
        for i, (y, x) in enumerate(pk, 1):
            disk = np.zeros(m.shape, np.uint8)
            cv2.circle(disk, (int(x), int(y)), rr, 1, -1)
            sel = (disk > 0) & m & (mk == 0)
            mk[sel] = i
            elev[sel] = 0
    else:
        for i, (y, x) in enumerate(pk, 1):
            mk[y, x] = i
    ws = _watershed(elev, mk, m)
    out = []
    for i in range(1, len(pk) + 1):
        leaf = np.where(ws == i, 255, 0).astype(np.uint8)
        # keep the largest piece of each region, fill holes
        n, l, st, _ = cv2.connectedComponentsWithStats(leaf, 8)
        if n > 2:
            big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            leaf = np.where(l == big, 255, 0).astype(np.uint8)
        out.append(fill_holes(leaf))
    return out


# ===================================================================== measurement
def extent(pts, center, v):
    p = (pts - center) @ v
    return p.min(), p.max()


def feret(hull_pts):
    """Max caliper diameter and width perpendicular to it (brute force on hull)."""
    h = hull_pts.astype(np.float64)
    d = ((h[:, None, :] - h[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    u = h[j] - h[i]
    L = np.linalg.norm(u)
    u = u / (L + 1e-12)
    n = np.array([-u[1], u[0]])
    return h[i], h[j], u, n


def count_lobes(contour, center, size, depth_frac):
    """Peaks of the circular radial-distance profile = lobe tips."""
    from scipy.signal import find_peaks
    r = np.linalg.norm(contour - center, axis=1)
    n = len(r)
    win = max(3, n // 100)
    r = np.convolve(np.r_[r[-win:], r, r[:win]], np.ones(win) / win, "same")[win:-win]
    rr = np.r_[r, r, r]                               # wrap so peaks at index 0 are found
    pk, _ = find_peaks(rr, prominence=depth_frac * size, distance=max(5, n // 40))
    pk = pk[(pk >= n) & (pk < 2 * n)] - n
    tips = [tuple(contour[i].astype(int)) for i in pk]
    return max(1, len(pk)), tips


def measure_leaf(leaf, lobe_depth=0.08):
    ys, xs = np.nonzero(leaf)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    c = pts.mean(axis=0)
    ev, evec = np.linalg.eigh(np.cov((pts - c).T))
    o = ev.argsort()[::-1]
    ev, evec = ev[o], evec[:, o]
    v1, v2 = evec[:, 0], evec[:, 1]

    a1, b1 = extent(pts, c, v1)
    a2, b2 = extent(pts, c, v2)

    cnt, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnt, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)[:, 0, :]
    fa, fb, u, n = feret(hull)
    fL = np.linalg.norm(fb - fa) + 1
    a3, b3 = extent(pts, c, n)

    # notches (for drawing) from convexity defects
    hidx = cv2.convexHull(cnt, returnPoints=False)
    notches = []
    try:
        defects = cv2.convexityDefects(cnt, hidx)
    except cv2.error:
        defects = None
    if defects is not None:
        notches = [tuple(cnt[d[2], 0]) for d in defects.reshape(-1, 4)
                   if d[3] / 256.0 > lobe_depth * fL]
    # lobes = tips = peaks of radial distance (centroid -> contour) with enough prominence
    lobes, tips = count_lobes(cnt[:, 0, :].astype(np.float64), c, fL, lobe_depth)
    if not notches:                     # no deep notch -> simple (unlobed) leaf
        lobes, tips = 1, []

    aniso = float(np.sqrt(ev[1] / ev[0]))
    solidity = cv2.countNonZero(leaf) / (cv2.contourArea(hull) + 1e-9)
    return dict(
        center=c, v1=v1, v2=v2, eig1=ev[0], eig2=ev[1],
        pca_len=b1 - a1 + 1, pca_wid=b2 - a2 + 1,
        pca_L=(c + a1 * v1, c + b1 * v1), pca_W=(c + a2 * v2, c + b2 * v2),
        feret_len=fL, feret_wid=b3 - a3 + 1, feret_L=(fa, fb), feret_W=(c + a3 * n, c + b3 * n),
        aniso=aniso, pca_unstable=aniso > 0.8,
        lobes=lobes, notches=notches, tips=tips, solidity=solidity,
        area=len(pts), angle=float(np.degrees(np.arctan2(v1[1], v1[0]))),
        contour=cnt, hull=hull,
    )


# ===================================================================== whole plant (canopy)
def group_plants(mask, gap_frac=0.05):
    """Leaf blobs closer than gap_frac * image diagonal belong to the same plant (pale petioles are often missing from the mask, so the gap must bridge them)."""
    d = max(3, int(gap_frac * np.hypot(*mask.shape)) | 1)
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    n, lab = cv2.connectedComponents(near, connectivity=8)
    lab[mask == 0] = 0
    plants = [np.where(lab == i, 255, 0).astype(np.uint8) for i in range(1, n)]
    return [p for p in plants if cv2.countNonZero(p)], lab


def measure_plant(pmask):
    """Canopy size of one plant: all its leaf pixels (petioles included)."""
    ys, xs = np.nonzero(pmask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    c = pts.mean(0)
    ev, evec = np.linalg.eigh(np.cov((pts - c).T))
    o = ev.argsort()[::-1]
    ev, evec = ev[o], evec[:, o]
    v1, v2 = evec[:, 0], evec[:, 1]
    a1, b1 = extent(pts, c, v1)
    a2, b2 = extent(pts, c, v2)
    hull = cv2.convexHull(pts.astype(np.int32))[:, 0, :]
    fa, fb, u, n = feret(hull)
    a3, b3 = extent(pts, c, n)
    (ex, ey), er = cv2.minEnclosingCircle(hull.reshape(-1, 1, 2))
    hull_area = cv2.contourArea(hull.reshape(-1, 1, 2).astype(np.int32))
    aniso = float(np.sqrt(ev[1] / ev[0]))
    return dict(
        center=c, v1=v1, v2=v2, eig1=ev[0], eig2=ev[1], aniso=aniso, pca_unstable=aniso > 0.8,
        pca_len=b1 - a1 + 1, pca_wid=b2 - a2 + 1,
        pca_L=(c + a1 * v1, c + b1 * v1), pca_W=(c + a2 * v2, c + b2 * v2),
        feret_len=np.linalg.norm(fb - fa) + 1, feret_wid=b3 - a3 + 1,
        feret_L=(fa, fb), feret_W=(c + a3 * n, c + b3 * n),
        circle=((ex, ey), er), circle_d=2 * er,
        hull=hull, hull_area=hull_area, leaf_area=len(pts),
        cover=len(pts) / (hull_area + 1e-9),
        angle=float(np.degrees(np.arctan2(v1[1], v1[0]))),
    )


def draw_plant(img, p, name):
    cv2.polylines(img, [p["hull"].reshape(-1, 1, 2).astype(np.int32)], True, (255, 255, 255), 2)
    (ex, ey), er = p["circle"]
    cv2.circle(img, (int(ex), int(ey)), int(er), (255, 255, 0), 1, cv2.LINE_AA)
    # canopy axes: v1 = yellow, v2 = magenta (dashed look via thick + thin)
    for (A, B), col in ((p["pca_L"], (0, 255, 255)), (p["pca_W"], (255, 0, 255))):
        cv2.line(img, _p(A), _p(B), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, _p(A), _p(B), col, 3, cv2.LINE_AA)
    top = p["hull"][np.argmin(p["hull"][:, 1])]
    _text(img, f"Plant {name}", (int(top[0]) - 40, int(top[1]) - 12), 0.8, (0, 255, 255))


# ===================================================================== drawing
def _p(x):
    return tuple(int(round(v)) for v in x)


def _text(img, t, org, scale=0.6, col=(255, 255, 255)):
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)


def draw(img, m, idx, debug=False):
    cv2.drawContours(img, [m["contour"]], -1, (0, 255, 0), 2)
    cv2.line(img, _p(m["pca_L"][0]), _p(m["pca_L"][1]), (0, 0, 255), 2)    # length  (v1)
    cv2.line(img, _p(m["pca_W"][0]), _p(m["pca_W"][1]), (255, 0, 0), 2)    # width   (v2)
    if debug:
        cv2.line(img, _p(m["feret_L"][0]), _p(m["feret_L"][1]), (255, 0, 255), 1)
        cv2.line(img, _p(m["feret_W"][0]), _p(m["feret_W"][1]), (255, 255, 0), 1)
        for q in m["notches"]:
            cv2.circle(img, _p(q), 5, (0, 165, 255), -1)
        for q in m["tips"]:
            cv2.drawMarker(img, _p(q), (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 12, 2)
    cv2.circle(img, _p(m["center"]), 5, (0, 255, 255), -1)
    _text(img, f"#{idx}", _p(m["center"] + np.array([8, -8])), 0.8)


def draw_table(img, rows, px_per_cm=None, plants=()):
    lines = []
    for name, p in plants:
        L, W = (p["feret_len"], p["feret_wid"]) if p["pca_unstable"] else (p["pca_len"], p["pca_wid"])
        t = f"Plant {name}: L={L:.0f} W={W:.0f}px  D={p['circle_d']:.0f}px"
        if px_per_cm:
            t = (f"Plant {name}: L={L / px_per_cm:.2f} W={W / px_per_cm:.2f}cm  "
                 f"D={p['circle_d'] / px_per_cm:.2f}cm")
        lines.append(t)
        if px_per_cm:
            lines.append(f"   canopy {p['hull_area'] / px_per_cm ** 2:.1f}cm2  "
                         f"leaf {p['leaf_area'] / px_per_cm ** 2:.1f}cm2  ({len(p['leaves'])} leaves)")
        else:
            lines.append(f"   canopy {p['hull_area']:.0f}px2 leaf {p['leaf_area']:.0f}px2 ({len(p['leaves'])} leaves)")
    lines.append("#   L(px)  W(px)" + ("   L(cm)  W(cm)" if px_per_cm else "") + "  note")
    for k, m in enumerate(rows, 1):
        L, W = (m["feret_len"], m["feret_wid"]) if m["pca_unstable"] else (m["pca_len"], m["pca_wid"])
        t = f"{k:<3} {L:6.0f} {W:6.0f}"
        if px_per_cm:
            t += f"  {L / px_per_cm:6.2f} {W / px_per_cm:6.2f}"
        t += "  Feret" if m["pca_unstable"] else ""
        lines.append(t)
    lines.append("leaf: red=L(v1) blue=W(v2)")
    if plants:
        lines.append("plant: yellow=L(v1) magenta=W(v2)")
        lines.append("  white=hull  cyan=enclosing circle D")
    h = 26 * len(lines) + 14
    w = max(330, max(len(t) for t in lines) * 11 + 20)
    ov = img.copy()
    cv2.rectangle(ov, (8, 8), (8 + w, 8 + h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (16, 34 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ===================================================================== main
DEFAULTS = dict(index=None, thresh="valley", min_area=1500, keep_stem=False, lobe_depth=0.08,
                mode="plant", seed_frac=0.12, seeds=None, px_per_cm=None,
                ruler_hsv="5,120,120,22,255,255", pot_size=None, pot_rim_in=None, pot_rim_cm=None,
                debug=False, work_size=1500)


def _primary(d):
    return (d["feret_len"], d["feret_wid"]) if d["pca_unstable"] else (d["pca_len"], d["pca_wid"])


def analyze(img, opts=None, mask=None, debug_base=None):
    """Run the whole pipeline on a BGR image.
    opts: dict, keys as DEFAULTS (px_per_cm may be a number or "auto"; seeds = [(x, y), ...] or
          "x,y;x,y" in ORIGINAL-image pixels).
    Returns dict(log, scale, leaves, plants, vis, mask, color) - pixel values in original-image px."""
    a = dict(DEFAULTS)
    a.update({k: v for k, v in (opts or {}).items() if v is not None})
    log = []
    say = log.append
    dbg = (lambda suffix: debug_base + suffix) if debug_base else (lambda suffix: None)
    scale = dict(px_per_cm=None, source=None)

    # --- scale from ruler (original-image px) -----------------------------------------------
    ppcm_orig, ruler_note = None, ""
    if a["px_per_cm"] == "auto":
        hv = [int(v) for v in str(a["ruler_hsv"]).split(",")]
        ppcm_orig, ruler_note = ruler_scale(img, hv[:3], hv[3:], debug_path=dbg("_ruler.png"))
        say(f"ruler: {ruler_note}")
        scale["ruler"] = dict(px_per_cm=ppcm_orig, note=ruler_note)
        if ppcm_orig:
            scale.update(px_per_cm=ppcm_orig, source="ruler")
    elif a["px_per_cm"]:
        ppcm_orig = float(a["px_per_cm"])
        scale.update(px_per_cm=ppcm_orig, source="manual")

    # --- standard working resolution --------------------------------------------------------
    sc = 1.0
    if a["work_size"] and max(img.shape[:2]) != a["work_size"]:
        sc = a["work_size"] / max(img.shape[:2])
        img = cv2.resize(img, None, fx=sc, fy=sc,
                         interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        say(f"working at {img.shape[1]}x{img.shape[0]} (x{sc:.3f}); pixel results are in original-image px")
    ppcm = ppcm_orig * sc if ppcm_orig else None          # working-resolution px/cm

    # --- scale from pot rim -----------------------------------------------------------------
    rim_in = a["pot_rim_in"] or (a["pot_rim_cm"] / 2.54 if a["pot_rim_cm"] else None)
    if rim_in is None and a["pot_size"]:
        ps = float(a["pot_size"])
        key = int(ps) if ps.is_integer() else ps
        if key in POT_RIM_CM:
            rim_in = POT_RIM_CM[key] / 2.54
            say(f"pot: {ps:g} in pot -> outer rim {POT_RIM_CM[key]:g} cm (POT_RIM_CM table)")
        else:
            rim_in = ps
            say(f"pot: no rim size on file for a {ps:g} in pot -> assuming rim = {ps * 2.54:.1f} cm; "
                f"real rims are often 3-7 % smaller")
    if rim_in:
        D, info = pot_rim(img, debug_path=dbg("_pot.png"))
        if D is None:
            say(f"pot: rim not found ({info})")
            scale["pot"] = dict(found=False, reason=info)
        else:
            ppcm_pot = D / (rim_in * 2.54)
            say(f"pot: rim {D / sc:.0f}px -> {ppcm_pot / sc:.2f} px/cm, confidence {info['confidence']} "
                f"({info['kind']}, inliers {info['inliers'] * 100:.0f}%, tilt ~{info['tilt_deg']:.0f} deg)")
            scale["pot"] = dict(found=True, rim_cm=rim_in * 2.54, rim_px=D / sc,
                                px_per_cm=ppcm_pot / sc, confidence=info["confidence"],
                                kind=info["kind"], tilt_deg=info["tilt_deg"],
                                center=[info["center"][0] / sc, info["center"][1] / sc],
                                axes=[info["axes"][0] / sc, info["axes"][1] / sc])
            if ppcm and "confidence LOW" not in ruler_note:
                diff = (ppcm_pot - ppcm) / ppcm * 100
                scale["pot"]["diff_vs_ruler_pct"] = diff
                say(f"  cross-check: pot vs ruler differ by {diff:+.1f}% -> using the ruler")
            else:
                ppcm = ppcm_pot
                scale.update(px_per_cm=ppcm_pot / sc, source="pot")

    # --- leaf mask --------------------------------------------------------------------------
    color = None
    if mask is None:
        idx = color_indices(img)
        best, stats = choose_index(idx, a["index"])
        t = stats[best][a["thresh"]]
        mask = (idx[best] > t).astype(np.uint8) * 255
        say(f"using '{best}', threshold={t} ({a['thresh']}; otsu={stats[best]['otsu']})")
        color = dict(index=best, threshold=int(t), idx=idx, stats=stats)
    elif mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = clean_mask(mask, a["min_area"])

    seeds = a["seeds"]
    if isinstance(seeds, str):
        seeds = [tuple(float(v) for v in p.split(",")) for p in seeds.split(";") if p.strip()]
    if seeds:
        # user taps are imprecise on a phone: snap each tap to the nearest leaf pixel within
        # 3 % of the image size; taps farther away are ignored
        from scipy import ndimage as ndi
        seeds = [(int(round(x * sc)), int(round(y * sc))) for x, y in seeds]
        seeds = [(x, y) for x, y in seeds if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]]
        dist, (iy, ix) = ndi.distance_transform_edt(mask == 0, return_indices=True)
        inside = ndi.distance_transform_edt(mask > 0)
        r_snap = 0.03 * max(img.shape[:2])
        r_in = max(3, int(0.02 * max(img.shape[:2])))
        H, W = mask.shape
        snapped = []
        for x, y in seeds:
            if dist[y, x] > r_snap:
                continue
            x, y = int(ix[y, x]), int(iy[y, x])            # nearest leaf pixel
            # then move inward: a tap on the leaf edge would be starved by the watershed
            y0, y1, x0, x1 = max(0, y - r_in), min(H, y + r_in + 1), max(0, x - r_in), min(W, x + r_in + 1)
            win = inside[y0:y1, x0:x1]
            dy, dx = np.unravel_index(np.argmax(win), win.shape)
            p = (x0 + int(dx), y0 + int(dy))
            if p not in snapped:
                snapped.append(p)
        say(f"seeds: {len(seeds)} taps -> {len(snapped)} on leaves")
        seeds = snapped

    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    leaves = []
    for i in range(1, n):
        comp = np.where(lab == i, 255, 0).astype(np.uint8)
        comp_seeds = [s for s in seeds if comp[s[1], s[0]]] if seeds else None
        if a["mode"] == "leaf":
            parts = [comp]
        else:
            # fewer than 2 taps on this plant -> automatic split
            parts = split_leaves(img, comp, a["seed_frac"],
                                 seeds=comp_seeds if comp_seeds and len(comp_seeds) >= 2 else None)
        leaves += [p for p in parts if cv2.countNonZero(p) >= a["min_area"]]
    leaves.sort(key=lambda p: tuple(np.argwhere(p)[:, ::-1].mean(0)[::-1] // 150))

    plant_masks, plant_lab = group_plants(mask) if a["mode"] == "plant" else ([], None)
    plants = []
    for j, pm in enumerate(plant_masks):
        p = measure_plant(pm)
        p["leaves"] = []
        plants.append((chr(ord("A") + j) if j < 26 else str(j + 1), p))
    plant_ids = [i for i in range(1, plant_lab.max() + 1) if (plant_lab == i).any()] if plants else []

    vis = img.copy()
    rng = np.random.default_rng(7)
    rows = []
    for k, leaf in enumerate(leaves, 1):
        full = leaf
        if not a["keep_stem"]:
            leaf = remove_stem(leaf)
        m = measure_leaf(leaf, a["lobe_depth"])
        m["plant"] = ""
        if plants:
            cy_, cx_ = np.argwhere(full).mean(0).astype(int)
            lab_id = plant_lab[cy_, cx_] or np.bincount(plant_lab[full > 0]).argmax()
            if lab_id in plant_ids:
                name, p = plants[plant_ids.index(lab_id)]
                m["plant"] = name
                p["leaves"].append(k)
        rows.append(m)
        sel = full > 0
        vis[sel] = (0.6 * vis[sel] + 0.4 * rng.integers(40, 255, 3)).astype(np.uint8)

    def to_orig(d):
        for key in ("pca_len", "pca_wid", "feret_len", "feret_wid", "circle_d"):
            if key in d:
                d[key] /= sc
        for key in ("area", "leaf_area", "hull_area", "eig1", "eig2"):
            if key in d:
                d[key] = d[key] / sc ** 2
        d["center"] = d["center"] / sc

    for k, m in enumerate(rows, 1):
        draw(vis, m, k, a["debug"])
        to_orig(m)
    for name, p in plants:
        draw_plant(vis, p, name)
        to_orig(p)
    u = scale["px_per_cm"]
    draw_table(vis, rows, u, plants)

    def cm(v, sq=False):
        return (v / u ** 2 if sq else v / u) if u else None

    leaf_out = []
    for k, m in enumerate(rows, 1):
        pl, pw = _primary(m)
        leaf_out.append(dict(
            leaf=k, plant=m["plant"], lobes=int(m["lobes"]),
            pca_length_px=m["pca_len"], pca_width_px=m["pca_wid"],
            feret_length_px=m["feret_len"], feret_width_px=m["feret_wid"],
            primary_length_px=pl, primary_width_px=pw, anisotropy=m["aniso"],
            pca_unstable=bool(m["pca_unstable"]), length_cm=cm(pl), width_cm=cm(pw),
            area_px=m["area"], area_cm2=cm(m["area"], True), solidity=m["solidity"],
            angle_deg=m["angle"], cx=m["center"][0], cy=m["center"][1],
            v1x=m["v1"][0], v1y=m["v1"][1], v2x=m["v2"][0], v2y=m["v2"][1],
            eigval1=m["eig1"], eigval2=m["eig2"]))
    plant_out = []
    for name, p in plants:
        pl, pw = _primary(p)
        plant_out.append(dict(
            plant=name, n_leaves=len(p["leaves"]), leaves=p["leaves"],
            pca_length_px=p["pca_len"], pca_width_px=p["pca_wid"],
            feret_length_px=p["feret_len"], feret_width_px=p["feret_wid"],
            primary_length_px=pl, primary_width_px=pw, circle_diameter_px=p["circle_d"],
            hull_area_px2=p["hull_area"], leaf_area_px2=p["leaf_area"], cover=p["cover"],
            anisotropy=p["aniso"], angle_deg=p["angle"], cx=p["center"][0], cy=p["center"][1],
            v1x=p["v1"][0], v1y=p["v1"][1], v2x=p["v2"][0], v2y=p["v2"][1],
            length_cm=cm(pl), width_cm=cm(pw), circle_diameter_cm=cm(p["circle_d"]),
            hull_area_cm2=cm(p["hull_area"], True), leaf_area_cm2=cm(p["leaf_area"], True)))
    for d in leaf_out + plant_out:                       # plain floats for JSON
        for kk, vv in d.items():
            if isinstance(vv, (np.floating, np.integer)):
                d[kk] = vv.item()
    return dict(log=log, scale=scale, leaves=leaf_out, plants=plant_out,
                vis=vis, mask=mask, color=color, work_scale=sc)


LEAF_COLS = ["leaf", "plant", "lobes", "pca_length_px", "pca_width_px", "feret_length_px",
             "feret_width_px", "primary_length_px", "primary_width_px", "anisotropy", "pca_unstable",
             "length_cm", "width_cm", "area_px", "area_cm2", "solidity", "angle_deg", "cx", "cy",
             "v1x", "v1y", "v2x", "v2y", "eigval1", "eigval2"]
PLANT_COLS = ["plant", "n_leaves", "leaves", "pca_length_px", "pca_width_px", "feret_length_px",
              "feret_width_px", "primary_length_px", "primary_width_px", "circle_diameter_px",
              "hull_area_px2", "leaf_area_px2", "cover", "anisotropy", "angle_deg", "cx", "cy",
              "v1x", "v1y", "v2x", "v2y", "length_cm", "width_cm", "circle_diameter_cm",
              "hull_area_cm2", "leaf_area_cm2"]


def to_csv(rows, cols):
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        out = []
        for c in cols:
            v = r.get(c)
            if v is None:
                out.append("")
            elif isinstance(v, bool):
                out.append(v)
            elif isinstance(v, float):
                out.append(f"{v:.4f}" if c.startswith(("v1", "v2")) else f"{v:.2f}")
            elif isinstance(v, list):
                out.append(" ".join(map(str, v)))
            else:
                out.append(v)
        w.writerow(out)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--index", choices=["exg", "exgr", "lab_a", "hsv"], help="force colour index")
    ap.add_argument("--thresh", choices=["valley", "otsu"], default="valley")
    ap.add_argument("--mask", help="binary mask (white = leaf) instead of colour segmentation")
    ap.add_argument("--min-area", type=int, default=1500)
    ap.add_argument("--keep-stem", action="store_true",
                    help="measure petiole too (default: petiole is stripped)")
    ap.add_argument("--lobe-depth", type=float, default=0.08,
                    help="notch depth (fraction of Feret length) counted as a lobe split")
    ap.add_argument("--mode", choices=["plant", "leaf"], default="plant",
                    help="plant: whole plant top view, split blob into leaves (watershed); "
                         "leaf: detached leaves, each blob = one (possibly lobed) leaf")
    ap.add_argument("--seed-frac", type=float, default=0.12,
                    help="min seed spacing = frac*sqrt(plant area); lower -> more leaves")
    ap.add_argument("--seeds", help='manual leaf seeds "x1,y1;x2,y2;..." (one point per leaf)')
    ap.add_argument("--px-per-cm", help="scale: a number, or 'auto' to read it from a ruler in the image")
    ap.add_argument("--ruler-hsv", default="5,120,120,22,255,255",
                    help="HSV range of the ruler's coloured stripe for --px-per-cm auto (default orange)")
    ap.add_argument("--pot-size", type=float,
                    help="nominal pot size in inches (e.g. 8) -> rim diameter via POT_RIM_CM")
    ap.add_argument("--pot-rim-in", type=float, help="actual OUTER rim diameter of the pot in inches")
    ap.add_argument("--pot-rim-cm", type=float, help="actual OUTER rim diameter of the pot in cm")
    ap.add_argument("--debug", action="store_true", help="also draw Feret lines, lobe tips, notches")
    ap.add_argument("--work-size", type=int, default=1500,
                    help="process at this long-side size (px); outputs are in original-image px. 0 = off")
    ap.add_argument("--out")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    base = os.path.splitext(a.image)[0]
    mask = None
    if a.mask:
        mask = (cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
    opts = {k: v for k, v in vars(a).items() if k in DEFAULTS}
    if a.px_per_cm and a.px_per_cm != "auto":
        opts["px_per_cm"] = float(a.px_per_cm)
    r = analyze(img, opts, mask=mask, debug_base=base)
    for line in r["log"]:
        print(line)
    for d in r["leaves"]:
        cmtxt = (f"  -> {d['length_cm']:.2f} x {d['width_cm']:.2f} cm" if d["length_cm"] else "")
        print(f"Leaf {d['leaf']}: L={d['primary_length_px']:.1f}px W={d['primary_width_px']:.1f}px "
              f"lobes={d['lobes']} aniso={d['anisotropy']:.2f}"
              f"{' (PCA unstable -> Feret)' if d['pca_unstable'] else ''}{cmtxt}")
    for p in r["plants"]:
        cmtxt = (f"  -> {p['length_cm']:.2f} x {p['width_cm']:.2f} cm, D {p['circle_diameter_cm']:.2f} cm, "
                 f"leaf area {p['leaf_area_cm2']:.1f} cm2" if p["length_cm"] else "")
        print(f"Plant {p['plant']}: {p['n_leaves']} leaves {p['leaves']}  "
              f"L={p['primary_length_px']:.1f}px W={p['primary_width_px']:.1f}px{cmtxt}")
    out = a.out or base + "_measured.png"
    cv2.imwrite(out, r["vis"])
    cv2.imwrite(base + "_mask.png", r["mask"])
    if r["color"]:
        c = r["color"]
        save_histograms(c["idx"], c["stats"], c["index"], base + "_hist.png")
    with open(base + "_measure.csv", "w", newline="") as f:
        f.write(to_csv(r["leaves"], LEAF_COLS))
    if r["plants"]:
        with open(base + "_plant.csv", "w", newline="") as f:
            f.write(to_csv(r["plants"], PLANT_COLS))
    print(f"saved: {out}, {base}_measure.csv" + (f", {base}_plant.csv" if r["plants"] else ""))


if __name__ == "__main__":
    main()
