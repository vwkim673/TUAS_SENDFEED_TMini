"""
target_overlay.py - overlay targeting points on the intensity image for XT / IGH-separated /
IGH-merged samples. Produces one self-contained HTML (base64 jpg + SVG circles).

Colors (per request):  CC (corner casting / twist-lock hole) = RED,
                        1st (raw) target = YELLOW,   offset-applied target = GREEN.

The intensity image is 512x424 = the organized cloud grid, so cloud[i] <-> pixel (i%512, i//512)
(GROUNDTRUTH_GUIDELINE). A 3D target is projected by nearest VALID organized-cloud point -> its index.

NOTE: cloud-based REPLICA of the C++ targeting (approx). The C++ FORCE_ROI batch debug images
(_ROI_target.pcd / _debug_result.jpg) are authoritative. This is for visualising the raw->offset step.
"""
import sys, os, base64, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R          # read_pcd (ordered), dense_extreme

W, H = 512, 424
ORIG = r"K:\TUAS\South_Endfeed\GroundTruth_Set\OriginalData"
IGH_MERGE_OFFSET = 30.0
XT_INSET = 130.0

SAMPLES = [
    dict(kind="XT 40ft center 2-cone (CS)",  chassis="XT",  merged=False, twocone=True,
         job=r"XT\REAR\20260606_162958_298_B3_Mount_Big_XT_CHS40_SPRD20_REAR_0", tor=True, isBig=True),
    dict(kind="IGH separated (CS Big CHS45)",  chassis="IGH", merged=False,
         job=r"IGH\REAR\20260606_211756_721_B2_Mount_Big_IGH_CHS45_SPRD20_REAR_0", tor=True, isBig=True),
    dict(kind="IGH merged (CS/REAR Big CHS40)", chassis="IGH", merged=True,
         job=r"IGH\REAR\20260606_220941_895_B3_Mount_Big_IGH_CHS40_SPRD20_REAR_0",  tor=True,  isBig=True),
]

def dz(v, w=40, m=10, mx=False): return R.dense_extreme(np.asarray(v), w, m, find_max=mx)

def _frame_dirs(job):
    """All <frame>_Image folders holding a _base_converted.pcd (across the varying batch dir names)."""
    jd = os.path.join(ORIG, job)
    par, name = os.path.dirname(jd), os.path.basename(jd)
    for cand in (os.path.join(jd, "*_Image", "*_base_converted.pcd"),
                 os.path.join(par, "batch_result", name, "*_Image", "*_base_converted.pcd"),
                 os.path.join(par, "BatchResult", name, "*_Image", "*_base_converted.pcd")):
        fs = sorted(glob.glob(cand))
        if fs: return [os.path.dirname(p) for p in fs]
    return []

def _clean(a):
    a = a[np.isfinite(a).all(1)]
    return a[~((a[:,0]==0)&(a[:,1]==0)&(a[:,2]==0))]

def _pt(d, *pats):
    for pat in pats:
        f = glob.glob(os.path.join(d, pat))
        if f:
            a = _clean(R.read_pcd(f[0]))
            if len(a): return a.mean(0)
    return None

def _cloud(d, *pats):
    for pat in pats:
        f = glob.glob(os.path.join(d, pat))
        if f:
            a = _clean(R.read_pcd(f[0]))
            if len(a): return a
    return None

def load_saved(job):
    """Pick a frame with C++-saved target + hole. Returns dict of real points/clouds."""
    dirs = _frame_dirs(job)
    # prefer a frame that ALSO has a hole (CC); else any with AB_va_conePos
    with_hole = [d for d in dirs if glob.glob(os.path.join(d,"*_AB_va_conePos.pcd")) and
                 (glob.glob(os.path.join(d,"*_hole_pos.pcd")) or glob.glob(os.path.join(d,"*_pcHole.pcd")))]
    cand = with_hole or [d for d in dirs if glob.glob(os.path.join(d,"*_AB_va_conePos.pcd"))]
    if not cand: return None
    d = cand[len(cand)//2]; fn = os.path.basename(d)
    scene = _clean(R.read_pcd(os.path.join(d, fn + "_base_converted.pcd")))
    return dict(dir=d, scene=scene,
                cone   = _pt(d, "*_AB_va_conePos.pcd"),                 # VA raw target
                body   = _cloud(d, "*_AB_va_body.pcd"),                 # VA cone/guide region
                crop   = _cloud(d, "*_ROI_TRK_roiCrop.pcd", "*_ROI_ACQ_roiCrop.pcd"),  # ROI crop (full)
                roi    = _pt(d, "*_ROI_TRK_conePos.pcd", "*_ROI_ACQ_conePos.pcd"),  # ROI final target
                hole   = _pt(d, "*_hole_pos.pcd", "*_pcHole.pcd"))      # CC / corner casting

def load_grid(job):
    """Return (grid Nx3 in organized order with invalids kept as nan, jpg_path)."""
    jd = os.path.join(ORIG, job)
    par, name = os.path.dirname(jd), os.path.basename(jd)
    frames = []
    for cand in (os.path.join(jd, "*_Image", "*_base_converted.pcd"),
                 os.path.join(par, "batch_result", name, "*_Image", "*_base_converted.pcd"),
                 os.path.join(par, "BatchResult", name, "*_Image", "*_base_converted.pcd")):
        frames = sorted(glob.glob(cand))
        if frames: break
    if not frames: return None, None
    # pick a middle frame (clearer chassis)
    pcd = frames[len(frames)//2]
    fdir = os.path.dirname(pcd)
    fname = os.path.basename(pcd).replace("_base_converted.pcd", "")
    jpg = None
    for cand in (os.path.join(jd, "Image", fname + ".jpg"),          # clean intensity (IGH)
                 os.path.join(fdir, fname + "_debug_result.jpg"),    # annotated (XT frame folder)
                 os.path.join(fdir, fname + ".jpg")):
        if os.path.isfile(cand): jpg = cand; break
    if jpg is None:
        js = glob.glob(os.path.join(jd, "Image", "*.jpg")) or glob.glob(os.path.join(fdir, "*.jpg"))
        jpg = js[len(js)//2] if js else None
    pts = R.read_pcd(pcd)                      # ordered W*H x 3
    if pts.shape[0] < W*H:                      # pad if short
        pad = np.full((W*H - pts.shape[0], 3), np.nan); pts = np.vstack([pts, pad])
    pts = pts[:W*H].astype(float)
    bad = ~np.isfinite(pts).all(1) | ((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))
    pts[bad] = np.nan
    return pts, jpg

def project(grid, p3):
    """nearest valid organized point -> (px,py)."""
    if p3 is None or not np.all(np.isfinite(p3)): return None
    d = np.nansum((grid - p3)**2, axis=1)
    d[~np.isfinite(grid).all(1)] = np.inf
    i = int(np.argmin(d))
    return (i % W, i // W)

def valid(grid):
    m = np.isfinite(grid).all(1)
    return grid[m]

def height_band(pts, ylo=-600, yhi=60):
    return pts[(pts[:,1] >= ylo) & (pts[:,1] <= yhi)]

def guide_band(pts):
    """dominant dense-Y band ~ the guide/cone mass."""
    if len(pts) < 30: return pts
    y = pts[:,1]; gy = dz(y, 150, 30, False)     # near lower dense (guide sits mid)
    # dominant: histogram peak
    hist, edges = np.histogram(y, bins=40)
    cy = 0.5*(edges[np.argmax(hist)] + edges[np.argmax(hist)+1])
    return pts[np.abs(pts[:,1]-cy) <= 140]

def near_z_corner(pts, tor):
    """target_on_right extreme corner: near-Z tips, then min/max-X + corner height."""
    if len(pts) < 20: return None
    zN = dz(pts[:,2], 40, 10, False)
    tips = pts[(pts[:,2] <= zN + 150)]
    if len(tips) < 10: tips = pts
    xext = tips[:,0].min() if tor else tips[:,0].max()      # tor=min-X, !tor=max-X
    col = tips[np.abs(tips[:,0]-xext) <= 60]
    if len(col) < 5: col = tips
    tz = np.percentile(col[:,2], 20)
    ty = np.median(col[:,1])
    return np.array([xext, ty, tz]), xext

def compute(sample, grid):
    pts = valid(grid)
    band = guide_band(height_band(pts))
    if len(band) < 20: band = height_band(pts)
    cc = None; raw = None; off = None
    if sample["merged"]:
        gxmin, gxmax = band[:,0].min(), band[:,0].max()
        cx = 0.5*(gxmin+gxmax)
        raw = np.array([cx, np.median(band[np.abs(band[:,0]-cx)<=60][:,1]) if (np.abs(band[:,0]-cx)<=60).any() else band[:,1].mean(),
                        np.percentile(band[np.abs(band[:,0]-cx)<=60][:,2],20) if (np.abs(band[:,0]-cx)<=60).any() else np.percentile(band[:,2],20)])
        tx = cx + (IGH_MERGE_OFFSET if sample["isBig"] else -IGH_MERGE_OFFSET)
        col = band[np.abs(band[:,0]-tx)<=60]
        if len(col) >= 5:
            off = np.array([tx, np.median(col[:,1]), np.percentile(col[:,2],20)])
        else:
            off = np.array([tx, raw[1], raw[2]])
    else:
        r = near_z_corner(band, sample["tor"])
        if r:
            raw, xext = r
            if sample["chassis"] == "XT":
                tx = xext - XT_INSET if not sample["tor"] else xext + XT_INSET   # inset inward
                col = band[np.abs(band[:,0]-tx)<=60]
                off = np.array([tx, np.median(col[:,1]) if len(col) else raw[1],
                                np.percentile(col[:,2],20) if len(col) else raw[2]])
            else:
                off = raw.copy()     # separated IGH: corner detector IS the target (no extra inset)
    # CC (corner casting / twist-lock hole): nearest-Z protrusion in the UPPER band (above the guide)
    up = pts[(pts[:,1] >= (band[:,1].max() if len(band) else 0) - 40)]
    if len(up) >= 20:
        zc = dz(up[:,2], 40, 10, False)
        castle = up[up[:,2] <= zc + 80]
        if len(castle) >= 10:
            cc = np.array([np.median(castle[:,0]), np.median(castle[:,1]), np.median(castle[:,2])])
    return cc, raw, off

import struct
def _packrgb(r, g, b):
    i = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack('f', struct.pack('I', i))[0]

def _sphere(center, color, n=400, rad=28.0, seed=0):
    """Dense colored blob so the marker is BIG/visible at any viewer point size."""
    rng = np.random.RandomState(seed)
    p = rng.normal(size=(n, 3)); p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    p *= rng.uniform(0, rad, size=(n, 1)) ** (1/3) * rad ** (2/3)   # fill volume
    p = p + np.asarray(center, float)
    rgb = np.full((n, 1), _packrgb(*color))
    return np.hstack([p, rgb])

def write_colored_pcd(path, scene, markers):
    """scene = Nx3 (gray); markers = list of (Nx4 xyz+rgb). ascii XYZRGB PCD (CloudCompare-ready)."""
    gray = np.full((scene.shape[0], 1), _packrgb(170, 170, 170))
    sc = np.hstack([scene, gray])
    allp = np.vstack([sc] + markers)
    n = allp.shape[0]
    hdr = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\n"
           "COUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n" % (n, n))
    with open(path, 'w') as f:
        f.write(hdr)
        for r in allp:
            f.write("%.1f %.1f %.1f %.9e\n" % (r[0], r[1], r[2], r[3]))

def main_pcd():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_pcd")
    os.makedirs(outdir, exist_ok=True)
    def col(band, tx, win=45):
        m = np.abs(band[:,0]-tx) <= win
        c = band[m] if m.sum() >= 5 else band
        ty = dz(c[:,1], 40, 8, True)          # highest dense Y = guide/cone top
        tz = float(np.percentile(c[:,2], 10)) # closest Z (near-Z)
        return ty, tz
    written = []
    for s in SAMPLES:
        D = load_saved(s["job"])
        if D is None or D["scene"] is None:
            print("NO SAVED TARGET", s["job"]); continue
        region = D["crop"] if D["crop"] is not None else D["body"]
        raw = off = None
        if region is not None and len(region) >= 30:
            band = guide_band(height_band(region)) if s["chassis"] == "IGH" else height_band(region)
            if len(band) < 20: band = region
            tips = band[band[:,2] <= dz(band[:,2], 40, 10, False) + 150]
            if len(tips) < 15: tips = band
            xmin, xmax = float(tips[:,0].min()), float(tips[:,0].max())
            if s["merged"]:
                cx = 0.5*(xmin + xmax)                                   # width/2 midpoint = centre
                ty, tz = col(band, cx); raw = np.array([cx, ty, tz])
                tx = cx + (IGH_MERGE_OFFSET if s["isBig"] else -IGH_MERGE_OFFSET)
                ty2, tz2 = col(band, tx); off = np.array([tx, ty2, tz2])
            elif s.get("twocone"):                                       # XT 2-cone: end -> inset inward
                xend = xmax if s["isBig"] else xmin                      # targetMaxX = (jobPos==Big)
                ty, tz = col(band, xend); raw = np.array([xend, ty, tz])
                tx = xend - XT_INSET if s["isBig"] else xend + XT_INSET
                ty2, tz2 = col(band, tx); off = np.array([tx, ty2, tz2])
            else:                                                        # IGH separated: end corner only
                xend = xmin if s["tor"] else xmax                        # tor=min-X
                ty, tz = col(band, xend); raw = np.array([xend, ty, tz])
        if raw is None: raw = D["cone"]                                  # fallback: VA target
        cc = D["hole"]           # red: corner casting (real C++ hole detection)
        scene = D["scene"]
        if scene.shape[0] > 60000:
            idx = np.random.RandomState(0).choice(scene.shape[0], 60000, replace=False); scene = scene[idx]
        markers = []
        if cc  is not None: markers.append(_sphere(cc,  (255, 40, 40), 600, 34, 1))   # RED    = CC (hole)
        if raw is not None: markers.append(_sphere(raw, (255, 225, 0), 600, 34, 2))   # YELLOW = raw (before offset)
        if off is not None: markers.append(_sphere(off, (40, 230, 40), 600, 34, 3))   # GREEN  = after offset
        tag = s["kind"].split("(")[0].strip().replace(" ", "_").replace("/", "")
        p = os.path.join(outdir, tag + ".pcd")
        write_colored_pcd(p, scene, markers)
        written.append(p)
        print("wrote", os.path.basename(p), "| CC", None if cc is None else np.round(cc).astype(int).tolist(),
              "raw", None if raw is None else np.round(raw).astype(int).tolist(),
              "off", None if off is None else np.round(off).astype(int).tolist(),
              "| frame", os.path.basename(D["dir"]))
    return written

def main():
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_overlay.html")
    panels = []
    for s in SAMPLES:
        grid, jpg = load_grid(s["job"])
        if grid is None or jpg is None:
            panels.append(f"<div><h3>{s['kind']}</h3><p>NO DATA ({s['job']})</p></div>"); continue
        cc, raw, off = compute(s, grid)
        pcc, praw, poff = project(grid, cc), project(grid, raw), project(grid, off)
        b64 = base64.b64encode(open(jpg,'rb').read()).decode()
        def circ(p, color, lab):
            if p is None: return ""
            return (f'<circle cx="{p[0]}" cy="{p[1]}" r="7" fill="none" stroke="{color}" stroke-width="2.5"/>'
                    f'<line x1="{p[0]-11}" y1="{p[1]}" x2="{p[0]+11}" y2="{p[1]}" stroke="{color}" stroke-width="1"/>'
                    f'<line x1="{p[0]}" y1="{p[1]-11}" x2="{p[0]}" y2="{p[1]+11}" stroke="{color}" stroke-width="1"/>'
                    f'<text x="{p[0]+10}" y="{p[1]-9}" fill="{color}" font-size="12" font-family="monospace">{lab}</text>')
        svg = (f'<svg viewBox="0 0 {W} {H}" style="position:absolute;left:0;top:0;width:{W}px;height:{H}px">'
               + circ(pcc, "#ff3030", "CC") + circ(praw, "#ffe000", "raw") + circ(poff, "#20e020", "offset") + '</svg>')
        info = f"CC={None if cc is None else np.round(cc).astype(int).tolist()} raw={None if raw is None else np.round(raw).astype(int).tolist()} offset={None if off is None else np.round(off).astype(int).tolist()}"
        panels.append(
            f'<div style="margin:14px">'
            f'<h3 style="font-family:monospace">{s["kind"]}</h3>'
            f'<div style="position:relative;width:{W}px;height:{H}px;border:1px solid #444">'
            f'<img src="data:image/jpeg;base64,{b64}" width="{W}" height="{H}"/>{svg}</div>'
            f'<pre style="font-size:11px">{os.path.basename(s["job"])}\n{info}</pre></div>')
    html = ('<html><body style="background:#111;color:#ddd;font-family:sans-serif">'
            '<h2>Target overlay &mdash; CC(red) / raw target(yellow) / offset target(green)</h2>'
            '<p style="font-size:12px;color:#999">cloud-based replica (approx); C++ FORCE_ROI debug image is authoritative</p>'
            + '<div style="display:flex;flex-wrap:wrap">' + ''.join(panels) + '</div></body></html>')
    open(dst, 'w', encoding='utf-8').write(html)
    print("wrote", dst)

if __name__ == "__main__":
    main_pcd()
