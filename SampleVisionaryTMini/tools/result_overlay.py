"""Overlay REAL C++ batch targets (Result_merged) on the cloud. red=CC(hole), yellow=before-offset,
green=after-offset(final). Big colored spheres. Reads the actual saved _igh_pos/_cone_pos/_hole_pos."""
import os, glob, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R
from target_overlay import _sphere, write_colored_pcd, _clean, IGH_MERGE_OFFSET, XT_INSET

RES = r"K:\TUAS\South_Endfeed\GT_TargetCheck\Result_live"
JOBS = [
    ("XT_40_Big_2cone",   "*060334*XT_CHS40*",  "xt",   True),
    ("IGH_40_Small_sep",  "*140350*IGH_CHS40*", "sep",  False),
    ("IGH_40_Small_merged","*193212*IGH_CHS40*", "merged", False),  # Small -> offset -30
]
def pt(d, *p):
    for x in p:
        f = glob.glob(os.path.join(d, x))
        if f:
            a = _clean(R.read_pcd(f[0]))
            if len(a): return a.mean(0)
    return None

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_pcd"); os.makedirs(out, exist_ok=True)
for tag, pat, kind, isBig in JOBS:
    jobs = glob.glob(os.path.join(RES, pat))
    if not jobs: print("no job", pat); continue
    frames = sorted(glob.glob(os.path.join(jobs[0], "*_Image")))
    fr = [d for d in frames if glob.glob(os.path.join(d, "*_igh_pos.pcd")) or glob.glob(os.path.join(d, "*_ighs_pos.pcd")) or glob.glob(os.path.join(d, "*_cone_pos.pcd"))]
    if not fr: print("no frame w/ target for", tag); continue
    d = fr[len(fr)//2]; fn = os.path.basename(d)
    scene = _clean(R.read_pcd(os.path.join(d, fn + "_base_converted.pcd")))
    final = pt(d, "*_igh_pos.pcd", "*_ighs_pos.pcd", "*_cone_pos.pcd")   # real final target (after offset)
    cc    = pt(d, "*_hole_pos.pcd")                        # corner casting (if present)
    raw = None
    if kind == "merged":
        # centre = final - appliedOffset;  applied = +30(Big) / -30(Small)
        raw = final - (IGH_MERGE_OFFSET if isBig else -IGH_MERGE_OFFSET) * np.array([1.0,0,0])
    elif kind == "xt":    raw = final + np.array([XT_INSET, 0, 0])           # extreme (= final + 130 back out)
    # sep: no separate before-offset
    if scene.shape[0] > 60000:
        scene = scene[np.random.RandomState(0).choice(scene.shape[0], 60000, replace=False)]
    m = []
    if cc    is not None: m.append(_sphere(cc,    (255,40,40),  600, 34, 1))   # red   CC
    if raw   is not None: m.append(_sphere(raw,   (255,225,0),  600, 34, 2))   # yellow before-offset
    if final is not None: m.append(_sphere(final, (40,230,40),  600, 34, 3))   # green after-offset
    p = os.path.join(out, tag + "_REAL.pcd"); write_colored_pcd(p, scene, m)
    print("wrote", os.path.basename(p), "| CC", None if cc is None else np.round(cc).astype(int).tolist(),
          "raw", None if raw is None else np.round(raw).astype(int).tolist(),
          "final", None if final is None else np.round(final).astype(int).tolist(), "| frame", fn)
