"""Draw the ROI-extracted region (cyan) + VA body (orange) + target on the cloud, for the 3 CS/40ft
samples. Region/body from FORCE_ROI batch (Result_conly); LIVE target from Result_live."""
import os, glob, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R
from target_overlay import _sphere, write_colored_pcd, _clean, _packrgb, IGH_MERGE_OFFSET, XT_INSET

RC = r"K:\TUAS\South_Endfeed\GT_TargetCheck\Result_conly"   # region/body/crop
RL = r"K:\TUAS\South_Endfeed\GT_TargetCheck\Result_live"    # live target
S = [  # (tag, ts, kind, isBig)
    ("XT_40_Big_2cone",    "060334", "xt",     True),
    ("IGH_40_Small_sep",   "140350", "sep",    False),
    ("IGH_40_Small_merged","193212", "merged", False),
]
def fdir(root, ts):
    d = glob.glob(os.path.join(root, "*"+ts+"*", "*_Image"))
    return d[len(d)//2] if d else None
def cloud(d, *pats):
    for p in pats:
        f = glob.glob(os.path.join(d, "*"+p))
        if f:
            a = _clean(R.read_pcd(f[0]))
            if len(a): return a
    return None
def colcloud(pts, rgb):
    return np.hstack([pts, np.full((len(pts),1), _packrgb(*rgb))])

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_pcd"); os.makedirs(out, exist_ok=True)
for tag, ts, kind, isBig in S:
    dc, dl = fdir(RC, ts), fdir(RL, ts)
    if dc is None: print("no conly", ts); continue
    fn = os.path.basename(dc)
    scene = _clean(R.read_pcd(os.path.join(dc, fn+"_base_converted.pcd")))
    if scene.shape[0] > 55000: scene = scene[np.random.RandomState(0).choice(scene.shape[0],55000,False)]
    region = cloud(dc, "_ROI_ACQ_coneRegion.pcd", "_Region.pcd")     # ROI torn region (cyan)
    body   = cloud(dc, "_AB_va_body.pcd")                            # VA box body (orange)
    # LIVE final target
    final = None
    if dl:
        f = glob.glob(os.path.join(dl, "*_igh_pos.pcd")) or glob.glob(os.path.join(dl,"*_ighs_pos.pcd")) or glob.glob(os.path.join(dl,"*_cone_pos.pcd"))
        if f:
            a=_clean(R.read_pcd(f[0])); final=a.mean(0) if len(a) else None
    raw = None
    if final is not None:
        if kind=="merged": raw = final - (IGH_MERGE_OFFSET if isBig else -IGH_MERGE_OFFSET)*np.array([1.,0,0])
        elif kind=="xt":   raw = final + np.array([XT_INSET,0,0])
    # assemble: gray scene + orange (VA-only) + cyan (ROI-only) + MAGENTA (overlap = in BOTH)
    parts = [np.hstack([scene, np.full((len(scene),1), _packrgb(150,150,150))])]
    def keyset(a): return set(map(tuple, np.round(a,0).astype(np.int64)))
    if body is not None and region is not None:
        rk, bk = keyset(region), keyset(body)
        bmask = np.array([tuple(p) in rk for p in np.round(body,0).astype(np.int64)])
        rmask = np.array([tuple(p) in bk for p in np.round(region,0).astype(np.int64)])
        if (~bmask).any(): parts.append(colcloud(body[~bmask],   (255,150,40)))  # orange = VA-only
        if (~rmask).any(): parts.append(colcloud(region[~rmask], (0,220,230)))    # cyan   = ROI-only
        ov = body[bmask]
        if len(ov):        parts.append(colcloud(ov,             (255,0,255)))    # MAGENTA = overlap (both)
        print("   overlap pts:", int(bmask.sum()), "| VA-only:", int((~bmask).sum()), "| ROI-only:", int((~rmask).sum()))
    elif body is not None:   parts.append(colcloud(body,   (255,150,40)))
    elif region is not None: parts.append(colcloud(region, (0,220,230)))
    markers = []
    if raw   is not None: markers.append(_sphere(raw,   (255,225,0), 700, 40, 2))
    if final is not None: markers.append(_sphere(final, (40,230,40), 700, 40, 3))
    allp = np.vstack(parts + markers)
    p = os.path.join(out, tag+"_REGION.pcd")
    n=len(allp); hdr=("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n"%(n,n))
    with open(p,"w") as fo:
        fo.write(hdr)
        for r in allp: fo.write("%.1f %.1f %.1f %.9e\n"%(r[0],r[1],r[2],r[3]))
    print("wrote", os.path.basename(p), "| region", None if region is None else "%dpts span%d"%(len(region),region[:,0].max()-region[:,0].min()),
          "| body", None if body is None else "span%d"%(body[:,0].max()-body[:,0].min()),
          "| raw", None if raw is None else np.round(raw).astype(int).tolist(), "final", None if final is None else np.round(final).astype(int).tolist())
