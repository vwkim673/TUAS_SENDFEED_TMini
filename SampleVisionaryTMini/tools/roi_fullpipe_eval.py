"""
roi_fullpipe_eval.py - Full ROI pipeline (crop -> detect) on base_converted.pcd,
scored against human GT. Lets us iterate presets + extraction with no C++ rebuild.

Per GT (job,frame,state in CONLY/LANDING):
  1. read OriginalData/XT/<job>/<frame>/<frame>_base_converted.pcd  (organized 512x424)
  2. resolve preset key (cpp logic) -> preset center (Tc,Gc) from PRESETS table
  3. crop world ROI: T in [Tc+-halfT], G in [Gc+-halfG], H in [Hmin,Hmax], drop (0,0,0)
  4. detect cone: dense nearest-Z -> thin slab [zNear-zPad, zNear+zSlab]
       - CENTROID = slab mean (current ChassisRoi.h behaviour)
       - CORNER   = directional dense T-edge (+T unless target_on_right) ; H at edge ; G centroid
  5. error vs GT for both, aggregated by state and config.

Usage: python roi_fullpipe_eval.py [--half-t 150] [--half-g 150] [--csv out.csv]
"""
import sys, os, struct, argparse, re
import numpy as np

GT_ROOT  = r"K:\TUAS\South_Endfeed\GroundTruth_Set\GroundTruth\XT"
ORIG_ROOT= r"K:\TUAS\South_Endfeed\GroundTruth_Set\OriginalData\XT"
# Dense, world-converted clouds (the ones the GT was annotated on) live in two places:
#   old jobs: <ORIG_ROOT>/<job>/<frame>/<frame>_base_converted.pcd          (exact GT frame)
#   new jobs: <ORIG_ROOT>/<spos>/BatchResult/<job>/<bframe>/<bframe>_base_converted.pcd
#             -- batch saves only ~1 frame in 10, so the exact GT frame is usually absent;
#                CONLY chassis is stationary, so the NEAREST-timestamp batch frame is the
#                same scene (validated: GT body sits on it ~99-100% @25mm, vs ~22% for the
#                raw Depth/*.pcd which is a DIFFERENT/sparse cloud -> never use that).
BATCH_ROOTS = {
    'FRONT': r"K:\TUAS\South_Endfeed\GroundTruth_Set\OriginalData\XT\FRONT\BatchResult",
    'REAR' : r"K:\TUAS\South_Endfeed\GroundTruth_Set\OriginalData\XT\REAR\BatchResult",
}

def _frame_ms(name):
    m = re.match(r'\d{8}_(\d{2})(\d{2})(\d{2})_(\d{3})', name)
    if not m: return None
    hh,mm,ss,ml = map(int, m.groups())
    return ((hh*60+mm)*60+ss)*1000+ml

def resolve_pcd(job, frame):
    spos = 'FRONT' if '_FRONT_' in job else ('REAR' if '_REAR_' in job else None)
    # 1. exact GT frame, dense: OriginalData/XT/<spos>/<job>/<frame>/..._base_converted.pcd
    for base in ((os.path.join(ORIG_ROOT, spos),) if spos else ()) + (ORIG_ROOT,):
        p = os.path.join(base, job, frame, frame+'_base_converted.pcd')
        if os.path.isfile(p): return p
    jd = os.path.join(BATCH_ROOTS[spos], job) if spos in BATCH_ROOTS else None
    if jd and os.path.isdir(jd):                                           # 2. nearest batch frame
        t0 = _frame_ms(frame); best=None; bd=1e18
        if t0 is not None:
            for d in os.listdir(jd):
                if not d.endswith('_Image'): continue
                t = _frame_ms(d)
                if t is not None and abs(t-t0) < bd: bd = abs(t-t0); best = d
        if best is not None:
            p = os.path.join(jd, best, best+'_base_converted.pcd')
            if os.path.isfile(p): return p
    return None

def load_cloud(job, frame):
    """Resolve + read a frame's cloud, dropping NaN/inf and (0,0,0) invalid pixels.
    Returns Nx3 float array, or None if the cloud can't be found."""
    p = resolve_pcd(job, frame)
    if p is None: return None
    pts = read_pcd(p)
    pts = pts[np.isfinite(pts).all(1)]
    return pts[~((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))]

HEIGHT_MIN, HEIGHT_MAX = -600.0, 50.0    # global cone-height band (process.ini)
# NOTE: a -1100 floor was tried 2026-06-29 to capture low CHS20/RS_Center cones, but it
# admits low rails/wheel that hijack the cone_columns protrusion (tor=rightmost then picks
# the spurious low structure). CHS20 needs a per-config low band instead -- see CHS20_HMIN.
CHS20_HMIN = -850.0    # CHS20 chassis is low; widen the floor ONLY for CHS20 configs.
# NOTE: ceiling lowered 200->50 (2026-06): the +200 ceiling admitted high-H
# structures that hijacked the nearest-Z anchor (e.g. Center CHS40 FS missed the
# cone, anchoring on a H~+99 structure). Cones sit at H<=-100; 50 keeps margin.
ZSLAB = 450.0      # depth slab from near edge inward: [zNear-zPad, zNear+ZSLAB].
# Trolley(Z) is the depth axis; the minZ face is the plate's leading edge toward
# the sensor and the plate extends INWARD (deeper Z). REAR views are oblique so the
# plate is deep (~460mm) vs FRONT (~230mm) -> 450 covers both. The trolley crop is
# asymmetric (generous inner side) so the slab is never clipped (see preset_crop).
HALF = 150.0       # +-150 position tolerance on Gantry(X) and the near Trolley side.
CONE_BACK = 70.0   # protrusion window: extent away from the edge side
CONE_FWD  = 130.0  # protrusion window: extent toward the edge side (partner ~210mm out)
PROTR_RADIUS = 120.0  # anchor the protrusion within +-this of preset center Tc

# Per-JOB-POSITION preset centers = GT target-point mean (Gantry/X col0, Trolley/Z col2).
# Keyed by actual config (sensor_jobPos_CHS) -- NOT the legacy lumped F_4XF/R_4XF keys,
# which merged different job positions into one box and mis-centred the crop.
# The ROI box is generous (center +- object/2 + position-tolerance): the target POINT is
# within +-150 (stopper-bounded) and the plate object is ~300x300, so we crop wide and
# let the protrusion/slab refinement isolate the cone.
# (Gantry/X center, Trolley/Z center, Height/Y center) per config = GT point means.
# Y center is used in Stage-2 to keep the cone-height layer and EXCLUDE the wheel
# (CHS20 chassis is low; the wheel is nearer-Z than the cone, ~300-480mm below it).
PRESETS = {
    # cg=Gantry(col0), ct=Trolley(col2), cy=cone-height(col1). Values are the GT
    # target-POINT mean (NOT body centroid: a body-centroid cy sits too low and pulls the
    # cone_columns candidate band onto the plate/rail; the target-point cy keeps it on the
    # protruding cone). Crop is generous (col0 +-300) to absorb +-150 position tolerance.
    'CS_Big_CHS20'   : ( 832.0, 1794.0, -422.0),
    'CS_Big_CHS40'   : ( 852.0, 1828.0, -265.0),
    'CS_Big_CHS45'   : ( 865.0, 1896.0, -232.0),   # NEW 2026-06-29 (1 job, GT T/G/H mean)
    'CS_Small_CHS40' : ( 636.0, 1879.0, -205.0),
    'FS_Big_CHS20'   : ( 718.0, 1672.0, -163.0),
    'FS_Big_CHS40'   : ( 815.0, 1746.0, -112.0),
    'FS_Big_CHS45'   : ( 849.0, 1665.0, -120.0),   # NEW 2026-06-29 (1 job, GT T/G/H mean)
    'FS_Center_CHS40': ( 855.0, 1808.0, -188.0),
    'FS_Center_CHS45': (-758.0, 1729.0, -148.0),   # NEW 2026-07 (3 jobs T-673..-830; job
                                                   # 150426_451 B1 targets the OPPOSITE cone
                                                   # (T+193) -- excluded, preset can't cover it
    'RS_Center_CHS40': (-745.0, 1816.0, -320.0),
    'RS_Center_CHS45': ( 851.0, 1811.0, -323.0),   # NEW 2026-07 (5 jobs T+759..+910; same
                                                   # 150426_451 outlier at T-656 excluded)
    'RS_Small_CHS40' : (-670.0, 1769.0, -324.0),
}
# Generous ROI half-extents (mm): Gantry +-GANTRY_HALF; Trolley near -TROLLEY_BACK,
# inner +TROLLEY_FWD (plate extends inward, deeper for oblique REAR).
GANTRY_HALF  = 300.0
TROLLEY_BACK = 250.0
TROLLEY_FWD  = 450.0

def config_key(sensor, jp, chs):
    return f'{sensor}_{jp}_CHS{chs}'

def preset_for(job, frame):
    """Return (centerGantry_col0, centerTrolley_col2) for a frame, or None."""
    jp, chs, sprd, spos = parse_job(job)
    return PRESETS.get(config_key(frame_sensor(frame), jp, chs))

# ---------------------------------------------------------------- PCD reader
def lzf_decompress(data, expected_len):
    out = bytearray(); i, n = 0, len(data)
    while i < n:
        ctrl = data[i]; i += 1
        if ctrl < 32:
            length = ctrl + 1; out += data[i:i+length]; i += length
        else:
            length = ctrl >> 5
            ref = len(out) - ((ctrl & 0x1f) << 8) - 1
            if length == 7: length += data[i]; i += 1
            ref -= data[i]; i += 1
            for _ in range(length + 2): out.append(out[ref]); ref += 1
    return bytes(out[:expected_len])

def read_pcd(path):
    with open(path, 'rb') as f: raw = f.read()
    hdr_end = raw.index(b'DATA '); line_end = raw.index(b'\n', hdr_end)
    header = raw[:line_end].decode('ascii', 'replace')
    fields, sizes, counts = [], [], []
    npoints = 0; datafmt = 'ascii'; width = height = 0
    for ln in header.splitlines():
        p = ln.split()
        if not p: continue
        if p[0]=='FIELDS': fields = p[1:]
        elif p[0]=='SIZE': sizes = list(map(int,p[1:]))
        elif p[0]=='COUNT': counts = list(map(int,p[1:]))
        elif p[0]=='POINTS': npoints = int(p[1])
        elif p[0]=='WIDTH': width = int(p[1])
        elif p[0]=='HEIGHT': height = int(p[1])
        elif p[0]=='DATA': datafmt = p[1]
    if npoints == 0: npoints = width*height
    if not counts: counts = [1]*len(fields)
    body = raw[line_end+1:]
    stride = sum(s*c for s, c in zip(sizes, counts))
    offs = {}; acc = 0
    for fn, sz, ct in zip(fields, sizes, counts):
        offs[fn] = acc; acc += sz*ct
    if datafmt == 'binary_compressed':
        comp, uncomp = struct.unpack('<II', body[:8])
        dec = lzf_decompress(body[8:8+comp], uncomp)
        arrs = {}; off = 0
        for fn, sz, ct in zip(fields, sizes, counts):
            blocklen = npoints*ct*sz
            arrs[fn] = np.frombuffer(dec[off:off+blocklen], dtype='<f4').copy()
            off += blocklen
        return np.stack([arrs['x'], arrs['y'], arrs['z']], axis=1)
    elif datafmt == 'binary':
        buf = np.frombuffer(body[:npoints*stride], dtype=np.uint8).reshape(npoints, stride)
        def col(fn):
            o = offs[fn]; return buf[:, o:o+4].copy().view('<f4').ravel()
        return np.stack([col('x'), col('y'), col('z')], axis=1)
    else:
        pts=[]
        for ln in body.decode('ascii','replace').splitlines():
            p=ln.split()
            if len(p)>=3: pts.append([float(p[0]),float(p[1]),float(p[2])])
        return np.array(pts)

def preset_crop(pts, c0_gantry, c2_trolley):
    """Generous preset ROI. Axes: col0=Gantry(X), col1=Height(Y), col2=Trolley(Z,depth).
    Covers target-position variation (+-150) PLUS the plate object (~300x300):
    Gantry +-GANTRY_HALF; Height band; Trolley near -TROLLEY_BACK, inner +TROLLEY_FWD
    (plate extends inward, deeper for oblique REAR). Wide on purpose -- the slab/protrusion
    step isolates the cone; excess data is filtered there."""
    m = ((pts[:,0] >= c0_gantry-GANTRY_HALF) & (pts[:,0] <= c0_gantry+GANTRY_HALF) &
         (pts[:,1] >= HEIGHT_MIN) & (pts[:,1] <= HEIGHT_MAX) &
         (pts[:,2] >= c2_trolley-TROLLEY_BACK) & (pts[:,2] <= c2_trolley+TROLLEY_FWD))
    return pts[m]

def write_pcd(path, pts):
    """Minimal ASCII XYZ PCD writer (numpy-only)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    n = len(pts)
    hdr = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
           "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
           f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n")
    with open(path, 'w') as f:
        f.write(hdr)
        for p in pts:
            f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")

# ---------------------------------------------------------------- config/key
def parse_job(job):
    parts = job.split('_')
    jp = next((p for p in parts if p in ("Center","Big","Small")), "?")
    chs = next((int(p[3:]) for p in parts if p.startswith("CHS")), 0)
    sprd= next((int(p[4:]) for p in parts if p.startswith("SPRD")), 0)
    spos = "FRONT" if "_FRONT_" in job else ("REAR" if "_REAR_" in job else "?")
    return jp, chs, sprd, spos

def frame_sensor(frame):
    return 'FS' if '_FS' in frame else ('CS' if '_CS' in frame else ('RS' if '_RS' in frame else '??'))

def resolve_key(spos, chs, sprd, jp):
    if spos == "FRONT":
        if chs==45 and sprd==40: k="4XR"
        elif chs==45 and jp=="Small": k="20R"
        elif chs==40 and jp=="Small": k="20F"
        else: k="4XF"
        return "F_XT_"+k
    else:
        if jp=="Big": k="20F"
        elif chs==40: k="4XF"
        else: k="4XR"
        return "R_XT_"+k

def target_on_right(sensor, jp, chs):
    # cpp: cam_pos==Center && (jobPos==Big || chassisLen==20); FS->Front, CS->Center, RS->Rear
    cam = {'FS':'Front','CS':'Center','RS':'Rear'}.get(sensor,'Front')
    return cam=="Center" and (jp=="Big" or chs==20)

# ---------------------------------------------------------------- detector
def dense_extreme(vals, window=40.0, min_pts=30, find_max=False):
    v = np.sort(vals)
    if len(v) < min_pts: return float(v[-1] if find_max else v[0])
    if find_max:
        hi = len(v)-1; lo = hi
        while hi >= 0:
            while lo>0 and v[hi]-v[lo-1] <= window: lo-=1
            if hi-lo+1 >= min_pts: return float(v[lo:hi+1].mean())
            hi-=1
            if lo>hi: lo=hi
        return float(v[-1])
    lo=0; hi=0; nn=len(v)
    while lo<nn:
        while hi<nn-1 and v[hi+1]-v[lo] <= window: hi+=1
        if hi-lo+1 >= min_pts: return float(v[lo:hi+1].mean())
        lo+=1
        if hi<lo: hi=lo
    return float(v[0])

def _corner_of(cluster, tor, minPts=30, edgeTol=50.0):
    find_max = not tor   # +T edge unless target_on_right
    tEdge = dense_extreme(cluster[:,0], 40.0, max(8,minPts//3), find_max=find_max)
    band = cluster[cluster[:,0] >= tEdge-edgeTol] if find_max else cluster[cluster[:,0] <= tEdge+edgeTol]
    if len(band)==0: band = cluster
    return np.array([band[:,0].mean(), band[:,1].mean(), cluster[:,2].mean()])

def t_clusters(slab, gap=70.0, minPts=20):
    """Split slab into 1-D trolley clusters by gaps; return list of arrays."""
    order = np.argsort(slab[:,0]); s = slab[order]
    cuts = np.where(np.diff(s[:,0]) > gap)[0]
    groups = np.split(s, cuts+1)
    return [g for g in groups if len(g) >= minPts]

def t_of_protrusion(slab, binw=25.0, minbin=8, Tc=None, radius=None):
    """T-center of the slab's nearest-Z protrusion (scan dense-min Z per T-bin).
    If Tc/radius given, only bins with |t-Tc|<=radius are eligible (the target
    is within the lane-border tolerance of the preset center; the partner cone
    ~210mm away is thus excluded as an anchor)."""
    tmin, tmax = slab[:,0].min(), slab[:,0].max()
    best_t, best_z = None, 1e9
    t = tmin
    while t < tmax:
        c = t + binw/2
        if radius is None or Tc is None or abs(c-Tc) <= radius:
            b = slab[(slab[:,0] >= t) & (slab[:,0] < t+binw)]
            if len(b) >= minbin:
                z = dense_extreme(b[:,2], 40.0, min(len(b), minbin), find_max=False)
                if z < best_z: best_z, best_t = z, c
        t += binw
    return best_t

def detect(pts, tor, Tc=None, zSlab=None, zPad=30.0, minPts=30, edgeTol=50.0):
    """Returns centroid/corner (whole-slab) plus clustered corners by rule."""
    if zSlab is None: zSlab = ZSLAB
    if len(pts) < minPts: return None
    zNear = dense_extreme(pts[:,2], 40.0, minPts, find_max=False)
    slab = pts[(pts[:,2] >= zNear-zPad) & (pts[:,2] <= zNear+zSlab)]
    if len(slab) < minPts: return None
    centroid = slab.mean(axis=0)
    corner   = _corner_of(slab, tor, minPts, edgeTol)            # no clustering
    out = dict(centroid=centroid, corner=corner, nslab=len(slab))
    # protrusion-anchored: isolate the cone bump (local nearest-Z), then edge.
    # asymmetric window: extend toward the edge side (partner is ~210mm beyond).
    tp = t_of_protrusion(slab, Tc=Tc, radius=PROTR_RADIUS)
    if tp is not None:
        back, fwd = CONE_BACK, CONE_FWD   # back = away from edge, fwd = toward edge
        if not tor:   # +T edge
            lo, hi = tp-back, tp+fwd
        else:         # -T edge
            lo, hi = tp-fwd, tp+back
        bump = slab[(slab[:,0] >= lo) & (slab[:,0] <= hi)]
        if len(bump) >= max(10, minPts//3):
            out['bump'] = bump                      # ROI region: full depth (the plate)
            # corner POINT from the near-edge sub-slab only (leading ~100mm), so
            # H/G stay at the plate's landing edge, not the deep centroid.
            bz = dense_extreme(bump[:,2], 40.0, max(8,minPts//3), find_max=False)
            near = bump[bump[:,2] <= bz + 100.0]
            out['cor_protr'] = _corner_of(near if len(near) >= max(10,minPts//3) else bump,
                                          tor, minPts, edgeTol)
    out.setdefault('cor_protr', corner)
    out['slab'] = slab
    # clustered selection
    cl = t_clusters(slab)
    if cl:
        # rule NEARZ: cluster with the smallest dense-min Z (most protruding)
        nz = min(cl, key=lambda g: dense_extreme(g[:,2],40.0,max(8,minPts//3),False))
        out['cor_nearz'] = _corner_of(nz, tor, minPts, edgeTol)
        # rule NEARTC: cluster whose mean-T is closest to preset center
        if Tc is not None:
            nc = min(cl, key=lambda g: abs(g[:,0].mean()-Tc))
            out['cor_neartc'] = _corner_of(nc, tor, minPts, edgeTol)
        # rule BIG: largest cluster
        bg = max(cl, key=len)
        out['cor_big'] = _corner_of(bg, tor, minPts, edgeTol)
        out['nclust'] = len(cl)
    else:
        out['cor_nearz']=corner; out['cor_neartc']=corner; out['cor_big']=corner
        out['nclust']=0
    return out

# ---------------------------------------------------------------- GT load
def load_gt(root):
    rows=[]
    with open(os.path.join(root,'groundtruth.txt'),encoding='utf-8') as f:
        for ln in f.read().splitlines()[1:]:
            c=[x.strip() for x in ln.split(';')]
            if len(c)<6: continue
            job,frame,state=c[0],c[1],c[2]
            try: T,H,G=float(c[3]),float(c[4]),float(c[5])
            except ValueError: continue
            rows.append(dict(job=job,frame=frame,state=state,T=T,H=H,G=G))
    return rows

def main():
    global CONE_BACK, CONE_FWD, PROTR_RADIUS
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=GT_ROOT)
    ap.add_argument('--orig', default=ORIG_ROOT)
    ap.add_argument('--half-t', type=float, default=150.0)
    ap.add_argument('--half-g', type=float, default=150.0)
    ap.add_argument('--csv', default='')
    ap.add_argument('--cone-back', type=float, default=70.0)
    ap.add_argument('--cone-fwd', type=float, default=130.0)
    ap.add_argument('--protr-radius', type=float, default=120.0)
    ap.add_argument('--z-slab', type=float, default=None, help='depth slab from near edge inward')
    ap.add_argument('--hmax', type=float, default=None, help='override cone-height ceiling')
    ap.add_argument('--hmin', type=float, default=None, help='override cone-height floor')
    ap.add_argument('--save', action='store_true',
                    help='write _NEW_ artifacts (crop/slab/bump/conePos) into each frame folder')
    ap.add_argument('--save-root', default='',
                    help='write artifacts under this root/<job>/<frame>/ instead of the orig frame folder')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    CONE_BACK, CONE_FWD, PROTR_RADIUS = a.cone_back, a.cone_fwd, a.protr_radius
    global ZSLAB
    if a.z_slab is not None: ZSLAB = a.z_slab
    global HEIGHT_MIN, HEIGHT_MAX
    if a.hmax is not None: HEIGHT_MAX = a.hmax
    if a.hmin is not None: HEIGHT_MIN = a.hmin

    gt = load_gt(a.gt)
    out_rows = []
    miss_crop = 0
    for g in gt:
        if g['state'] not in ('CONLY','LANDING'): continue
        frame = g['frame']
        bp = os.path.join(a.orig, g['job'], frame, frame+'_base_converted.pcd')
        if not os.path.isfile(bp):
            if a.verbose: print("missing:", bp)
            continue
        pts = read_pcd(bp)
        pts = pts[~((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))]
        jp,chs,sprd,spos = parse_job(g['job'])
        sensor = frame_sensor(frame)
        pc = preset_for(g['job'], frame)
        if pc is None: continue
        cg, ct, cy = pc                        # (gantry col0, trolley col2, height col1)
        crop = preset_crop(pts, cg, ct)
        tor = target_on_right(sensor, jp, chs)
        det = detect(crop, tor, cg)            # protrusion anchored on Gantry(col0) center
        key = config_key(sensor, jp, chs)
        cfg = f"{sensor} {jp} CHS{chs}"
        gtv = np.array([g['T'],g['H'],g['G']])
        if det is None:
            out_rows.append((key,cfg,g['state'],frame,None,len(crop),g))
            if a.verbose: print(f"NO DETECT {cfg} {frame} cropN={len(crop)}")
            continue
        errs = {est: (det[est]-gtv) for est in
                ('centroid','corner','cor_nearz','cor_neartc','cor_big','cor_protr') if det.get(est) is not None}
        out_rows.append((key,cfg,g['state'],frame,errs,len(crop),g))

        if a.save:
            if a.save_root:
                outdir = os.path.join(a.save_root, g['job'], frame); os.makedirs(outdir, exist_ok=True)
            else:
                outdir = os.path.join(a.orig, g['job'], frame)
            base = os.path.join(outdir, frame)
            try:
                write_pcd(base+'_NEW_crop.pcd', crop)
                if det.get('slab') is not None: write_pcd(base+'_NEW_slab.pcd', det['slab'])
                if det.get('bump') is not None: write_pcd(base+'_NEW_bump.pcd', det['bump'])
                # cone point as a small cluster so it's visible in CloudCompare
                cp = det['cor_protr']
                pts9 = cp + np.array([[dx,dy,0] for dx in (-3,0,3) for dy in (-3,0,3)])
                write_pcd(base+'_NEW_conePos.pcd', pts9)
                write_pcd(base+'_NEW_GTpos.pcd',
                          gtv + np.array([[dx,dy,0] for dx in (-3,0,3) for dy in (-3,0,3)]))
            except Exception as e:
                if a.verbose: print('save fail', frame, e)

    ESTS = ['centroid','corner','cor_nearz','cor_neartc','cor_big','cor_protr']
    def agg(rows, est, state=None, key=None):
        sel=[r[4][est] for r in rows if r[4] is not None and est in r[4] and
             (state is None or r[2]==state) and (key is None or r[0]==key)]
        if not sel: return None
        a_=np.abs(np.array(sel)); b_=np.array(sel)
        return len(sel), np.median(a_,0), np.percentile(a_,95,0), np.median(b_,0)

    print(f"\nhalf_t={a.half_t}  half_g={a.half_g}   (errors = est - GT, mm). FOCUS = CONLY chassis accuracy.\n")
    # CONLY summary per estimator, per key + overall
    for est in ESTS:
        print(f"================== {est} ==================")
        print(f"{'group':22} {'st':8} {'n':>3}  {'|dT|':>5} {'|dH|':>5} {'|dG|':>5}  {'p95T':>5} {'biasT':>6}")
        for key in sorted(PRESETS):
            r = agg(out_rows, est, 'CONLY', key)
            if not r: continue
            n,med,p95,bias = r
            print(f"{key:22} {'CONLY':8} {n:3d}  {med[0]:5.0f} {med[1]:5.0f} {med[2]:5.0f}  {p95[0]:5.0f} {bias[0]:6.0f}")
        for st in ('CONLY','LANDING'):
            r = agg(out_rows, est, st)
            if not r: continue
            n,med,p95,bias = r
            print(f"{'-- ALL':22} {st:8} {n:3d}  {med[0]:5.0f} {med[1]:5.0f} {med[2]:5.0f}  {p95[0]:5.0f} {bias[0]:6.0f}")
        print()

    nnod = sum(1 for r in out_rows if r[4] is None)
    print(f"frames scored: {sum(1 for r in out_rows if r[4] is not None)}   no-detect: {nnod}")

    if a.csv:
        import csv
        with open(a.csv,'w',newline='') as f:
            w=csv.writer(f)
            cols=['key','cfg','state','frame','cN']
            for e in ESTS: cols += [e+'_dT',e+'_dH',e+'_dG']
            cols += ['GT_T','GT_H','GT_G']
            w.writerow(cols)
            for key,cfg,st,frame,errs,cN,g in out_rows:
                row=[key,cfg,st,frame,cN]
                for e in ESTS:
                    row += list(errs[e]) if (errs and e in errs) else ['','','']
                row += [g['T'],g['H'],g['G']]
                w.writerow(row)
        print("wrote", a.csv)

if __name__ == '__main__':
    main()
