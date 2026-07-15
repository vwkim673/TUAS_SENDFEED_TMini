"""
roi_cone_extract.py - Stage-2 cone/plate extraction from the (wide) preset ROI.

Stage 1 (preset_crop, already good): a generous ROI that INCLUDES the target.
Stage 2 (here): tighten to the cone/plate inside that ROI, by:
  1. nearest-Z passthrough  -> keep the layer facing the sensor (drops the deeper deck)
  2. statistical outlier removal (noise filter) -> drop flyers
  3. Euclidean clustering (grid connected-components in Gantry-X / Trolley-Z)
  4. select the cluster nearest the preset centre (the target plate, not a partner)

Scored against the human GT cone body:
  recall   = % of GT body covered by OUR cone (within TOL)   <- "GT에 반도 안들어온다"
  prec     = % of OUR cone sitting on GT body
  IoU-ish  = recall*prec proxy

Usage: python roi_cone_extract.py [--plate-depth 300] [--sor-k 12 --sor-std 2.0]
                                   [--clust-bin 35] [--state CONLY] [--show N]
"""
import sys, os, argparse
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R
import roi_containment as C

# ---------------- Stage-2 pieces ----------------
def sor(pts, k=12, std_mult=2.0):
    """Statistical outlier removal: drop points whose mean kNN distance is an outlier."""
    if len(pts) <= k: return pts
    tree = cKDTree(pts)
    d,_ = tree.query(pts, k=k+1)          # incl. self
    md = d[:,1:].mean(1)
    thr = md.mean() + std_mult*md.std()
    return pts[md <= thr]

def grid_clusters(pts, binw=35.0, min_pts=40):
    """Connected components on an (X,Z) occupancy grid; returns list of point arrays."""
    if len(pts) < min_pts: return []
    X = pts[:,0]; Z = pts[:,2]
    xi = ((X - X.min())/binw).astype(int); zi = ((Z - Z.min())/binw).astype(int)
    nx, nz = xi.max()+1, zi.max()+1
    grid = np.zeros((nx, nz), bool); grid[xi, zi] = True
    lbl, n = ndimage.label(grid, structure=np.ones((3,3)))
    out = []
    plab = lbl[xi, zi]
    for i in range(1, n+1):
        m = plab == i
        if m.sum() >= min_pts: out.append(pts[m])
    return out

def cone_z_extent(z, binw=25.0, frac=0.18, mind=150.0, maxd=450.0):
    """Valid target-Z band. The near-Z FRONT is found by walking out from the
    density PEAK while density >= frac*peak -> drops the sparse near-Z NOISE in
    front of the cone. The BACK extends to the density valley (deck) but is floored
    at front+mind (so a sharp single-peak histogram doesn't collapse the plate) and
    capped at front+maxd. Returns (front, back)."""
    if len(z) < 8: return float(z.min()), float(z.min())+mind
    bins = np.arange(z.min(), z.max()+binw, binw)
    if len(bins) < 3: return float(z.min()), float(z.min())+mind
    h,_ = np.histogram(z, bins=bins)
    pk = int(np.argmax(h)); thr = frac*h[pk]
    i = pk
    while i > 0 and h[i-1] >= thr: i -= 1
    j = pk
    while j < len(h)-1 and h[j+1] >= thr: j += 1
    front = bins[i]
    back = min(max(bins[j+1], front+mind), front+maxd)
    return float(front), float(back)

def front_z(z, window=50.0, min_pts=12):
    """Valid front (nearest) Z after denoise: the first Z where a small but real
    cluster begins (>=min_pts within window). Catches the sparse cone TIP without
    latching a single flyer. Assumes z already noise-filtered."""
    return R.dense_extreme(z, window, min_pts, find_max=False)

def x_gap_clusters(pts, gap=70.0, min_pts=25):
    """Split by gaps along Gantry-X (cone tips separate at the near-Z layer)."""
    if len(pts) < min_pts: return []
    s = pts[np.argsort(pts[:,0])]
    cuts = np.where(np.diff(s[:,0]) > gap)[0]
    return [g for g in np.split(s, cuts+1) if len(g) >= min_pts]

def cone_columns(crop, xbin=40.0, protrude_tol=150.0, gap=70.0, min_pts=25,
                 znear_pct=10.0, zpad=30.0, slab=130.0, min_col_pts=60):
    """Per-Gantry-X-column cone detection by NEAREST-Z PROTRUSION.

    Why not a single global near-Z slab: when two cones sit at different depths
    (e.g. a width pair, target deeper than its partner), one global near-Z anchors
    on the NEARER cone and the slab excludes the deeper target -> it never becomes a
    candidate. Here each X-bin gets its OWN near-Z (p10); cone bins are those whose
    near-Z is within `protrude_tol` of the global min (both cones protrude), while the
    deck/bridge BETWEEN them sits farther in Z and is dropped -> contiguous cone bins
    split into one candidate PER cone regardless of their relative depth.
    Returns a list of point arrays (each cone's near-Z protrusion), for the side-rule."""
    if len(crop) < min_pts: return []
    xs = crop[:,0]
    edges = np.arange(xs.min(), xs.max()+xbin, xbin)
    cen=[]; nz=[]
    for i in range(len(edges)-1):
        m = (xs>=edges[i]) & (xs<edges[i+1])
        # min_col_pts is high on purpose: a real cone column is dense (100s of pts);
        # a thin near-Z FLYER/edge structure (a few dozen pts) must NOT set the global
        # protrusion reference, else protrude_tol excludes the true cone. near-Z via
        # dense_extreme (needs a real cluster) also rejects intra-bin flyers.
        if m.sum() < min_col_pts: continue
        cen.append(0.5*(edges[i]+edges[i+1]))
        nz.append(R.dense_extreme(crop[m,2], 40.0, 15, find_max=False))
    if not cen: return []
    cen=np.array(cen); nz=np.array(nz); gmin=nz.min()
    is_cone = nz <= gmin + protrude_tol         # protruding bins = cone, deck excluded
    groups=[]; cur=[]
    for k in range(len(cen)):
        if is_cone[k]:
            if cur and cen[k]-cen[cur[-1]] > gap: groups.append(cur); cur=[]
            cur.append(k)
        elif cur:
            groups.append(cur); cur=[]
    if cur: groups.append(cur)
    out=[]
    for grp in groups:
        xlo=cen[grp[0]]-xbin; xhi=cen[grp[-1]]+xbin; znear=nz[grp].min()
        m=(xs>=xlo)&(xs<=xhi)&(crop[:,2]>=znear-zpad)&(crop[:,2]<=znear+slab)
        if m.sum() >= min_pts: out.append(crop[m])
    return out

def extract_cone(crop, cg, ct, cy=None, tor=None, plate_depth=200.0, zpad=30.0,
                 sor_k=12, sor_std=2.0, tip_depth=90.0, gap=70.0, cone_min=25,
                 yhalf=150.0, xspan=300.0, yband=200.0):
    """Stage-2: find cone tips, pick THE target by job side-rule, take its tight plate.
    cy  = preset cone-height (Y); restricts work to the cone layer, EXCLUDING the wheel
          (CHS20: wheel is nearer-Z than the cone, ~300-480mm below it).
    tor = target_on_right (True -> rightmost / max-X cone; False -> leftmost / min-X)."""
    if crop is None or len(crop) < cone_min: return None
    # 0. keep only the cone-height layer (drop wheel/ground below, high structures above)
    if cy is not None:
        crop = crop[(crop[:,1] >= cy-yband) & (crop[:,1] <= cy+yband)]
        if len(crop) < cone_min: return None
    # 1. cone candidates = per-X-column near-Z protrusions. Each cone keeps its OWN
    #    near-Z, so a deeper target cone is NOT lost behind a nearer partner (the
    #    failure of a single global near-Z slab). Fallback to the old global-tip split
    #    if the column method finds nothing (e.g. very sparse crop).
    cands = cone_columns(crop, gap=gap, min_pts=cone_min, zpad=zpad, slab=tip_depth+40.0)
    if not cands:
        zNear = front_z(crop[:,2])
        tip = crop[(crop[:,2] >= zNear-zpad) & (crop[:,2] <= zNear+tip_depth)]
        tip = sor(tip, sor_k, sor_std)
        cands = x_gap_clusters(tip, gap, cone_min) or grid_clusters(tip, 35.0, cone_min) or [tip]
    # 2. SIDE-RULE selection among cone candidates
    if tor is True:    target = max(cands, key=lambda c: c[:,0].mean())   # right cone
    elif tor is False: target = min(cands, key=lambda c: c[:,0].mean())   # left cone
    else:              target = min(cands, key=lambda c: abs(c[:,0].mean()-cg))
    # 3. TIGHT ~300^3 box. The tip candidate is only the protruding NOSE of the cone,
    #    so DON'T use its narrow X-extent; centre an xspan-wide window on the selected
    #    cone's nose to capture the full plate width. (Y: cone height +- yhalf drops the
    #    tyre/wheel below; Z: dense-cone band, set later.)
    cx = float(np.median(target[:,0]))
    xlo, xhi = cx - xspan/2.0, cx + xspan/2.0
    coneY = cy if cy is not None else float(np.median(target[:,1]))
    ylo, yhi = coneY-yhalf, coneY+yhalf
    col = crop[(crop[:,0] >= xlo) & (crop[:,0] <= xhi) &
               (crop[:,1] >= ylo) & (crop[:,1] <= yhi)]
    if len(col) < cone_min: return target
    # valid target-Z = dense cone Z-cluster (drops front noise + back deck), not a
    # blind fixed depth that would either keep the noise or cut the back of the cone.
    zf, zb = cone_z_extent(col[:,2], maxd=plate_depth+120.0)
    plate = col[(col[:,2] >= zf) & (col[:,2] <= zb)]
    return plate if len(plate) else target

# ---------------- eval ----------------
def cov_prec(our, body, tol=25.0):
    if our is None or len(our)==0: return 0.0, 0.0
    recall = C.nn_frac(body, our, tol)   # % GT covered by OUR
    prec   = C.nn_frac(our, body, tol)   # % OUR on GT
    return recall, prec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=R.GT_ROOT); ap.add_argument('--orig', default=R.ORIG_ROOT)
    ap.add_argument('--plate-depth', type=float, default=300.0)
    ap.add_argument('--zpad', type=float, default=30.0)
    ap.add_argument('--sor-k', type=int, default=12)
    ap.add_argument('--sor-std', type=float, default=2.0)
    ap.add_argument('--clust-bin', type=float, default=35.0)
    ap.add_argument('--clust-min', type=int, default=40)
    ap.add_argument('--tol', type=float, default=25.0)
    ap.add_argument('--state', default='CONLY')
    ap.add_argument('--show', type=int, default=0)
    a = ap.parse_args()

    from collections import defaultdict
    res = defaultdict(list)
    for g in R.load_gt(a.gt):
        if a.state!='ALL' and g['state']!=a.state: continue
        if g['state'] not in ('CONLY','LANDING'): continue
        bp = os.path.join(a.orig, g['job'], g['frame'], g['frame']+'_base_converted.pcd')
        body = C.gt_body(a.gt, g['job'], g['frame'], g['state'])
        if not os.path.isfile(bp) or body is None: continue
        pc = R.preset_for(g['job'], g['frame'])
        if pc is None: continue
        cg,ct,cy = pc
        jp,chs,sprd,spos = R.parse_job(g['job']); sensor = R.frame_sensor(g['frame'])
        tor = R.target_on_right(sensor, jp, chs)
        pts = R.read_pcd(bp); pts = pts[~((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))]
        crop = R.preset_crop(pts, cg, ct)
        our = extract_cone(crop, cg, ct, cy, tor, a.plate_depth, a.zpad, a.sor_k, a.sor_std)
        rec, prec = cov_prec(our, body, a.tol)
        dXc = (our[:,0].mean()-body[:,0].mean()) if (our is not None and len(our)) else 9999
        xs = (our[:,0].max()-our[:,0].min()) if (our is not None and len(our)) else 0
        ys = (our[:,1].max()-our[:,1].min()) if (our is not None and len(our)) else 0
        zs = (our[:,2].max()-our[:,2].min()) if (our is not None and len(our)) else 0
        cfg = f"{sensor} {jp} CHS{chs}"
        res[cfg].append((rec, prec, dXc, xs, ys, zs, g['frame']))
        if a.show:
            on = 0 if our is None else len(our)
            print(f"{cfg:16} tor={str(tor):5} dXc={dXc:6.0f} box(XxYxZ)={xs:3.0f}x{ys:3.0f}x{zs:3.0f} rec={rec:3.0f}% prec={prec:3.0f}% n={on:5d} {g['frame']}")

    print(f"\n=== Stage-2 cone (side-rule, tight box) vs GT body ({a.state}, pd={a.plate_depth}) ===")
    print(f"{'config':16} {'n':>3} {'|dXcen|':>8} {'wrong':>6} {'Xsz':>4} {'Ysz':>4} {'Zsz':>4} {'rec%':>5} {'prec%':>6}")
    allr=[]; allp=[]; alld=[]; allx=[]; ally=[]; allz=[]
    for cfg in sorted(res):
        r=np.array([x[0] for x in res[cfg]]); p=np.array([x[1] for x in res[cfg]]); d=np.array([x[2] for x in res[cfg]])
        xs=np.array([x[3] for x in res[cfg]]); ys=np.array([x[4] for x in res[cfg]]); zs=np.array([x[5] for x in res[cfg]])
        allr+=list(r); allp+=list(p); alld+=list(np.abs(d)); allx+=list(xs); ally+=list(ys); allz+=list(zs)
        wrong=int((np.abs(d)>150).sum())
        print(f"{cfg:16} {len(r):3d} {np.median(np.abs(d)):8.0f} {wrong:3d}/{len(r):<2d} "
              f"{np.median(xs):4.0f} {np.median(ys):4.0f} {np.median(zs):4.0f} {np.median(r):5.0f} {np.median(p):6.0f}")
    print(f"{'-- ALL':16} {len(allr):3d} {np.median(alld):8.0f} {int((np.array(alld)>150).sum()):3d}/{len(alld):<2d} "
          f"{np.median(allx):4.0f} {np.median(ally):4.0f} {np.median(allz):4.0f} {np.median(allr):5.0f} {np.median(allp):6.0f}")

if __name__ == '__main__':
    main()
