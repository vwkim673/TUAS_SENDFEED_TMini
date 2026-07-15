"""
roi_plane_target.py - EXPERIMENTAL plane-based target extraction (vs current pipeline).

Same preset crop as production, but the target is derived from three 2D cross-sections
instead of the candBand/cone_columns/tight-box chain:

  P1  X-Z (top view, Y prior band cy+-140):
      per-X near-Z profile -> protruding column groups -> side-rule -> target X range + zFront
  P2  Y-Z (side view, slice at the target X range, FULL Y):
      occupancy grid -> yTop(z) profile -> coneTop, plate level, zBack (where the top
      surface drops below plate level = body/deck starts)
  P3  X-Y (front view, near-Z face z in [zFront-30, zFront+90]):
      plate-band X edge (side-rule) -> corner X ; corner Y = local plate surface ;
      corner Z = near-Z at the corner column

Outputs per frame: PLANE target point + region box; compared to GT and to the current
NEW pipeline on all GT CONLY frames.

Usage: python roi_plane_target.py [--show]
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R
import roi_containment as C
import roi_cone_extract as E

BIN = 30.0

def dz(v, w=40, m=30, mx=False):
    v = np.asarray(v)
    return R.dense_extreme(v, w, m, find_max=mx) if len(v) else 0.0

def dense_edge(v, win, k, find_max):
    v = np.sort(np.asarray(v))
    if len(v) < k: return float(v[-1] if find_max else v[0]) if len(v) else 0.0
    if find_max:
        for i in range(len(v)-1, k-2, -1):
            if v[i]-v[i-k+1] <= win: return float(v[i])
        return float(v[-1])
    for i in range(len(v)-k+1):
        if v[i+k-1]-v[i] <= win: return float(v[i])
    return float(v[0])

# ---------------- P1 : X-Z top view ----------------
def p1_xz_v2(crop, cy, tor, yband=140.0, minCol=40, protrude=100.0, gap=60.0):
    """per-X near-Z profile -> protruding groups -> side-rule -> (x0,x1,zFront)."""
    band = crop[np.abs(crop[:,1]-cy) <= yband]
    if len(band) < 60: return None
    xs = band[:,0]
    edges = np.arange(xs.min(), xs.max()+BIN, BIN)
    cen, nz = [], []
    for i in range(len(edges)-1):
        m = (xs >= edges[i]) & (xs < edges[i+1])
        if m.sum() < minCol: continue
        cen.append(edges[i]+BIN/2); nz.append(dz(band[m,2], 40, 10, False))
    if not cen: return None
    cen = np.array(cen); nz = np.array(nz); gmin = nz.min()
    isc = nz <= gmin + protrude
    groups, cur = [], []
    for k in range(len(cen)):
        if isc[k]:
            if cur and cen[k]-cen[cur[-1]] > gap: groups.append(cur); cur = []
            cur.append(k)
        elif cur: groups.append(cur); cur = []
    if cur: groups.append(cur)
    if not groups: return None
    if tor is True:    grp = max(groups, key=lambda g: np.mean(cen[g]))   # rightmost cone
    elif tor is False: grp = min(groups, key=lambda g: np.mean(cen[g]))   # leftmost cone
    else:              grp = min(groups, key=lambda g: abs(np.mean(cen[g])))
    x0 = cen[grp[0]] - BIN/2; x1 = cen[grp[-1]] + BIN/2
    zF = float(np.min(nz[grp]))
    return x0, x1, zF

# ---------------- P2 : Y-Z side view ----------------
def p2_yz(crop, x0, x1, zF, cy, occMin=4, zSpan=420.0, coneSpan=160.0, dropTol=60.0, yPrior=320.0):
    """yTop(z) profile in the target X slice -> coneTop, plateY, zBack."""
    sl = crop[(crop[:,0] >= x0-30) & (crop[:,0] <= x1+30)]
    sl = sl[(sl[:,1] >= cy-yPrior) & (sl[:,1] <= cy+250)]  # asym Y prior: cone tops out ~+150 above cy
    sl = sl[(sl[:,2] >= zF-30) & (sl[:,2] <= zF+zSpan)]
    if len(sl) < 40: return None
    zb = np.arange(zF-30, zF+zSpan+BIN, BIN)
    yTop = np.full(len(zb)-1, np.nan); zc = zb[:-1]+BIN/2
    for i in range(len(zb)-1):
        m = (sl[:,2] >= zb[i]) & (sl[:,2] < zb[i+1])
        if m.sum() < occMin: continue
        ys = np.sort(sl[m,1])
        yTop[i] = ys[max(0, int(len(ys)*0.95)-1)]      # robust top (95th pct)
    ok = ~np.isnan(yTop)
    if ok.sum() < 3: return None
    # cone top = max yTop near the front
    frontZone = ok & (zc <= zF+coneSpan)
    if not frontZone.any(): frontZone = ok
    coneTop = float(np.nanmax(yTop[frontZone]))
    # plate level = median yTop over the zone BEHIND the cone bump (where profile flattens)
    behind = ok & (zc > zF+coneSpan*0.6)
    plateY = float(np.nanmedian(yTop[behind])) if behind.any() else float(np.nanmedian(yTop[ok]))
    # zBack: first z beyond zF+120 where yTop drops below plateY-dropTol (deck/gap) -> body cut
    zBack = zF + zSpan
    for i in range(len(zc)):
        if zc[i] < zF+120: continue
        if not ok[i] or yTop[i] < plateY - dropTol:
            zBack = float(zc[i] - BIN/2); break
    return coneTop, plateY, float(zBack)

# ---------------- P3 : X-Y front view ----------------
def p3_xy(crop, x0, x1, zF, coneTop, plateY, tor, faceDepth=90.0, edgeK=8):
    """corner from the near-Z face: plate-band X edge (side-rule) + local surface Y +
    near-Z at the corner column."""
    face = crop[(crop[:,2] >= zF-30) & (crop[:,2] <= zF+faceDepth)]
    face = face[(face[:,1] >= plateY-60) & (face[:,1] <= coneTop+20)]   # plate..coneTop band
    # widen ONLY toward the corner side (group extent is coarse but the far side must stay shut)
    if tor is False: face = face[(face[:,0] >= x0-30) & (face[:,0] <= x1+120)]
    else:            face = face[(face[:,0] >= x0-120) & (face[:,0] <= x1+30)]
    if len(face) < 20: return None
    kk = max(6, min(edgeK, len(face)//15))                              # adaptive edge support
    xE = dense_edge(face[:,0], 40, kk, find_max=(tor is False))
    col = face[np.abs(face[:,0]-xE) <= 60]
    if len(col) < 6: col = face
    cx = float(np.median(col[:,0]))
    czv = float(dz(col[:,2], 40, max(5, len(col)//8), False))
    m = (np.abs(face[:,0]-cx) <= 60) & (np.abs(face[:,2]-czv) <= 60)
    cyv = float(np.median(face[m,1])) if m.sum() else float(np.median(col[:,1]))
    return np.array([cx, cyv, czv])

def plane_target(crop, cy, tor):
    r1 = p1_xz_v2(crop, cy, tor)
    if r1 is None: return None, None
    x0, x1, zF = r1
    r2 = p2_yz(crop, x0, x1, zF, cy)
    if r2 is None: return None, None
    coneTop, plateY, zBack = r2
    pt = p3_xy(crop, x0, x1, zF, coneTop, plateY, tor)
    box = (x0, x1, plateY-100, coneTop, zF, zBack)
    return pt, box

# ---------------- current NEW (reference) ----------------
def new_target(cone, tor):
    c = E.sor(cone, 12, 2.0)
    if len(c) >= 30: cone = c
    midY = 0.5*(cone[:,1].min()+cone[:,1].max()); up = cone[cone[:,1] >= midY]
    if len(up) < 20: up = cone
    zN = dz(up[:,2],40,30,False); f = up[up[:,2] <= zN+80]
    if len(f) < 20: f = up
    # detector
    zNear=dz(f[:,2],40,30,False); zs=f[(f[:,2]>=zNear-100)&(f[:,2]<=zNear+300)]
    if len(zs)<5: zs=f
    if not tor: mx=dz(zs[:,0],40,30,True); filt=zs[(zs[:,0]>=mx-300)&(zs[:,0]<=mx+100)]
    else: mn=dz(zs[:,0],40,30,False); filt=zs[(zs[:,0]>=mn-100)&(zs[:,0]<=mn+300)]
    if len(filt)<5: filt=zs
    ylo,yhi=filt[:,1].min(),filt[:,1].max();best=None;bn=0;cyv=yhi
    while cyv>ylo:
        sl=filt[(filt[:,1]>=cyv-30)&(filt[:,1]<cyv)]
        if len(sl)>bn:bn=len(sl);best=sl
        cyv-=30
    mpy=best if best is not None and len(best)>0 else filt
    def d2(a,ax):
        lo,hi=a[:,ax].min(),a[:,ax].max();b=None;bb=0;c2=lo
        while c2<hi:
            sl=a[(a[:,ax]>=c2)&(a[:,ax]<c2+20)]
            if len(sl)>bb:bb=len(sl);b=sl
            c2+=20
        return b if b is not None else a
    mpx=d2(mpy,0);mpz=d2(mpy,2)
    refx=mpz[:,0].min() if tor else mpz[:,0].max()
    d=(mpy[:,0]-refx)**2+(mpy[:,2]-mpx[:,2].min())**2; p=mpy[int(np.argmin(d))]
    cx,cz=float(p[0]),float(p[2]); m=(np.abs(f[:,0]-cx)<=60)&(np.abs(f[:,2]-cz)<=60)
    pt=np.array([cx,float(np.median(f[m,1])) if m.sum() else float(p[1]),cz])
    if len(f)<250:
        xE=dense_edge(f[:,0],40,8,find_max=not tor)
        if (tor and xE<pt[0]-40) or ((not tor) and xE>pt[0]+40):
            col=f[np.abs(f[:,0]-xE)<=60]
            if len(col)>=12:
                cx=float(np.median(col[:,0])); cz=dense_edge(col[:,2],25,6,find_max=False)
                m=(np.abs(f[:,0]-cx)<=60)&(np.abs(f[:,2]-cz)<=60)
                cyv=float(np.median(f[m,1])) if m.sum() else pt[1]
                pt=np.array([cx,cyv,cz])
    return pt

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--show', action='store_true')
    a = ap.parse_args()
    import collections
    res = collections.defaultdict(lambda: {'NEW': [], 'PLN': []})
    miss = 0
    for g in R.load_gt(R.GT_ROOT):
        if g['state'] != 'CONLY': continue
        jp, chs, sprd, spos = R.parse_job(g['job']); sensor = R.frame_sensor(g['frame']); key = R.config_key(sensor, jp, chs)
        pts = R.load_cloud(g['job'], g['frame'])
        if pts is None: continue
        cg, ct, cy = R.PRESETS[key]; tor = R.target_on_right(sensor, jp, chs)
        crop = R.preset_crop(pts, cg, ct)
        cone = E.extract_cone(crop, cg, ct, cy, tor)
        if cone is None or len(cone) < 30: continue
        gt = np.array([g['T'], g['H'], g['G']])
        nw = new_target(cone, tor)
        pl, box = plane_target(crop, cy, tor)
        if pl is None: miss += 1; continue
        res[key]['NEW'].append((g['frame'][9:15], nw-gt)); res[key]['PLN'].append((g['frame'][9:15], pl-gt))
        if a.show:
            print('%-16s %s NEW(%+4.0f,%+4.0f,%+4.0f)  PLN(%+4.0f,%+4.0f,%+4.0f)' % (key, g['frame'][9:15], *(nw-gt), *(pl-gt)))
    print()
    print('%-16s |  NEW dT dH dG |  PLN dT dH dG' % 'config')
    for k in sorted(res):
        def md(v, i): return np.median([x[1][i] for x in v])
        n = res[k]['NEW']; p = res[k]['PLN']
        print('%-16s | %4.0f%4.0f%4.0f | %4.0f%4.0f%4.0f' % (k, md(n,0), md(n,1), md(n,2), md(p,0), md(p,1), md(p,2)))
    for lab in ('NEW', 'PLN'):
        aa = [x[1] for v in res.values() for x in v[lab]]
        print('%s median|err| dT %.0f dH %.0f dG %.0f | w20 %d/%d w30 %d w40 %d' % (lab,
            np.median([abs(x[0]) for x in aa]), np.median([abs(x[1]) for x in aa]), np.median([abs(x[2]) for x in aa]),
            sum(1 for x in aa if max(abs(x)) <= 20), len(aa), sum(1 for x in aa if max(abs(x)) <= 30), sum(1 for x in aa if max(abs(x)) <= 40)))
    print('plane-miss frames:', miss)

if __name__ == '__main__':
    main()
