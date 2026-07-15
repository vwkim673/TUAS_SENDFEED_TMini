"""
roi_user_pipeline.py - user-proposed pipeline (2026-07), evaluated vs current NEW.

  1. preset crop (same)
  2. Y floor by camera: FS >= -500, CS >= -550, RS >= -600   (drop below)
  3. X-Z: walk Z from near, FIRST 25mm slice with >= zThr pts = plate front edge;
     keep Z in [zc-50, zc+150]  (zc = slice centre)
  4. SOR isolated-flyer filter
  5. X-Y: per-30mm Y band -> (count, robust X width). Plate band = widest band with
     enough points whose neighbours above AND below are narrower (plate signature).
  6. keep Y in [plateCenter-100, plateCenter+100]
  7. existing corner detector + local-median corner height

Usage: python roi_user_pipeline.py [--show]
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R
import roi_containment as C
import roi_cone_extract as E

YFLOOR = {'FS': -500.0, 'CS': -550.0, 'RS': -600.0}

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

def detect(cone, tor):
    """existing corner detector + local-median corner height (same as NEW reference)"""
    zNear=dz(cone[:,2],40,30,False); zs=cone[(cone[:,2]>=zNear-100)&(cone[:,2]<=zNear+300)]
    if len(zs)<5: zs=cone
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
    cx,cz=float(p[0]),float(p[2]); m=(np.abs(cone[:,0]-cx)<=60)&(np.abs(cone[:,2]-cz)<=60)
    return np.array([cx,float(np.median(cone[m,1])) if m.sum() else float(p[1]),cz])

def xgroup_siderule(c, tor, xbin=40.0, minCol=40, protrude=100.0, gap=60.0):
    """lateral target-cone selection (the missing piece for width pairs): per-X near-Z
    protrusion groups on the floored crop -> side-rule -> [x0, x1]."""
    xs = c[:,0]
    edges = np.arange(xs.min(), xs.max()+xbin, xbin)
    cen, nz = [], []
    for i in range(len(edges)-1):
        m = (xs >= edges[i]) & (xs < edges[i+1])
        if m.sum() < minCol: continue
        cen.append(edges[i]+xbin/2); nz.append(dz(c[m,2], 40, 10, False))
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
    if tor is True:    grp = max(groups, key=lambda g: np.mean(cen[g]))
    elif tor is False: grp = min(groups, key=lambda g: np.mean(cen[g]))
    else:              grp = min(groups, key=lambda g: abs(np.mean(cen[g])))
    return cen[grp[0]]-xbin/2, cen[grp[-1]]+xbin/2

def user_pipeline(crop, sensor, tor, cy=None, zThr=30, zBin=25.0, yBin=30.0, densThr=25, dbg=None):
    # 2. per-camera Y floor
    fl = YFLOOR.get(sensor, -550.0)
    c = crop[crop[:,1] >= fl]
    if len(c) < 60: return None
    # 2.5 lateral cone selection (side-rule) so the Z edge is the TARGET's edge.
    #     Grouping runs on the cy+-140 band when cy is available: the CHS20 wheel top
    #     (-480) overlaps the plate bottom (-476) in Y, so NO Y-floor can exclude the
    #     wheel -- only the cone-height prior can (same reason production uses candBand).
    gsrc = c if cy is None else c[np.abs(c[:,1]-cy) <= 140]
    if len(gsrc) < 60: gsrc = c
    xr = xgroup_siderule(gsrc, tor)
    if xr is not None:
        c = c[(c[:,0] >= xr[0]-30) & (c[:,0] <= xr[1]+30)]
        if len(c) < 60: return None
    # 3. first valid Z slice = plate front edge -> Z window [zc-50, zc+150].
    #    The SEARCH runs on the cone-height band when cy is known: the FS CHS20 wheel sits
    #    at the SAME X as the cone and NEARER Z, so without the cy prior the first valid
    #    slice is the wheel front. The WINDOW is then cut from the floored crop.
    zsrc = c if cy is None else c[np.abs(c[:,1]-cy) <= 140]
    if len(zsrc) < 40: zsrc = c
    zb = np.arange(zsrc[:,2].min(), zsrc[:,2].max()+zBin, zBin)
    zc = None
    for i in range(len(zb)-1):
        if ((zsrc[:,2] >= zb[i]) & (zsrc[:,2] < zb[i+1])).sum() >= zThr:
            zc = zb[i] + zBin/2; break
    if zc is None: return None
    c = c[(c[:,2] >= zc-50) & (c[:,2] <= zc+150)]
    if len(c) < 40: return None
    # 4. SOR
    s = E.sor(c, 12, 2.0)
    if len(s) >= 30: c = s
    # 5. X-Y: plate band = widest dense Y band, narrower above AND below
    yb = np.arange(c[:,1].min(), c[:,1].max()+yBin, yBin)
    nB = len(yb)-1
    cnt = np.zeros(nB); wid = np.zeros(nB)
    for i in range(nB):
        m = (c[:,1] >= yb[i]) & (c[:,1] < yb[i+1])
        cnt[i] = m.sum()
        if cnt[i] >= 6:
            xs = np.sort(c[m,0]); wid[i] = xs[max(0,int(len(xs)*0.95)-1)] - xs[int(len(xs)*0.05)]
    okd = cnt >= densThr
    if not okd.any(): return None
    # plate = the TOPMOST dense band that is still wide (>=70% of max width):
    # the cone ABOVE the plate is narrow (user's check), the deck BELOW is wide but
    # LOWER -- so scanning from the top, the first wide band is the plate.
    wmax = wid[okd].max() if okd.any() else 0.0
    pick = None
    for i in range(nB-1, -1, -1):
        if okd[i] and wid[i] >= 0.7*wmax: pick = i; break
    if pick is None: pick = int(np.argmax(np.where(okd, wid, 0)))
    plateY = yb[pick] + yBin/2
    if dbg is not None: dbg.update(zc=zc, plateY=plateY, plateW=wid[pick])
    # 6. Y window plate centre +-100
    c = c[(c[:,1] >= plateY-100) & (c[:,1] <= plateY+100)]
    if len(c) < 30: return None
    # 7. detector
    return detect(c, tor)

def new_target(cone, tor):
    c = E.sor(cone, 12, 2.0)
    if len(c) >= 30: cone = c
    midY = 0.5*(cone[:,1].min()+cone[:,1].max()); up = cone[cone[:,1] >= midY]
    if len(up) < 20: up = cone
    zN = dz(up[:,2],40,30,False); f = up[up[:,2] <= zN+80]
    if len(f) < 20: f = up
    pt = detect(f, tor)
    if len(f) < 250:
        xE = dense_edge(f[:,0],40,8,find_max=not tor)
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
    res = collections.defaultdict(lambda: {'NEW': [], 'USR': []})
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
        us = user_pipeline(crop, sensor, tor, cy)
        if us is None: miss += 1; continue
        res[key]['NEW'].append((g['frame'][9:15], nw-gt)); res[key]['USR'].append((g['frame'][9:15], us-gt))
        if a.show:
            print('%-16s %s NEW(%+4.0f,%+4.0f,%+4.0f)  USR(%+4.0f,%+4.0f,%+4.0f)' % (key, g['frame'][9:15], *(nw-gt), *(us-gt)))
    print()
    print('%-16s |  NEW dT dH dG |  USR dT dH dG' % 'config')
    for k in sorted(res):
        def md(v, i): return np.median([x[1][i] for x in v])
        n = res[k]['NEW']; u = res[k]['USR']
        print('%-16s | %4.0f%4.0f%4.0f | %4.0f%4.0f%4.0f' % (k, md(n,0), md(n,1), md(n,2), md(u,0), md(u,1), md(u,2)))
    for lab in ('NEW', 'USR'):
        aa = [x[1] for v in res.values() for x in v[lab]]
        print('%s median|err| dT %.0f dH %.0f dG %.0f | w20 %d/%d w30 %d w40 %d' % (lab,
            np.median([abs(x[0]) for x in aa]), np.median([abs(x[1]) for x in aa]), np.median([abs(x[2]) for x in aa]),
            sum(1 for x in aa if max(abs(x)) <= 20), len(aa), sum(1 for x in aa if max(abs(x)) <= 30), sum(1 for x in aa if max(abs(x)) <= 40)))
    print('pipeline-miss frames:', miss)

if __name__ == '__main__':
    main()
