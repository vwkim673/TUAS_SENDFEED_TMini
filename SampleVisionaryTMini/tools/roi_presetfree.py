"""
roi_presetfree.py - Find chassis-cone ROI(s) with MINIMAL assumptions (no preset).

Only assumption: cones live in a cone-height band (global H window) and the
twist-lock cone is a LOCAL nearest-Z PROTRUSION (sticks toward the side sensor).
No per-config trolley/gantry box. Returns ALL cone candidates (a job may have
several chassis cones; which one is the target is a SEPARATE selection step).

Method (organized-cloud friendly, O(N)):
  1. height-band + valid filter.
  2. project to a (T,G) grid; per cell keep nearest-Z (min depth) -> Z-surface.
  3. background = large-window MAX-Z (deck behind); protrusion = bg - Zsurf > thr.
  4. connected-components (scipy.ndimage.label) on protrusion cells -> candidates.
  5. filter by physical size (cone foot-print); rank by nearest-Z.

Eval vs GT: is the target cone among candidates? rank by nearest-Z? #candidates?
Times the core (cloud already in RAM in C++, so read time is excluded).

Usage: python roi_presetfree.py [--bin 30] [--prot 80] [--state CONLY]
"""
import sys, os, time, argparse
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R

def find_cones(pts, protrusion=80.0, bg_win=51, min_pix=20, max_pix=4000,
               max_foot=450.0, hmin=-600.0, hmax=200.0, W=512, Hgt=424):
    """pts = full organized cloud (W*Hgt, 3) in (T,H,G). G is depth toward the
    side sensor; the cone is a LOCAL MIN in G. Returns candidates + timing."""
    t0 = time.perf_counter()
    grid = pts.reshape(Hgt, W, 3)
    T = grid[:,:,0]; H = grid[:,:,1]; G = grid[:,:,2]
    valid = ~((T==0)&(H==0)&(G==0)) & (H>=hmin) & (H<=hmax)
    # depth image: G where valid, +inf elsewhere (so it never wins a min/can be ignored)
    Gimg = np.where(valid, G, np.inf).astype(np.float32)
    # background deck = farthest valid G in a window around each pixel
    Gback = np.where(valid, G, -np.inf).astype(np.float32)
    bg = ndimage.maximum_filter(Gback, size=bg_win, mode='nearest')
    prot = valid & np.isfinite(bg) & ((bg - Gimg) > protrusion)   # nearer than deck behind
    lbl, n = ndimage.label(prot)
    t1 = time.perf_counter()
    cands = []
    if n:
        objs = ndimage.find_objects(lbl)
        for i in range(1, n+1):
            sl = objs[i-1]
            sub = (lbl[sl]==i)
            npix = int(sub.sum())
            if npix < min_pix or npix > max_pix: continue
            mp = grid[sl][sub]
            mp = mp[~((mp[:,0]==0)&(mp[:,1]==0)&(mp[:,2]==0))]
            if len(mp) < min_pix: continue
            footT = mp[:,0].max()-mp[:,0].min(); footH = mp[:,1].max()-mp[:,1].min()
            if footT > max_foot or footH > max_foot: continue
            nearG = float(np.percentile(mp[:,2], 5))
            cands.append(dict(T=float(mp[:,0].mean()), H=float(mp[:,1].mean()),
                              G=float(mp[:,2].mean()), nearG=nearG, n=len(mp),
                              footT=float(footT), footH=float(footH)))
    cands.sort(key=lambda c: c['nearG'])     # nearest (smallest G) first
    return cands, (t1-t0)*1000.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=R.GT_ROOT); ap.add_argument('--orig', default=R.ORIG_ROOT)
    ap.add_argument('--bin', type=float, default=30.0)
    ap.add_argument('--prot', type=float, default=80.0)
    ap.add_argument('--bg-win', type=int, default=51)
    ap.add_argument('--match', type=float, default=120.0, help='mm to count a candidate as the GT cone')
    ap.add_argument('--state', default='CONLY')
    ap.add_argument('--show', type=int, default=0, help='print top-K candidates per frame')
    a = ap.parse_args()

    rows = []; times = []
    for g in R.load_gt(a.gt):
        if g['state'] not in ('CONLY','LANDING'): continue
        if a.state and g['state'] != a.state: continue
        bp = os.path.join(a.orig, g['job'], g['frame'], g['frame']+'_base_converted.pcd')
        if not os.path.isfile(bp): continue
        pts = R.read_pcd(bp)
        cands, ms = find_cones(pts, protrusion=a.prot, bg_win=a.bg_win); times.append(ms)
        gp = np.array([g['T'],g['H'],g['G']])
        # match by T,G (cone selection is lateral); rank among candidates by nearest-Z
        rank = None; best = None
        for idx,c in enumerate(cands):
            d = np.hypot(c['T']-g['T'], c['G']-g['G'])
            if d <= a.match and (best is None or d < best):
                best = d; rank = idx+1
        jp,chs,sprd,spos = R.parse_job(g['job']); cfg=f"{R.frame_sensor(g['frame'])} {jp} CHS{chs}"
        rows.append((cfg, g['frame'], len(cands), rank, best))
        if a.show:
            print(f"\n{cfg} {g['frame']}  GT(T{g['T']:.0f} G{g['G']:.0f})  ncand={len(cands)} match_rank={rank}")
            for idx,c in enumerate(cands[:a.show]):
                print(f"   #{idx+1} T{c['T']:6.0f} G{c['G']:6.0f} nearG{c['nearG']:6.0f} n{c['n']:4d} foot{c['footT']:.0f}x{c['footH']:.0f}")

    found = [r for r in rows if r[3] is not None]
    print(f"\n=== preset-free finder ({a.state}, bin={a.bin} prot={a.prot}) ===")
    print(f"frames {len(rows)}   target FOUND among candidates: {len(found)}/{len(rows)} "
          f"({100*len(found)/max(1,len(rows)):.0f}%)")
    if found:
        ranks=np.array([r[3] for r in found]); dists=np.array([r[4] for r in found])
        print(f"   when found: match dist median {np.median(dists):.0f}mm | "
              f"rank-by-nearestZ: #1 in {100*(ranks==1).mean():.0f}%, <=2 in {100*(ranks<=2).mean():.0f}%")
    ncs=np.array([r[2] for r in rows])
    print(f"   candidates per frame: median {np.median(ncs):.0f}, max {ncs.max()}")
    print(f"   core cost: median {np.median(times):.1f}ms, max {np.max(times):.1f}ms  (excl. python read)")
    miss=[r for r in rows if r[3] is None]
    if miss:
        print(f"   MISSED ({len(miss)}):")
        for cfg,frame,nc,rk,bd in miss: print(f"     {cfg:16} ncand={nc} {frame}")

if __name__ == '__main__':
    main()
