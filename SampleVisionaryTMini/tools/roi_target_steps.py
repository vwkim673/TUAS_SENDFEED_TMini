"""
roi_target_steps.py - Dump per-frame STEP point clouds (3D) for the target pipeline, so
each stage can be inspected in CloudCompare.

Per CONLY frame, into <dst>/steps/<job>/<frame>_*.pcd, numbered in PIPELINE ORDER:
  01_crop        preset ROI crop (context)
  02_candBand    candidate-DETECTION layer = crop @ cy +- CAND_YBAND (input to cone_columns;
                 comes BEFORE the region -- the region is built FROM the selected candidate)
  03_cand_sel    the SELECTED cone candidate (side-rule)   + 03_cand_other (rejected)
  04_coneRegion  extract_cone output: tight box around the selected cone, re-cut from the
                 FULL crop (X=nose+-150, Z=front+200, Y=cone_plate_yband) -- so the plate
                 bottom is not lost to the thin candidate band
  04b_regionClean SOR isolated-flyer removal (floating scatter would inflate max-Y and
                 push the upper-half mid-Y up)
  05_upperHalf   region upper half (Y >= region mid-Y)
  06_nearFace    near-Z face fed to the detector (front slab of the upper half)
  06b_rescueCol  sparse-edge rescue column (ONLY when the rescue fired)
  07_tgt_OLD / 07_tgt_NEW / 07_tgt_GT   single target points (NEW includes the rescue)

Usage: python roi_target_steps.py [--dst <dir>] [--frame <tok>]
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R
import roi_cone_extract as E

CAND_YBAND = 140.0
FACE_DEPTH = 80.0

def dz(v, w=40, m=30, mx=False):
    v = np.asarray(v)
    return R.dense_extreme(v, w, m, find_max=mx) if len(v) else 0.0

def detect(cone, tor):
    zNear = dz(cone[:,2],40,30,False); zs = cone[(cone[:,2]>=zNear-100)&(cone[:,2]<=zNear+300)]
    if len(zs) < 5: zs = cone
    if not tor: mx = dz(zs[:,0],40,30,True); filt = zs[(zs[:,0]>=mx-300)&(zs[:,0]<=mx+100)]
    else:       mn = dz(zs[:,0],40,30,False); filt = zs[(zs[:,0]>=mn-100)&(zs[:,0]<=mn+300)]
    if len(filt) < 5: filt = zs
    ylo, yhi = filt[:,1].min(), filt[:,1].max(); best = None; bn = 0; cyv = yhi
    while cyv > ylo:
        sl = filt[(filt[:,1]>=cyv-30)&(filt[:,1]<cyv)]
        if len(sl) > bn: bn = len(sl); best = sl
        cyv -= 30
    mpy = best if best is not None and len(best) > 0 else filt
    def dens(a, ax):
        lo, hi = a[:,ax].min(), a[:,ax].max(); b = None; bb = 0; c = lo
        while c < hi:
            sl = a[(a[:,ax]>=c)&(a[:,ax]<c+20)]
            if len(sl) > bb: bb = len(sl); b = sl
            c += 20
        return b if b is not None else a
    mpx = dens(mpy,0); mpz = dens(mpy,2)
    refx = mpz[:,0].min() if tor else mpz[:,0].max()
    d = (mpy[:,0]-refx)**2 + (mpy[:,2]-mpx[:,2].min())**2; p = mpy[int(np.argmin(d))]
    cx, cz = float(p[0]), float(p[2])
    m = (np.abs(cone[:,0]-cx)<=60)&(np.abs(cone[:,2]-cz)<=60)
    return np.array([[cx, float(np.median(cone[m,1])) if m.sum() else float(p[1]), cz]])

def clean(c):
    """Isolated-flyer removal (SOR) -- floating scatter above the cone otherwise
    inflates max-Y and pushes the upper-half mid-Y up."""
    if len(c) < 30: return c
    s = E.sor(c, 12, 2.0)
    return s if len(s) >= 30 else c

def upper_half(c):
    if len(c) < 20: return c
    midY = 0.5*(c[:,1].min()+c[:,1].max()); u = c[c[:,1] >= midY]
    return u if len(u) >= 20 else c

def near_face(c, fd=FACE_DEPTH):
    c = upper_half(c)                       # upper half (region mid-Y) first
    if len(c) < 20: return c
    zN = dz(c[:,2],40,30,False); f = c[c[:,2] <= zN+fd]; return f if len(f) >= 20 else c

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

def target_NEW(face, tor):
    """Detector on the near-face + sparse-edge rescue (mirrors ChassisRoi.h).
    Returns (point, rescueCol or None)."""
    p = detect(face, tor)[0]
    if len(face) < 250:
        xE = dense_edge(face[:,0], 40, 8, find_max=not tor)
        if (tor and xE < p[0]-40) or ((not tor) and xE > p[0]+40):
            col = face[np.abs(face[:,0]-xE) <= 60]
            if len(col) >= 12:
                cx = float(np.median(col[:,0])); cz = dense_edge(col[:,2], 25, 6, find_max=False)
                m = (np.abs(face[:,0]-cx) <= 60) & (np.abs(face[:,2]-cz) <= 60)
                cyv = float(np.median(face[m,1])) if m.sum() else p[1]
                return np.array([[cx, cyv, cz]]), col
    return p.reshape(1,3), None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dst', default=r"K:\TUAS\South_Endfeed\GroundTruth_Set\Comparison\XT_target")
    ap.add_argument('--frame', default='')
    a = ap.parse_args()
    n = 0
    for g in R.load_gt(R.GT_ROOT):
        if g['state'] != 'CONLY': continue
        if a.frame and a.frame not in g['frame']: continue
        jp, chs, sprd, spos = R.parse_job(g['job']); sensor = R.frame_sensor(g['frame']); key = R.config_key(sensor, jp, chs)
        pts = R.load_cloud(g['job'], g['frame'])
        if pts is None: continue
        cg, ct, cy = R.PRESETS[key]; tor = R.target_on_right(sensor, jp, chs)
        crop = R.preset_crop(pts, cg, ct)
        cone = E.extract_cone(crop, cg, ct, cy, tor)
        if cone is None or len(cone) < 30: continue
        candBand = crop[(crop[:,1] >= cy-CAND_YBAND) & (crop[:,1] <= cy+CAND_YBAND)]
        cands = E.cone_columns(candBand)
        sel = None; other = []
        if cands:
            sel = (min if tor is False else max)(cands, key=lambda c: c[:,0].mean()) if tor is not None else min(cands, key=lambda c: abs(c[:,0].mean()-cg))
            selx = sel[:,0].mean(); other = [c for c in cands if abs(c[:,0].mean()-selx) > 1e-6]
        gt = np.array([[g['T'], g['H'], g['G']]])
        d = os.path.join(a.dst, 'steps', g['job']); os.makedirs(d, exist_ok=True)
        b = os.path.join(d, g['frame'])
        def w(sfx, arr):
            arr = np.asarray(arr).reshape(-1,3)
            if len(arr): R.write_pcd(b + sfx, arr)
        w('_01_crop.pcd', crop)
        w('_02_candBand.pcd', candBand)
        if sel is not None: w('_03_cand_sel.pcd', sel)
        if other: w('_03_cand_other.pcd', np.vstack(other))
        w('_04_coneRegion.pcd', cone)
        coneClean = clean(cone)                  # SOR flyer removal (production order)
        w('_04b_regionClean.pcd', coneClean)
        w('_05_upperHalf.pcd', upper_half(coneClean))
        face = near_face(coneClean)
        w('_06_nearFace.pcd', face)
        newPt, rescueCol = target_NEW(face, tor)
        if rescueCol is not None: w('_06b_rescueCol.pcd', rescueCol)
        w('_07_tgt_OLD.pcd', detect(cone, tor))
        w('_07_tgt_NEW.pcd', newPt)
        w('_07_tgt_GT.pcd', gt)
        n += 1
    print('wrote step clouds for %d frames under %s' % (n, os.path.join(a.dst, 'steps')))

if __name__ == '__main__':
    main()
