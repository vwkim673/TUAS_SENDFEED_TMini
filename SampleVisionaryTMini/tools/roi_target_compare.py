"""
roi_target_compare.py - 3-way comparison sheet: OLD vs NEW target position vs GT.

For every CONLY frame (preset ROI) it extracts the cone region, then computes the target
corner two ways and compares to GT:
  OLD = pc_cone_detection replica on the FULL cone+plate region (current C++ behaviour)
  NEW = same detector on the NEAR-Z FACE only (Part B: front slab; back body excluded)
Renders one self-contained index.html: a 3-way error table (OLD/NEW dT,dH,dG, colour-coded)
plus per-frame X-Y (front) and X-Z (top) cross-sections with GT (red o), OLD (yellow +),
NEW (cyan x) overlaid. Inline SVG, numpy only.

Usage: python roi_target_compare.py [--dst <dir>]   (default: Comparison/XT_target)
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R
import roi_cone_extract as E

def dz(v, w=40, m=30, mx=False): return R.dense_extreme(np.asarray(v), w, m, find_max=mx)

def detect(cone, tor):
    """pc_cone_detection_naive replica: densest-slice corner (x,z) + local median-Y."""
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
    cyv2 = float(np.median(cone[m,1])) if m.sum() else float(p[1])
    return np.array([cx, cyv2, cz])

def target_OLD(cone, tor): return detect(cone, tor)

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

def target_NEW(cone, tor, fd=80):
    c = E.sor(cone, 12, 2.0)                              # isolated-flyer removal
    if len(c) >= 30: cone = c
    midY = 0.5*(cone[:,1].min()+cone[:,1].max())         # upper half (region mid-Y)
    up = cone[cone[:,1] >= midY]
    if len(up) < 20: up = cone
    zN = dz(up[:,2],40,30,False); face = up[up[:,2] <= zN+fd]   # + near-Z face
    if len(face) < 20: face = up
    p = detect(face, tor)
    # sparse-face X-edge rescue (mirrors ChassisRoi.h chassis_roi_sparse_edge_rescue)
    if len(face) < 250:
        xE = dense_edge(face[:,0], 40, 8, find_max=not tor)
        if (tor and xE < p[0]-40) or ((not tor) and xE > p[0]+40):
            col = face[np.abs(face[:,0]-xE) <= 60]
            if len(col) >= 12:
                cx = float(np.median(col[:,0])); cz = dense_edge(col[:,2], 25, 6, find_max=False)
                m = (np.abs(face[:,0]-cx) <= 60) & (np.abs(face[:,2]-cz) <= 60)
                cyv = float(np.median(face[m,1])) if m.sum() else p[1]
                p = np.array([cx, cyv, cz])
    return p

def svg(pts, a, b, gt, old, new, xlab, ylab, W=300, H=240, pad=26):
    P = pts[:, [a, b]]
    xs = np.concatenate([P[:,0], [gt[0], old[0], new[0]]]); ys = np.concatenate([P[:,1], [gt[1], old[1], new[1]]])
    x0, x1 = xs.min()-20, xs.max()+20; y0, y1 = ys.min()-20, ys.max()+20
    if x1-x0 < 1: x1 = x0+1
    if y1-y0 < 1: y1 = y0+1
    def sx(x): return pad + (x-x0)/(x1-x0)*(W-2*pad)
    def sy(y): return H-pad - (y-y0)/(y1-y0)*(H-2*pad)
    s = '<svg width="%d" height="%d" style="background:#111;border:1px solid #333">' % (W, H)
    s += ''.join('<circle cx="%.1f" cy="%.1f" r="1.1" fill="#889"/>' % (sx(x), sy(y)) for x, y in P)
    gx, gy = sx(gt[0]), sy(gt[1]); s += '<circle cx="%.1f" cy="%.1f" r="7" fill="none" stroke="#f44" stroke-width="2"/>' % (gx, gy)
    ox, oy = sx(old[0]), sy(old[1])
    s += '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#dd0" stroke-width="2"/>' % (ox-6, oy, ox+6, oy)
    s += '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#dd0" stroke-width="2"/>' % (ox, oy-6, ox, oy+6)
    nx, ny = sx(new[0]), sy(new[1])
    s += '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#0cf" stroke-width="2"/>' % (nx-5, ny-5, nx+5, ny+5)
    s += '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#0cf" stroke-width="2"/>' % (nx-5, ny+5, nx+5, ny-5)
    s += '<text x="4" y="14" fill="#888" font-size="11">%s (w=%.0fmm)</text>' % (xlab+'/'+ylab, x1-x0)
    return s + '</svg>'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dst', default=r"K:\TUAS\South_Endfeed\GroundTruth_Set\Comparison\XT_target")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    rows = []
    for g in R.load_gt(R.GT_ROOT):
        if g['state'] != 'CONLY': continue
        jp, chs, sprd, spos = R.parse_job(g['job']); sensor = R.frame_sensor(g['frame']); key = R.config_key(sensor, jp, chs)
        pts = R.load_cloud(g['job'], g['frame'])
        if pts is None: continue
        cg, ct, cy = R.PRESETS[key]; tor = R.target_on_right(sensor, jp, chs)
        cone = E.extract_cone(R.preset_crop(pts, cg, ct), cg, ct, cy, tor)
        if cone is None or len(cone) < 30: continue
        gt = np.array([g['T'], g['H'], g['G']]); old = target_OLD(cone, tor); new = target_NEW(cone, tor)
        reg = cone if len(cone) <= 600 else cone[np.linspace(0, len(cone)-1, 600).astype(int)]
        xy = svg(reg, 0, 1, gt[[0,1]], old[[0,1]], new[[0,1]], 'X', 'Y')
        xz = svg(reg, 0, 2, gt[[0,2]], old[[0,2]], new[[0,2]], 'X', 'Z')
        rows.append(dict(key=key, frame=g['frame'][9:15], tor=tor, do=old-gt, dn=new-gt, xy=xy, xz=xz))
    rows.sort(key=lambda r: (r['key'], r['frame']))
    def stat(sel): return (np.median([abs(r[sel][0]) for r in rows]), np.median([abs(r[sel][1]) for r in rows]), np.median([abs(r[sel][2]) for r in rows]))
    oT,oH,oG = stat('do'); nT,nH,nG = stat('dn')
    def col(v): return '#4c4' if abs(v) < 25 else ('#cc4' if abs(v) < 60 else '#e55')
    h = ['<html><head><meta charset="utf-8"><style>',
         'body{background:#0a0a0a;color:#ddd;font-family:Segoe UI,Arial;margin:16px}',
         'table{border-collapse:collapse;margin-bottom:16px}td,th{border:1px solid #333;padding:3px 7px;font-size:12px}',
         '.card{display:inline-block;vertical-align:top;margin:6px;padding:6px;background:#161616;border:1px solid #2a2a2a;border-radius:6px}',
         '.h{font-size:12px;margin-bottom:4px}b{color:#fff}</style></head><body>']
    h.append('<h2>Target: <span style="color:#f44">o GT</span> &nbsp; <span style="color:#dd0">+ OLD(full region)</span> &nbsp; <span style="color:#0cf">x NEW(near-Z face)</span></h2>')
    h.append('<p>n=%d &nbsp; median |err| mm &nbsp; <b>OLD</b> dT %.0f/dH %.0f/dG %.0f &nbsp;&nbsp; <b>NEW</b> dT %.0f/dH %.0f/dG %.0f</p>' % (len(rows), oT,oH,oG, nT,nH,nG))
    h.append('<table><tr><th>config</th><th>frame</th><th colspan=3>OLD dT/dH/dG</th><th colspan=3>NEW dT/dH/dG</th></tr>')
    for r in rows:
        h.append('<tr><td>%s</td><td>%s</td>' % (r['key'], r['frame'])
                 + ''.join('<td style="color:%s;text-align:right">%+.0f</td>' % (col(r['do'][i]), r['do'][i]) for i in range(3))
                 + ''.join('<td style="color:%s;text-align:right">%+.0f</td>' % (col(r['dn'][i]), r['dn'][i]) for i in range(3))
                 + '</tr>')
    h.append('</table>')
    for r in rows:
        h.append('<div class="card"><div class="h">%s <b>%s</b> tor=%s &nbsp; OLD %+.0f,%+.0f,%+.0f &nbsp; NEW %+.0f,%+.0f,%+.0f</div>%s%s</div>'
                 % (r['key'], r['frame'], r['tor'], *r['do'], *r['dn'], r['xy'], r['xz']))
    h.append('</body></html>')
    out = os.path.join(a.dst, 'index.html')
    with open(out, 'w', encoding='utf-8') as f: f.write('\n'.join(h))
    print('wrote', out, '(%d frames)  OLD dG=%.0f  NEW dG=%.0f' % (len(rows), oG, nG))

if __name__ == '__main__':
    main()
