"""
roi_igh_sheet.py - build a 3-way overview HTML from the roi_igh_eval --dump PCDs.

Reads <dst>/<job>/<frame>_{region,tgt,gt}.pcd and renders, per frame, X-Y (front) and
X-Z (top) cross-sections: region points gray, GT red o, our target blue x. Grouped by
config with per-frame dT/dH/dG. Axes: X = Gantry, Y = Height, Z = Trolley.

Usage: python roi_igh_sheet.py [--dst <dir>]
"""
import sys, os, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R

def cfg_of(job):
    chs = 45 if 'CHS45' in job else 40
    sensor = 'CS' if '_REAR_' in job else 'FS'   # REAR Big -> CS, FRONT Big -> FS
    return f'{sensor}_Big_CHS{chs}'

def svg(pts, a, b, gt, tg, xlab, ylab, W=300, H=240, pad=26):
    P = pts[:, [a, b]]
    xs = np.concatenate([P[:,0], [gt[0], tg[0]]]); ys = np.concatenate([P[:,1], [gt[1], tg[1]]])
    x0, x1 = xs.min()-20, xs.max()+20; y0, y1 = ys.min()-20, ys.max()+20
    if x1-x0 < 1: x1 = x0+1
    if y1-y0 < 1: y1 = y0+1
    sx = lambda x: pad + (x-x0)/(x1-x0)*(W-2*pad)
    sy = lambda y: H-pad - (y-y0)/(y1-y0)*(H-2*pad)
    s = '<svg width="%d" height="%d" style="background:#111;border:1px solid #333">' % (W, H)
    step = max(1, len(P)//500)
    s += ''.join('<circle cx="%.1f" cy="%.1f" r="1.1" fill="#889"/>' % (sx(x), sy(y)) for x, y in P[::step])
    gx, gy = sx(gt[0]), sy(gt[1]); s += '<circle cx="%.1f" cy="%.1f" r="7" fill="none" stroke="#f44" stroke-width="2"/>' % (gx, gy)
    tx, ty = sx(tg[0]), sy(tg[1])
    s += '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#0cf" stroke-width="2"/>' % (tx-5, ty-5, tx+5, ty+5)
    s += '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#0cf" stroke-width="2"/>' % (tx-5, ty+5, tx+5, ty-5)
    s += '<text x="4" y="14" fill="#888" font-size="11">%s (w=%.0fmm)</text>' % (xlab+'/'+ylab, x1-x0)
    return s + '</svg>'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dst', default=r"K:\TUAS\South_Endfeed\GroundTruth_Set\Comparison\IGH_target")
    a = ap.parse_args()
    cards = {}
    for reg_pcd in glob.glob(os.path.join(a.dst, '*', '*_region.pcd')):
        base = reg_pcd[:-len('_region.pcd')]
        gt_pcd, tg_pcd = base + '_gt.pcd', base + '_tgt.pcd'
        if not (os.path.exists(gt_pcd) and os.path.exists(tg_pcd)): continue
        reg = R.read_pcd(reg_pcd); gt = R.read_pcd(gt_pcd); tg = R.read_pcd(tg_pcd)
        if reg is None or len(reg) < 5 or gt is None or tg is None: continue
        gt = gt[0]; tg = tg[0]; job = os.path.basename(os.path.dirname(reg_pcd))
        key = cfg_of(job); fr = os.path.basename(base)[9:15]
        d = tg - gt
        xy = svg(reg, 0, 1, gt[[0,1]], tg[[0,1]], 'X', 'Y')   # front (Gantry vs Height)
        xz = svg(reg, 0, 2, gt[[0,2]], tg[[0,2]], 'X', 'Z')   # top (Gantry vs Trolley)
        cards.setdefault(key, []).append((fr, d, xy, xz))
    h = ['<html><head><meta charset="utf-8"><style>',
         'body{background:#0a0a0a;color:#ddd;font-family:Segoe UI,Arial;margin:16px}',
         'h2{margin:18px 0 6px}.sub{color:#888;font-size:12px;margin-bottom:6px}',
         '.card{display:inline-block;vertical-align:top;margin:6px;padding:6px;background:#161616;border:1px solid #2a2a2a;border-radius:6px}',
         '.h{font-size:12px;margin-bottom:4px}b{color:#fff}</style></head><body>']
    h.append('<h2>IGH target — <span style="color:#f44">o GT (guide top-left)</span> &nbsp; <span style="color:#0cf">x OUR</span> &nbsp; gray = region</h2>')
    h.append('<div class="sub">X = Gantry · Y = Height · Z = Trolley. Left panel = front (X-Y), right = top (X-Z).</div>')
    for key in sorted(cards):
        rows = sorted(cards[key], key=lambda r: r[0])
        med = np.median(np.abs([r[1] for r in rows]), axis=0)
        h.append('<h2>%s <span class="sub">n=%d · median|dT/dH/dG| = %.0f/%.0f/%.0f mm</span></h2>' % (key, len(rows), med[0], med[1], med[2]))
        for fr, d, xy, xz in rows:
            bad = 'color:#e55' if max(abs(d)) > 40 else 'color:#4c4' if max(abs(d)) <= 20 else 'color:#cc4'
            h.append('<div class="card"><div class="h"><b>%s</b> <span style="%s">dT%+.0f dH%+.0f dG%+.0f</span></div>%s%s</div>' % (fr, bad, d[0], d[1], d[2], xy, xz))
    h.append('</body></html>')
    out = os.path.join(a.dst, 'index.html')
    with open(out, 'w', encoding='utf-8') as f: f.write('\n'.join(h))
    print('wrote', out, '(%d frames)' % sum(len(v) for v in cards.values()))

if __name__ == '__main__':
    main()
