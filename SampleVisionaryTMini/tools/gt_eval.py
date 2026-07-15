"""
gt_eval.py - Evaluate chassis_roi_detect_cone() against human ground truth.

The GT set gives, per (job,frame,state): a hand-cropped cone-body PCD and a
hand-picked target point (T,H,G). This replicates the C++ detector's
post-extraction step (dense nearest-Z -> thin slab -> slab centroid) on each
cropped body and reports the error vs the GT target.

Because the GT crop is already a tight cone body, this isolates the
slab->centroid target-point logic from the ROI range-selection step.

Usage: python gt_eval.py <gt_root>
"""
import sys, os, struct
import numpy as np

# ---- PCD reader (LZF binary_compressed / binary / ascii) -------------------
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
    stride = sum(s*c for s, c in zip(sizes, counts))   # bytes per point
    # byte offset of each field within a point record
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
        pts = []
        for ln in body.decode('ascii','replace').splitlines():
            p = ln.split()
            if len(p) >= 3: pts.append([float(p[0]),float(p[1]),float(p[2])])
        return np.array(pts)

def valid_pts(path):
    pts = read_pcd(path)
    return pts[~((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))]

# ---- replicate chassis_roi_detail::denseExtreme (low end) ------------------
def dense_min(vals, window=40.0, min_pts=30):
    v = np.sort(vals)
    if len(v) < min_pts: return v[0]
    lo = 0; hi = 0; nn = len(v)
    while lo < nn:
        while hi < nn-1 and v[hi+1]-v[lo] <= window: hi += 1
        if hi-lo+1 >= min_pts: return float(v[lo:hi+1].mean())
        lo += 1
        if hi < lo: hi = lo
    return float(v[0])

# ---- replicate chassis_roi_detect_cone ------------------------------------
def detect_cone(pts, zSlab=100.0, zPad=30.0, minConePts=30):
    if len(pts) < minConePts: return None, 0
    zNear = dense_min(pts[:,2], 40.0, minConePts)
    slab = pts[(pts[:,2] >= zNear-zPad) & (pts[:,2] <= zNear+zSlab)]
    if len(slab) < minConePts: return None, len(slab)
    return slab.mean(axis=0), len(slab)

# ---- load GT table ---------------------------------------------------------
def load_gt(root):
    gt = {}
    with open(os.path.join(root,'groundtruth.txt'),encoding='utf-8') as f:
        for ln in f.read().splitlines()[1:]:
            c = [x.strip() for x in ln.split(';')]
            if len(c) < 6: continue
            job, frame, state, X, Y, Z = c[0], c[1], c[2], c[3], c[4], c[5]
            gt[frame] = dict(job=job, state=state,
                             T=float(X), H=float(Y), G=float(Z))
    return gt

STATE_DIR = {'CONLY':'Chassis','LANDING':'Separated','LANDED':'Landed'}

def main():
    root = sys.argv[1] if len(sys.argv)>1 else \
        r"K:\TUAS\South_Endfeed\GroundTruth_Set\groundtruth_20260619"
    gt = load_gt(root)
    rows = []
    for frame, g in gt.items():
        sub = STATE_DIR.get(g['state'],'')
        # the cropped file is named <frame>_<STATE>.pcd  (note 1 file uses 'image')
        jobdir = os.path.join(root, g['job'], sub)
        if not os.path.isdir(jobdir): continue
        target = None
        for fn in os.listdir(jobdir):
            if fn.lower().startswith(frame.lower()) and fn.lower().endswith('.pcd'):
                target = os.path.join(jobdir, fn); break
        if not target: continue
        pts = valid_pts(target)
        cone, nslab = detect_cone(pts)
        if cone is None:
            rows.append((g['job'], frame, g['state'], None, None, None, None,
                         len(pts), nslab, g)); continue
        dT = cone[0]-g['T']; dH = cone[1]-g['H']; dG = cone[2]-g['G']
        rows.append((g['job'], frame, g['state'], cone[0], cone[1], cone[2],
                     (dT,dH,dG), len(pts), nslab, g))
    # ---- report ----
    print(f"{'state':8} {'config':28} {'frame':32} "
          f"{'GT_T':>7} {'det_T':>7} {'dT':>6} {'dH':>6} {'dG':>6} {'pts':>5} {'slab':>5}")
    aggs = {}
    for job, frame, state, cT, cH, cG, d, npts, nslab, g in sorted(rows, key=lambda r:(r[2],r[0])):
        cfg = derive_cfg(job, frame)
        if d is None:
            print(f"{state:8} {cfg:28} {frame:32} {g['T']:7.0f} {'--':>7} (no slab) pts={npts}")
            continue
        dT,dH,dG = d
        print(f"{state:8} {cfg:28} {frame:32} {g['T']:7.0f} {cT:7.0f} "
              f"{dT:6.0f} {dH:6.0f} {dG:6.0f} {npts:5d} {nslab:5d}")
        aggs.setdefault(state, []).append((dT,dH,dG))
        aggs.setdefault('ALL', []).append((dT,dH,dG))
    print("\n=== |error| medians by state (det - GT) ===")
    print(f"{'group':10} {'n':>4} {'|dT|med':>8} {'|dT|p95':>8} {'|dH|med':>8} {'|dG|med':>8} "
          f"{'dT_bias':>8} {'dH_bias':>8} {'dG_bias':>8}")
    for k in ['CONLY','LANDING','LANDED','ALL']:
        if k not in aggs: continue
        a = np.array(aggs[k])
        ab = np.abs(a)
        print(f"{k:10} {len(a):4d} {np.median(ab[:,0]):8.0f} {np.percentile(ab[:,0],95):8.0f} "
              f"{np.median(ab[:,1]):8.0f} {np.median(ab[:,2]):8.0f} "
              f"{np.median(a[:,0]):8.0f} {np.median(a[:,1]):8.0f} {np.median(a[:,2]):8.0f}")

def derive_cfg(job, frame):
    # sensor from frame token FS/CS/RS
    s = 'FS' if '_FS' in frame else ('CS' if '_CS' in frame else ('RS' if '_RS' in frame else '??'))
    parts = job.split('_')
    jp = next((p for p in parts if p in ('Center','Big','Small')), '?')
    chs = next((p for p in parts if p.startswith('CHS')), '?')
    return f"{s} {jp} {chs}"

if __name__ == '__main__':
    main()
