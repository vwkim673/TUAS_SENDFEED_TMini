"""
roi_igh_eval.py - IGH target-position eval harness (FRONT Mount, 2026-07).

IGH target = the GUIDE's top-left closest corner (sensor view) -- HIGHER (H~0) than the
XT plate corner. This harness loads the IGH GT, resolves clouds from the new
Mount_Result/ batch layout, runs a candidate targeting rule, and compares to GT. It also
dumps step PCDs so the guide geometry can be inspected.

GT: GroundTruth/IGH/groundtruth.txt   (job;frame;state;T;H;G, same schema as XT)
Clouds: OriginalData/IGH/FRONT/BatchResult/Mount_Result/<job>/<frame>/<frame>_base_converted.pcd

Usage: python roi_igh_eval.py [--state CONLY] [--dump] [--rule xt|ightop]
"""
import sys, os, glob, argparse, collections, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R
import roi_cone_extract as E
from roi_user_pipeline import new_target, detect, dz, dense_edge

# IGH crop widening: the CS(REAR) jobs have a ~600mm Z spread and targets up to Y~56, LARGER
# than the default fixed crop margins -> near-Z fronts and high targets fell OUTSIDE the crop
# (GT itself dropped). The preset centre is the GT mean; only the margins were too tight.
R.TROLLEY_BACK = 500.0   # near-Z front extent (was 250) -> covers CS guides at Z~1222
R.HEIGHT_MAX   = 100.0   # Y top (was 50) -> covers targets up to ~56

GT_FILE = r"K:\TUAS\South_Endfeed\GroundTruth_Set\GroundTruth\IGH\groundtruth.txt"
GT_ROI_ROOT = r"K:\TUAS\South_Endfeed\GroundTruth_Set\GroundTruth\IGH"   # \<job>\{Chassis,Landed,Separated}\<frame>_*.pcd
BR_ROOT = r"K:\TUAS\South_Endfeed\GroundTruth_Set\OriginalData\IGH"   # + \<spos>\BatchResult\Mount_Result
STATE_SUB = {'CONLY': 'Chassis', 'LANDED': 'Landed', 'LANDING': 'Separated'}

# IGH presets (cg=X-Gantry, ct=Z-Trolley, cy=Y-guide-height) from GT CONLY means.
# Axes: point-cloud X = Gantry (lateral), Y = Height (up +), Z = Trolley (depth, near-Z front).
# GT columns are positionally (Gantry, Height, Trolley) despite the T;H;G header labels.
IGH_PRESETS = {
    'FS_Big_CHS40': (812.0, 1671.0, -4.0),    # FRONT, GT CONLY mean (X=812, Z=1671, Y=-4)
    'FS_Big_CHS45': (847.0, 1643.0, -7.0),    # FRONT
    'CS_Big_CHS40': (848.0, 1660.0, -77.0),   # REAR main cluster (opposite-side guide outlier excluded)
    'CS_Big_CHS45': (797.0, 1673.0, -66.0),   # REAR main cluster
}

# Merged / long-guide configs need a WIDER Gantry(X) crop to include the full ~480mm
# structure (preset cg = one-guide label, not the merged centre, so the far end is cropped
# by the default +-300). Others use R.GANTRY_HALF.
IGH_GANTRY_HALF = {'CS_Big_CHS40': 450.0}

# Merged long guide (~460-500mm) also needs a wider REGION xspan: the default 300mm caps the
# extracted region at +-150 from the merged CENTRE (cx = median of the single merged candidate),
# cropping BOTH ends. Widen so the whole guide is passed on. Others keep 300 (single ~200mm).
IGH_XSPAN = {'CS_Big_CHS40': 600.0}

def br_mount(spos):
    return os.path.join(BR_ROOT, spos, 'BatchResult', 'Mount_Result')

def igh_config(job, frame):
    chs = 45 if 'CHS45' in job else 40
    sensor = 'FS' if '_FS' in frame else ('CS' if '_CS' in frame else ('RS' if '_RS' in frame else '?'))
    return f'{sensor}_Big_CHS{chs}'

def load_igh_gt():
    out = []
    for ln in open(GT_FILE, encoding='utf-8', errors='replace').read().splitlines():
        p = [x.strip() for x in ln.split(';')]
        if len(p) < 6: continue
        try:
            out.append(dict(job=p[0], frame=p[1], state=p[2],
                            T=float(p[3]), H=float(p[4]), G=float(p[5])))
        except ValueError:
            continue
    return out

def moved_jobs(gts, thr=100.0):
    """Jobs where the chassis MOVED (CONLY->LANDED |dX|>thr or |dZ|>thr). The CONLY frame is
    then at the START position (chassis moves to the job position only after the container is
    lowered), so its preset ROI misses the target. These CONLY samples are excluded."""
    by = collections.defaultdict(dict)
    for g in gts:
        by[g['job']][g['state']] = np.array([g['T'], g['H'], g['G']])
    mv = set()
    for job, st in by.items():
        if 'CONLY' in st and 'LANDED' in st:
            d = st['LANDED'] - st['CONLY']
            if abs(d[0]) > thr or abs(d[2]) > thr:
                mv.add(job)
    return mv

def resolve_igh(job, frame):
    spos = 'REAR' if '_REAR_' in job else 'FRONT'
    bm = br_mount(spos)
    exact = os.path.join(bm, job, frame, frame + '_base_converted.pcd')
    if os.path.exists(exact): return exact
    # nearest by timestamp within the job dir
    jd = os.path.join(bm, job)
    if not os.path.isdir(jd): return None
    cand = glob.glob(os.path.join(jd, '*_Image', '*_base_converted.pcd'))
    if not cand: return None
    def ms(pth):
        import re
        m = re.search(r'_(\d{2})(\d{2})(\d{2})_(\d{3})_', os.path.basename(pth))
        return ((int(m.group(1))*60+int(m.group(2)))*60+int(m.group(3)))*1000+int(m.group(4)) if m else 0
    import re
    m = re.search(r'_(\d{2})(\d{2})(\d{2})_(\d{3})_', frame)
    tgt = ((int(m.group(1))*60+int(m.group(2)))*60+int(m.group(3)))*1000+int(m.group(4)) if m else 0
    return min(cand, key=lambda c: abs(ms(c)-tgt))

def load_cloud_igh(job, frame):
    pth = resolve_igh(job, frame)
    if pth is None: return None
    p = R.read_pcd(pth)
    if p is None or len(p) == 0: return None
    p = p[~np.isnan(p).any(axis=1)]
    p = p[(p[:,0] != 0) | (p[:,1] != 0) | (p[:,2] != 0)]
    return p if len(p) else None

# --- IGH targeting rule: reuse XT detector for (x,z), override Y to the GUIDE TOP ---
# The XT corner detector lands the right lateral/depth corner (guide post) but its
# local-median height sits ~90mm below the guide top; the IGH target is the guide's
# TOP-left corner. So take the same (x,z) and set Y = dense-max Y in that X column.
def target_ightop(region, tor, xcol=60.0):
    p = new_target(region, tor)                          # XT pipeline: good (x,z) [dT11/dG15]
    c = E.sor(region, 12, 2.0)
    if len(c) >= 30: region = c
    col = region[np.abs(region[:,0] - p[0]) <= xcol]
    if len(col) >= 10:
        p[1] = float(dz(col[:,1], 40, 10, True))         # override Y -> guide top at corner column
    return p

def target_ighface(crop, cy, tor, sensor='FS', zguide=100.0, zregion=300.0,
                   yUp=50.0, yDown=210.0, single_w=220.0, merge_thr=350.0, ret_dbg=None):
    """IGH guide-face rule (user model 2026-07). Geometry (from site):
      - The Z-SIDE guide reads as a rectangle from the sensor; a SINGLE guide is ~200mm in
        X, a MERGED 40-chassis+20-guide (CS) is ~480mm in X.
      - The guide is TILTED in Y-Z: its TOP is nearest-Z (e.g. Z 1835) and the bottom is
        ~+100mm deeper (Z 1935); the guide is ~200mm tall in Y and ~100mm deep in Z.
      - Behind the guide the PLATE adds ~200mm more Z. So from the guide-top near-Z we
        extract Z + zregion(300) = guide(100) + plate(200) as the region.
    The target = guide TOP-LEFT corner = near-Z (guide top), target-X edge, guide-top Y.
    Axes: X = Gantry, Y = Height, Z = Trolley."""
    # guide-height band: cy is the guide TOP, so the ~200mm-tall guide extends DOWN from it.
    b = crop[(crop[:,1] <= cy + yUp) & (crop[:,1] >= cy - yDown)]
    if len(b) < 60: return None
    b = E.sor(b, 12, 2.0) if len(b) >= 60 else b
    zF = dz(b[:,2], 40, 30, False)                         # near-Z front = guide TOP
    guide = b[b[:,2] <= zF + zguide]                       # the tilted guide (front ~100mm, band Y)
    if len(guide) < 25: return None
    # _region = guide + plate: extract from the FULL crop (the plate sits BELOW the guide in
    # Y, so the guide-height band would clip it), Z from guide-top to +zregion(300).
    region = crop[(crop[:,2] >= zF - 10) & (crop[:,2] <= zF + zregion)]
    # MERGED (40-chassis + 20-guide connected as one, ~480mm X) vs SINGLE/SEPARATED
    # (~200mm). Site rule: separated -> Big=left guide, Small=right guide; MERGED -> target
    # the WHOLE structure and apply a targeting offset (GT labels a single guide there, so
    # a merged frame is NOT scored against GT). tor already encodes the side for split cases.
    xlo, xhi = np.percentile(guide[:,0], [3, 97])
    merged = (xhi - xlo) > merge_thr
    if not merged and (xhi - xlo) > single_w + 60:
        # borderline two-guide but with a gap -> separated: keep the target-side single guide
        if tor: guide = guide[guide[:,0] >= xhi - single_w]
        else:   guide = guide[guide[:,0] <= xlo + single_w]
        if len(guide) < 20: return None
    # Corner via the robust shared XT detector (near-Z + target-X), then override the
    # height to the guide TOP at that corner column.
    p = detect(guide, tor)
    col = guide[np.abs(guide[:,0] - p[0]) <= 60]
    if len(col) >= 8: p[1] = float(dz(col[:,1], 40, 8, True))
    if ret_dbg is not None: ret_dbg.update(face=guide, region=region, zF=zF, xspan=xhi-xlo, merged=merged)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default='CONLY')
    ap.add_argument('--spos', default='BOTH', choices=['FRONT', 'REAR', 'BOTH'])
    ap.add_argument('--rule', default='ightop', choices=['xt', 'ightop', 'ighface'])
    ap.add_argument('--dump', action='store_true')
    ap.add_argument('--keepmoved', action='store_true', help='do NOT exclude chassis-moved jobs')
    ap.add_argument('--dst', default=r"K:\TUAS\South_Endfeed\GroundTruth_Set\Comparison\IGH_target")
    a = ap.parse_args()
    gts = load_igh_gt()
    moved = moved_jobs(gts) if not a.keepmoved else set()
    res = collections.defaultdict(list); miss = []; excl = 0
    for g in gts:
        if a.spos != 'BOTH' and ('_' + a.spos + '_') not in g['job']: continue
        if a.state != 'ALL' and g['state'] != a.state: continue
        if g['job'] in moved: excl += 1; continue   # chassis moved -> CONLY at start pos, skip
        key = igh_config(g['job'], g['frame'])
        if key not in IGH_PRESETS: continue
        pts = load_cloud_igh(g['job'], g['frame'])
        if pts is None: miss.append((key, 'no-cloud', g['frame'][9:15])); continue
        cg, ct, cy = IGH_PRESETS[key]
        # X = Gantry; side-rule via the shared cpp convention (CS->Center camera + Big -> right).
        sensor = 'FS' if '_FS' in g['frame'] else ('CS' if '_CS' in g['frame'] else 'RS')
        chs = 45 if 'CHS45' in g['job'] else 40
        tor = R.target_on_right(sensor, 'Big', chs)
        gh0 = R.GANTRY_HALF; R.GANTRY_HALF = IGH_GANTRY_HALF.get(key, gh0)   # wider crop for merged
        crop = R.preset_crop(pts, cg, ct)
        R.GANTRY_HALF = gh0
        dbg = {}
        if a.rule == 'ighface':
            pt = target_ighface(crop, cy, tor, sensor, ret_dbg=dbg)
            region = crop   # for --dump, show the whole crop as context
            if pt is None: miss.append((key, 'no-face', g['frame'][9:15])); continue
        else:
            region = E.extract_cone(crop, cg, ct, cy, tor, xspan=IGH_XSPAN.get(key, 300.0))
            if region is None or len(region) < 30: miss.append((key, 'no-region', g['frame'][9:15])); continue
            pt = new_target(region, tor) if a.rule == 'xt' else target_ightop(region, tor)
        gt = np.array([g['T'], g['H'], g['G']])
        # MERGED frames: GT labels one of the two joined guides -> not a valid single-corner
        # reference. Bucket them separately (target = whole + offset, scored on consistency).
        bucket = key + (' [MERGED]' if dbg.get('merged') else '')
        res[bucket].append((g['frame'][9:15], pt - gt))
        if a.dump:
            d = os.path.join(a.dst, g['job']); os.makedirs(d, exist_ok=True)
            b = os.path.join(d, g['frame'])
            R.write_pcd(b + '_presetCrop.pcd', crop)                      # wide preset crop (context)
            # _Region = the extracted ROI we PASS ON to the detector (guide+plate); falls back to crop
            R.write_pcd(b + '_Region.pcd', dbg['region'] if 'region' in dbg else region)
            if 'face' in dbg: R.write_pcd(b + '_face.pcd', dbg['face'])   # the guide we actually targeted
            R.write_pcd(b + '_tgt.pcd', pt.reshape(1, 3))                 # our target point
            R.write_pcd(b + '_gt.pcd', gt.reshape(1, 3))
            # copy the worker-annotated GT ROI cloud for direct comparison
            gtsrc = os.path.join(GT_ROI_ROOT, g['job'], STATE_SUB.get(g['state'], 'Chassis'), g['frame'] + '_' + g['state'] + '.pcd')
            if not os.path.exists(gtsrc):
                cand = glob.glob(os.path.join(GT_ROI_ROOT, g['job'], '*', g['frame'] + '_*.pcd'))
                gtsrc = cand[0] if cand else None
            if gtsrc and os.path.exists(gtsrc): shutil.copy(gtsrc, b + '_GTroi.pcd')
    print('rule=%s state=%s' % (a.rule, a.state))
    print('%-14s | n | dT dH dG (median) | within20 within30' % 'config')
    allv = []
    for k in sorted(res):
        v = res[k]; allv += [x[1] for x in v]
        md = lambda i: np.median([x[1][i] for x in v])
        w20 = sum(1 for _, d in v if max(abs(d)) <= 20); w30 = sum(1 for _, d in v if max(abs(d)) <= 30)
        print('%-14s | %d | %4.0f %4.0f %4.0f | %d/%d %d' % (k, len(v), md(0), md(1), md(2), w20, len(v), w30))
    if allv:
        print('ALL median|err| dT %.0f dH %.0f dG %.0f | within20 %d/%d w30 %d w40 %d' % (
            np.median([abs(x[0]) for x in allv]), np.median([abs(x[1]) for x in allv]), np.median([abs(x[2]) for x in allv]),
            sum(1 for x in allv if max(abs(x)) <= 20), len(allv),
            sum(1 for x in allv if max(abs(x)) <= 30), sum(1 for x in allv if max(abs(x)) <= 40)))
    if excl: print('excluded (chassis moved CONLY->LANDED):', excl, 'frames from', len(moved), 'jobs')
    if miss: print('skipped:', miss)

if __name__ == '__main__':
    main()
