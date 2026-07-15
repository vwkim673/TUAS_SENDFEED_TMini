"""
roi_igh_cpp_eval.py - compare the C++ FORCE_ROI batch IGH target vs GT (2026-07).

Companion to roi_igh_eval.py. That harness runs the PYTHON targeting rule on the cloud;
this one reads the point the SHIPPED C++ chain wrote during a FORCE_ROI_MODE=1 batch run,
so it measures the production output (igh_target_from_region + the INI IGH_ROI_* crop)
against ground truth.

Source = the per-job `<job>_res.txt` (NOT pcd files). In FORCE_ROI mode each res.txt gains
the columns  ROI_State;ROI_T;ROI_H;ROI_G;ROI_CropPts;AB_Valid;AB_dT;AB_dH;AB_dG , where
ROI_T;ROI_H;ROI_G = _roi_detected.x;y;z = (Gantry, Height, Trolley) mm -- and for IGH
_roi_detected IS the igh_target_from_region output. Columns are located by NAME, so this
works regardless of how many trailing columns the build has.

NOTE: a res.txt WITHOUT ROI_T means that run was FORCE_ROI_MODE=0 (ROI_FALLBACK / production)
-- the target logic only fires on VA-fail frames there, so it cannot be scored vs GT. Such
jobs are reported under 'no-ROI-cols'. Re-run with FORCE_ROI_MODE=1 to validate.

The metric block mirrors roi_igh_eval exactly (dT/dH/dG = err[0]/[1]/[2] = Gantry/Height/
Trolley medians; within20/30), so the two sheets line up 1:1.

Usage:
  python roi_igh_cpp_eval.py [--spos BOTH] [--state CONLY] [--keepmoved]
                             [--br-root <...\\OriginalData\\IGH>] [--csv out.csv]
"""
import sys, os, glob, re, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_igh_eval as G

SENTINEL = -10000.0


def ms_of(name):
    """Capture time (ms since midnight) from a '_HHMMSS_mmm_' name; None if absent."""
    m = re.search(r'_(\d{2})(\d{2})(\d{2})_(\d{3})_', name)
    if not m:
        return None
    return ((int(m.group(1)) * 60 + int(m.group(2))) * 60 + int(m.group(3))) * 1000 + int(m.group(4))


def load_res_index(br_root, spos):
    """{job: [dict(frame, ms, T, H, G)]} from every FORCE_ROI <job>_res.txt under
    <br_root>/<spos>/BatchResult. Also returns the set of jobs whose res.txt lacked ROI cols."""
    root = os.path.join(br_root, spos, 'BatchResult')
    idx, noroi = {}, set()
    for res in glob.glob(os.path.join(root, '*', '*_res.txt')):
        job = os.path.basename(os.path.dirname(res))
        try:
            lines = open(res, encoding='utf-8', errors='replace').read().splitlines()
        except OSError:
            continue
        if not lines:
            continue
        hdr = [h.strip() for h in lines[0].split(';')]
        col = {n: (hdr.index(n) if n in hdr else -1) for n in ('Filename', 'ROI_T', 'ROI_H', 'ROI_G')}
        if min(col.values()) < 0:
            noroi.add(job)
            continue
        rows = []
        need = max(col.values())
        for ln in lines[1:]:
            p = ln.split(';')
            if len(p) <= need:
                continue
            try:
                T = float(p[col['ROI_T']]); H = float(p[col['ROI_H']]); Gg = float(p[col['ROI_G']])
            except ValueError:
                continue
            fn = p[col['Filename']].strip()
            rows.append(dict(frame=fn, ms=ms_of(fn), T=T, H=H, G=Gg))
        idx[job] = rows
    return idx, noroi


def match_row(rows, frame):
    """The res row for `frame`: exact Filename match, else nearest capture timestamp."""
    for r in rows:
        if r['frame'] == frame:
            return r
    tgt = ms_of(frame)
    cand = [r for r in rows if r['ms'] is not None]
    if tgt is None or not cand:
        return None
    return min(cand, key=lambda r: abs(r['ms'] - tgt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--br-root', default=G.BR_ROOT, help=r'...\OriginalData\IGH (has <spos>\BatchResult)')
    ap.add_argument('--spos', default='BOTH', choices=['FRONT', 'REAR', 'BOTH'])
    ap.add_argument('--state', default='CONLY', help='GT state (CONLY/LANDED/LANDING/ALL)')
    ap.add_argument('--keepmoved', action='store_true', help='do NOT exclude chassis-moved jobs')
    ap.add_argument('--csv', default='', help='optional per-frame CSV dump')
    a = ap.parse_args()

    spos_list = ['FRONT', 'REAR'] if a.spos == 'BOTH' else [a.spos]
    index, noroi = {}, set()
    for sp in spos_list:
        i, nr = load_res_index(a.br_root, sp)
        index.update(i); noroi |= nr

    gts = G.load_igh_gt()
    moved = G.moved_jobs(gts) if not a.keepmoved else set()
    res = collections.defaultdict(list)
    miss = collections.Counter()
    excl = 0
    rows = []
    for g in gts:
        if a.spos != 'BOTH' and ('_' + a.spos + '_') not in g['job']:
            continue
        if a.state != 'ALL' and g['state'] != a.state:
            continue
        if g['job'] in moved:
            excl += 1
            continue
        key = G.igh_config(g['job'], g['frame'])
        if key not in G.IGH_PRESETS:
            continue
        if g['job'] not in index:
            miss['no-ROI-cols' if g['job'] in noroi else 'no-res'] += 1
            continue
        r = match_row(index[g['job']], g['frame'])
        if r is None:
            miss['no-frame'] += 1
            continue
        if r['T'] == SENTINEL or r['H'] == SENTINEL or r['G'] == SENTINEL:
            miss['roi-sentinel'] += 1
            continue
        gt = np.array([g['T'], g['H'], g['G']], dtype=float)
        pt = np.array([r['T'], r['H'], r['G']], dtype=float)
        d = pt - gt
        res[key].append((g['frame'][9:15], d))
        rows.append((key, g['job'], g['frame'], r['frame'], pt[0], pt[1], pt[2],
                     gt[0], gt[1], gt[2], d[0], d[1], d[2]))

    print('C++ ROI target (res.txt ROI_T/H/G) vs GT   state=%s spos=%s' % (a.state, a.spos))
    print('  dT/dH/dG = median err  Gantry/Height/Trolley (mm)')
    print('%-16s | n | dT   dH   dG | within20 w30' % 'config')
    allv = []
    for k in sorted(res):
        v = res[k]
        allv += [x[1] for x in v]
        md = lambda i: np.median([x[1][i] for x in v])
        w20 = sum(1 for _, d in v if max(abs(d)) <= 20)
        w30 = sum(1 for _, d in v if max(abs(d)) <= 30)
        print('%-16s | %d | %4.0f %4.0f %4.0f | %d/%d %d' % (k, len(v), md(0), md(1), md(2), w20, len(v), w30))
    if allv:
        print('ALL median|err| dT %.0f dH %.0f dG %.0f | within20 %d/%d w30 %d w40 %d' % (
            np.median([abs(x[0]) for x in allv]), np.median([abs(x[1]) for x in allv]),
            np.median([abs(x[2]) for x in allv]),
            sum(1 for x in allv if max(abs(x)) <= 20), len(allv),
            sum(1 for x in allv if max(abs(x)) <= 30), sum(1 for x in allv if max(abs(x)) <= 40)))
    if excl:
        print('excluded (chassis moved CONLY->LANDED):', excl, 'frames')
    if miss:
        print('skipped:', dict(miss))
    if noroi:
        print('jobs with NO ROI columns (FORCE_ROI_MODE=0 run):', len(noroi))

    if a.csv and rows:
        with open(a.csv, 'w', encoding='utf-8') as f:
            f.write('config;gtJob;gtFrame;resFrame;cppGantry;cppHeight;cppTrolley;'
                    'gtGantry;gtHeight;gtTrolley;dGantry;dHeight;dTrolley\n')
            for r in rows:
                f.write('%s;%s;%s;%s;%.1f;%.1f;%.1f;%.1f;%.1f;%.1f;%.1f;%.1f;%.1f\n' % r)
        print('wrote', a.csv, '(%d rows)' % len(rows))


if __name__ == '__main__':
    main()
