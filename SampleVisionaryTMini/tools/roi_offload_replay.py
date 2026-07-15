"""
roi_offload_replay.py - Offload(CLPS/CLPS-OK) study by REVERSE-replaying Mount jobs.

A Mount recording runs cone-only -> landing -> landed. Played BACKWARDS it is an
Offload: landed -> hoist -> separation -> cone-only. Real offload recordings all end
while still landed (checked 2026-07), so this is the only sequence data with the
separation phase.

Per BR job (BatchResult scheme, per-frame res.txt rows):
  - timeline = frames by timestamp, REVERSED for Mount (kept for real Offload jobs)
  - per frame: VA (XT_index/prob, Hole_prob), visual Landed_Trigger, original-run 3D
    hole_y  +  OUR NEW 3D target (T,H,G) recomputed from the frame PCD
  - phase segmentation (reversed time): LANDED (Landed=1) / SEP (Landed=0, idx2 or
    hole visible) / CONLY (idx1 stable)
  - measurements: chassis-H rebound (unloaded vs loaded), VA class flapping,
    separation devY trajectory, and WHEN each candidate CLPS-OK rule would fire.

Usage: python roi_offload_replay.py [--spos FRONT|REAR|BOTH] [--job <substr>] [--per-frame]
"""
import sys, os, glob, argparse, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R
import roi_cone_extract as E
from roi_user_pipeline import new_target

BASE = r"K:\TUAS\South_Endfeed\GroundTruth_Set\OriginalData\XT"

def frame_ms(name):
    m = re.match(r'\d{8}_(\d{2})(\d{2})(\d{2})_(\d{3})', name)
    if not m: return 0
    hh, mm, ss, ml = map(int, m.groups())
    return ((hh*60+mm)*60+ss)*1000+ml

def parse_res(path):
    rows = {}
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    for ln in lines[1:]:
        c = ln.split(';')
        if len(c) < 22: continue
        fr = c[0]
        full = len(c) >= 40
        try:
            rows[fr] = dict(
                xt_idx=int(float(c[6])) if full else 0,
                xt_prob=float(c[11]) if full else 0.0,
                hole_prob=float(c[5]) if full else 0.0,
                landed=int(float(c[21])) if full else None,
                hole_y3d=float(c[29]) if full else -10000.0,
                cone_y3d_old=float(c[32]) if full else -10000.0)
        except Exception:
            continue
    return rows

def our_target(job, frame, sensor, jp, chs):
    key = R.config_key(sensor, jp, chs)
    if key not in R.PRESETS: return None
    pts = R.load_cloud(job, frame)
    if pts is None: return None
    cg, ct, cy = R.PRESETS[key]
    tor = R.target_on_right(sensor, jp, chs)
    cone = E.extract_cone(R.preset_crop(pts, cg, ct), cg, ct, cy, tor)
    if cone is None or len(cone) < 30: return None
    return new_target(cone, tor)

CACHE_FH = None

def analyze_job(spos, job, per_frame=False):
    jd = os.path.join(BASE, spos, 'BatchResult', job)
    res = glob.glob(os.path.join(jd, '*_res.txt'))
    if not res: return None
    rows = parse_res(res[0])
    frames = sorted([x for x in os.listdir(jd) if x.endswith('_Image')], key=frame_ms)
    if len(frames) < 8: return None
    jp, chs, sprd, _ = R.parse_job(job)
    is_offload = '_Offload_' in job
    if not is_offload: frames = frames[::-1]          # reverse mount -> offload
    recs = []
    for fr in frames:
        rr = rows.get(fr, {})
        sensor = R.frame_sensor(fr)
        t = our_target(job, fr, sensor, jp, chs)
        recs.append(dict(fr=fr, sensor=sensor,
                         idx=rr.get('xt_idx', 0), xp=rr.get('xt_prob', 0.0),
                         hp=rr.get('hole_prob', 0.0), landed=rr.get('landed'),
                         holeY=rr.get('hole_y3d', -10000.0),
                         H=(t[1] if t is not None else None),
                         T=(t[0] if t is not None else None)))
    # phase indices (replay order): landed while landed==1; first landed==0 = land-off
    landoff = next((i for i, r in enumerate(recs) if r['landed'] == 0), None)
    first_idx1 = next((i for i, r in enumerate(recs) if r['idx'] == 1 and r['xp'] >= 70), None)
    # cone-only segment = after the LAST frame where the hole (container) is still seen
    last_hole = max([i for i, r in enumerate(recs) if r['hp'] > 0], default=-1)
    conly0 = last_hole + 1
    # flapping: idx changes among detected frames
    seq = [r['idx'] for r in recs if r['idx'] in (1, 2)]
    flaps = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    # H segments
    def med(vals): return float(np.median(vals)) if vals else None
    H_land = med([r['H'] for i, r in enumerate(recs) if r['landed'] == 1 and r['H'] is not None])
    H_conly = med([r['H'] for r in recs[conly0:] if r['H'] is not None])
    sep0 = landoff if landoff is not None else 0
    H_sep = med([r['H'] for r in recs[sep0+2:conly0] if r['H'] is not None])   # skip 2 transition frames
    rebound = (H_conly - H_sep) if (H_conly is not None and H_sep is not None) else None
    dH_land = (H_land - H_conly) if (H_land is not None and H_conly is not None) else None
    # VA blindness in the cone-only segment (would kill the VA cone-only rule)
    va_blind = (first_idx1 is None) or (first_idx1 >= len(recs)-1)
    # candidate rule firing (replay order index)
    # OLD-VA: first idx1&prob>70 frame (instant, no debounce)
    fire_oldva = first_idx1
    # NEW-3D: ARMED after land-off; fire when our H is STABLE 3 consecutive frames
    # (range<=50mm) AND separation vs hole confirmed when hole visible (devY<-300)
    fire_new = None
    if landoff is not None:
        win = []
        for i in range(landoff, len(recs)):
            r = recs[i]
            if r['H'] is None: win = []; continue
            sep_ok = True
            if r['holeY'] != -10000.0: sep_ok = (r['H'] - r['holeY']) < -300
            if not sep_ok: win = []; continue
            win.append(r['H'])
            if len(win) > 3: win.pop(0)
            if len(win) == 3 and (max(win)-min(win)) <= 50: fire_new = i; break
    # FALSE-fire check: would the same stability rule fire during LANDED if not gated?
    false_fire = False
    win = []
    for i, r in enumerate(recs[:landoff if landoff is not None else len(recs)]):
        if r['H'] is None: win = []; continue
        win.append(r['H'])
        if len(win) > 3: win.pop(0)
        if len(win) == 3 and (max(win)-min(win)) <= 50: false_fire = True; break
    if CACHE_FH is not None:
        for i, r in enumerate(recs):
            CACHE_FH.write('%s;%s;%d;%s;%d;%.0f;%.0f;%s;%s;%s;%s\n' % (
                spos, job, i, r['fr'], r['idx'], r['xp'], r['hp'],
                '' if r['landed'] is None else r['landed'],
                '' if r['holeY'] == -10000.0 else '%.1f' % r['holeY'],
                '' if r['H'] is None else '%.1f' % r['H'],
                '' if r['T'] is None else '%.1f' % r['T']))
        CACHE_FH.flush()
    if per_frame:
        print('   i  frame            idx  xp  hp land  holeY    ourH')
        for i, r in enumerate(recs):
            print('  %3d %-16s %3d %3.0f %3.0f %4s %7s %7s' % (
                i, r['fr'][9:24], r['idx'], r['xp'], r['hp'],
                str(r['landed']), '%.0f' % r['holeY'] if r['holeY'] != -10000 else '-',
                '%.0f' % r['H'] if r['H'] is not None else '-'))
    return dict(job=job, spos=spos, n=len(recs), offload=is_offload,
                landoff=landoff, first_idx1=first_idx1, flaps=flaps,
                H_land=H_land, H_sep=H_sep, H_conly=H_conly, rebound=rebound, dH_land=dH_land,
                va_blind=va_blind, fire_oldva=fire_oldva, fire_new=fire_new, false_fire=false_fire)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spos', default='BOTH')
    ap.add_argument('--job', default='')
    ap.add_argument('--per-frame', action='store_true')
    ap.add_argument('--cache', default='')
    a = ap.parse_args()
    sposs = ['FRONT', 'REAR'] if a.spos == 'BOTH' else [a.spos]
    global CACHE_FH
    if a.cache:
        CACHE_FH = open(a.cache, 'w', encoding='utf-8')
        CACHE_FH.write('spos;job;i;frame;idx;xp;hp;landed;holeY;H;T\n')
    out = []
    for spos in sposs:
        br = os.path.join(BASE, spos, 'BatchResult')
        for job in sorted(os.listdir(br)):
            if a.job and a.job not in job: continue
            if not os.path.isdir(os.path.join(br, job)): continue
            r = analyze_job(spos, job, per_frame=a.per_frame)
            if r: out.append(r)
    print()
    print('%-42s %3s %5s %6s %5s %6s %6s %6s %7s %7s %6s %6s %6s' % (
        'job', 'n', 'lOff', 'idx1', 'flap', 'Hland', 'Hsep', 'Hconly', 'rebound', 'dHland', 'OLDf', 'NEWf', 'VAblind'))
    for r in out:
        f = lambda v, fmt='%.0f': (fmt % v) if v is not None else '-'
        print('%-42s %3d %5s %6s %5d %6s %6s %6s %7s %7s %6s %6s %6s' % (
            r['job'][:42], r['n'], str(r['landoff']), str(r['first_idx1']), r['flaps'],
            f(r['H_land']), f(r['H_sep']), f(r['H_conly']), f(r['rebound'], '%+.0f'), f(r['dH_land'], '%+.0f'),
            str(r['fire_oldva']), str(r['fire_new']), ('BLIND' if r['va_blind'] else '')))
    nf = sum(1 for r in out if r['false_fire'])
    print('jobs where UNGATED stability rule would false-fire during LANDED: %d/%d  (=> PLC land-off gate is mandatory)' % (nf, len(out)))

if __name__ == '__main__':
    main()
