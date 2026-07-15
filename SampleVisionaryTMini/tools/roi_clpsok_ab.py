"""
roi_clpsok_ab.py - A/B tuning of CLPS-OK rule variants on the reverse-replay cache.

Framework (all variants share it):
  - DYNAMIC gate: armed while PLC-landed==0; landed==1 -> reset counters AND release
    (re-land handling: the 134547 job proves lift -> re-land happens in real jobs).
  - landed==None (VA-blind frame): carry the last known landed state.
  - Fire = first frame in an armed period where the variant's conditions hold.

Variants (H = our NEW 3D target height, holeY = original-run 3D hole Y):
  V1 stab+devY300 : H stable W=3(<=50mm) AND (hole valid -> H-holeY < -300)
  V2 stab+devY150 : ... devY < -150
  V3 stab+riseGone: ... (devY < -150 OR holeY risen >=40mm over last 2 valid samples)
  V4 stabOnly     : H stable only (no hole condition)
  OLDVA           : first frame idx==1 & xp>=70 (the current C++ instant rule), same gate-free
                    behaviour as production (fires anywhere) -- for contrast.

Metrics: fires while landed==1 (MUST be 0 for gated variants), fire delay after the
FINAL land-off, brief-lift fires (armed periods that re-land afterwards), never-fired.

Usage: python roi_clpsok_ab.py [--cache _replay_cache.csv]
"""
import sys, os, argparse, collections
import numpy as np

def load_cache(path):
    jobs = collections.OrderedDict()
    for ln in open(path, encoding='utf-8').read().splitlines()[1:]:
        c = ln.split(';')
        if len(c) < 11: continue
        key = (c[0], c[1])
        jobs.setdefault(key, []).append(dict(
            i=int(c[2]), fr=c[3], idx=int(c[4]), xp=float(c[5]), hp=float(c[6]),
            landed=(None if c[7] == '' else int(c[7])),
            holeY=(None if c[8] == '' else float(c[8])),
            H=(None if c[9] == '' else float(c[9]))))
    for k in jobs: jobs[k].sort(key=lambda r: r['i'])
    return jobs

def carry_landed(recs):
    last = None
    out = []
    for r in recs:
        if r['landed'] is not None: last = r['landed']
        out.append(last)
    return out

def run_variant(recs, landed, variant, W=3, STAB=50.0):
    """Returns list of fire events: (index, armed_period_id). Gate: armed while landed==0."""
    fires = []
    win = []; fired_this_period = False; period = -1
    lastHoleYs = []
    for i, r in enumerate(recs):
        ld = landed[i]
        if ld != 0:
            # landed or unknown-before-first-bit: disarm + reset (+release)
            win = []; fired_this_period = False
            if ld == 1: period = period  # period increments on next arm
            continue
        # armed (landed==0). new period?
        if i == 0 or landed[i-1] != 0:
            period += 1; win = []; fired_this_period = False; lastHoleYs = []
        if r['holeY'] is not None: lastHoleYs.append(r['holeY'])
        if fired_this_period: continue
        if variant == 'OLDVA':
            if r['idx'] == 1 and r['xp'] >= 70:
                fires.append((i, period)); fired_this_period = True
            continue
        if r['H'] is None: win = []; continue
        # hole condition
        ok = True
        if r['holeY'] is not None:
            dev = r['H'] - r['holeY']
            if variant == 'V1': ok = dev < -300
            elif variant == 'V2': ok = dev < -150
            elif variant == 'V3':
                rise = len(lastHoleYs) >= 3 and (lastHoleYs[-1] - lastHoleYs[-3]) >= 40
                ok = (dev < -150) or rise
            elif variant == 'V4': ok = True
        if not ok: win = []; continue
        win.append(r['H'])
        if len(win) > W: win.pop(0)
        if len(win) == W and (max(win) - min(win)) <= STAB:
            fires.append((i, period)); fired_this_period = True
    return fires

def analyze(recs):
    landed = carry_landed(recs)
    # final land-off = LAST 1->0 transition
    Lf = None
    for i in range(1, len(landed)):
        if landed[i-1] == 1 and landed[i] == 0: Lf = i
    if Lf is None:
        Lf = next((i for i, l in enumerate(landed) if l == 0), None)
    # brief-lift periods = armed periods that are followed by landed==1 again
    return landed, Lf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '_replay_cache.csv'))
    a = ap.parse_args()
    jobs = load_cache(a.cache)
    variants = ['OLDVA', 'V1', 'V2', 'V3', 'V4']
    stats = {v: dict(delays=[], never=[], landedFP=[], brief=0, fired=0) for v in variants}
    njob = 0
    for (spos, job), recs in jobs.items():
        landed, Lf = analyze(recs)
        if Lf is None: continue          # never un-lands (real offload recs) -> skip
        njob += 1
        # does this job re-land after some armed period? (for brief-lift accounting)
        for v in variants:
            fires = run_variant(recs, landed, v)
            # fires while landed==1 (only possible for OLDVA which ignores gate)
            fp = [i for i, p in fires if landed[i] == 1]
            if v == 'OLDVA':
                # OLDVA as production: no gate -- recompute ungated: fire at first idx1 anywhere
                og = next((i for i, r in enumerate(recs) if r['idx'] == 1 and r['xp'] >= 70), None)
                fires = [(og, 0)] if og is not None else []
                fp = [i for i, p in fires if landed[i] == 1]
            final_fire = next((i for i, p in fires if i >= Lf), None)
            brief_fires = [i for i, p in fires if i < Lf]
            st = stats[v]
            if fp: st['landedFP'].append((job[:40], fp[0]))
            if final_fire is not None:
                st['delays'].append(final_fire - Lf); st['fired'] += 1
            else:
                st['never'].append(spos + ':' + job[:40])
            st['brief'] += len(brief_fires)
    print('jobs with a final land-off: %d' % njob)
    print('%-8s %6s %7s %7s %7s %6s %9s' % ('variant', 'fired', 'medDly', 'p90Dly', 'maxDly', 'brief', 'landedFP'))
    for v in variants:
        st = stats[v]
        d = st['delays']
        print('%-8s %4d/%-2d %7s %7s %7s %6d %9d' % (
            v, st['fired'], njob,
            '%.0f' % np.median(d) if d else '-', '%.0f' % np.percentile(d, 90) if d else '-',
            '%.0f' % max(d) if d else '-', st['brief'], len(st['landedFP'])))
    for v in variants:
        if stats[v]['never']:
            print('%s never-fired: %s' % (v, stats[v]['never']))
    if stats['OLDVA']['landedFP']:
        print('OLDVA fires while LANDED (production FP risk):', stats['OLDVA']['landedFP'][:6])

if __name__ == '__main__':
    main()
