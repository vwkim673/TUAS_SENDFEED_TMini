"""
roi_preset_analysis.py - GT spread + preset feasibility analysis for XT ROI.

Reads GroundTruth/XT/groundtruth.txt (target X;Y;Z = T,H,G mm). For each row:
  - derive sensor token (FS/CS/RS) from the frame name,
  - derive jobPos / chassis_len / sprd from the job folder name,
  - resolve the preset key exactly as SampleVisionaryTMini.cpp does
    (CURRENT_SENSOR_POSITION = FRONT/REAR from the folder suffix).
Then group CONLY+LANDING (LANDED reported separately, out of scope) and report
per-(preset-key) and per-(sensor,config) T/G centers and spreads, to decide
whether +-150mm windows are feasible and how keys must be split.

Optionally joins each job's <job>_res.txt (prior ROI run) ROI_T/H/G + AB_* to GT
for a baseline current-error snapshot.

Usage: python roi_preset_analysis.py [gt_root] [orig_root]
"""
import sys, os
import numpy as np

GT_ROOT  = r"K:\TUAS\South_Endfeed\GroundTruth_Set\GroundTruth\XT"
ORIG_ROOT= r"K:\TUAS\South_Endfeed\GroundTruth_Set\OriginalData\XT"

# ---- preset key resolution (mirrors cpp 2355-2411, XT only) ----------------
def resolve_key(sensor_pos, chs_len, sprd, job_pos):
    """sensor_pos in {FRONT,REAR}; returns e.g. 'F_XT_4XF'."""
    if sensor_pos == "FRONT":
        if chs_len == 45 and sprd == 40:      k = "4XR"
        elif chs_len == 45 and job_pos=="Small": k = "20R"
        elif chs_len == 40 and job_pos=="Small": k = "20F"
        else:                                  k = "4XF"
        return "F_XT_" + k
    else:
        if job_pos == "Big":   k = "20F"
        elif chs_len == 40:    k = "4XF"
        else:                  k = "4XR"
        return "R_XT_" + k

def parse_job(job):
    parts = job.split('_')
    jp = next((p for p in parts if p in ("Center","Big","Small")), "?")
    chs = next((int(p[3:]) for p in parts if p.startswith("CHS")), 0)
    sprd= next((int(p[4:]) for p in parts if p.startswith("SPRD")), 0)
    sensor_pos = "FRONT" if "_FRONT_" in job else ("REAR" if "_REAR_" in job else "?")
    return jp, chs, sprd, sensor_pos

def frame_sensor(frame):
    return 'FS' if '_FS' in frame else ('CS' if '_CS' in frame else ('RS' if '_RS' in frame else '??'))

# ---- load GT ---------------------------------------------------------------
def load_gt(root):
    rows = []
    with open(os.path.join(root,'groundtruth.txt'), encoding='utf-8') as f:
        for ln in f.read().splitlines()[1:]:
            c = [x.strip() for x in ln.split(';')]
            if len(c) < 6: continue
            job, frame, state = c[0], c[1], c[2]
            try: T,H,G = float(c[3]), float(c[4]), float(c[5])
            except ValueError: continue
            jp, chs, sprd, spos = parse_job(job)
            rows.append(dict(job=job, frame=frame, state=state, T=T,H=H,G=G,
                             jp=jp, chs=chs, sprd=sprd, spos=spos,
                             sensor=frame_sensor(frame),
                             key=resolve_key(spos, chs, sprd, jp)))
    return rows

def spread(vals):
    a = np.array(vals, float)
    return a.min(), a.max(), a.max()-a.min(), np.median(a)

def main():
    gt_root  = sys.argv[1] if len(sys.argv)>1 else GT_ROOT
    rows = load_gt(gt_root)
    inscope = [r for r in rows if r['state'] in ("CONLY","LANDING")]

    # config label
    def cfg(r): return f"{r['sensor']} {r['jp']} CHS{r['chs']} SPRD{r['sprd']}"

    print("="*100)
    print("PER-JOB targets (CONLY + LANDING), grouped by resolved preset KEY")
    print("="*100)
    print(f"{'KEY':10} {'config':26} {'sensor':6} {'state':8} {'T':>7} {'H':>7} {'G':>7}  {'job'}")
    for r in sorted(inscope, key=lambda r:(r['key'], cfg(r), r['state'])):
        print(f"{r['key']:10} {cfg(r):26} {r['sensor']:6} {r['state']:8} "
              f"{r['T']:7.0f} {r['H']:7.0f} {r['G']:7.0f}  {r['job']}")

    print("\n"+"="*100)
    print("SPREAD per preset KEY  (does +-150mm fit?  span<=300 => YES)")
    print("="*100)
    print(f"{'KEY':10} {'n':>3} {'sensors':14} "
          f"{'T_min':>7} {'T_max':>7} {'T_span':>7} {'T_mid':>7}  "
          f"{'G_min':>7} {'G_max':>7} {'G_span':>7} {'G_mid':>7}  fit?")
    keys = {}
    for r in inscope: keys.setdefault(r['key'], []).append(r)
    for k in sorted(keys):
        g = keys[k]
        Tmin,Tmax,Tspan,_ = spread([r['T'] for r in g])
        Gmin,Gmax,Gspan,_ = spread([r['G'] for r in g])
        Tmid = (Tmin+Tmax)/2; Gmid=(Gmin+Gmax)/2
        sensors = ",".join(sorted(set(r['sensor'] for r in g)))
        fit = "YES" if (Tspan<=300 and Gspan<=300) else "NO <-- split"
        print(f"{k:10} {len(g):3d} {sensors:14} "
              f"{Tmin:7.0f} {Tmax:7.0f} {Tspan:7.0f} {Tmid:7.0f}  "
              f"{Gmin:7.0f} {Gmax:7.0f} {Gspan:7.0f} {Gmid:7.0f}  {fit}")

    print("\n"+"="*100)
    print("SPREAD per (sensor, jobPos, CHS)  -- the geometry-clean grouping")
    print("="*100)
    print(f"{'config':26} {'n':>3} "
          f"{'T_min':>7} {'T_max':>7} {'T_span':>7} {'T_mid':>7}  "
          f"{'G_min':>7} {'G_max':>7} {'G_span':>7} {'G_mid':>7}  {'H_min':>7} {'H_max':>7}")
    cfgs = {}
    for r in inscope:
        ck = f"{r['sensor']} {r['jp']} CHS{r['chs']}"
        cfgs.setdefault(ck, []).append(r)
    for ck in sorted(cfgs):
        g = cfgs[ck]
        Tmin,Tmax,Tspan,_ = spread([r['T'] for r in g])
        Gmin,Gmax,Gspan,_ = spread([r['G'] for r in g])
        Hmin,Hmax,_,_     = spread([r['H'] for r in g])
        print(f"{ck:26} {len(g):3d} "
              f"{Tmin:7.0f} {Tmax:7.0f} {Tspan:7.0f} {(Tmin+Tmax)/2:7.0f}  "
              f"{Gmin:7.0f} {Gmax:7.0f} {Gspan:7.0f} {(Gmin+Gmax)/2:7.0f}  {Hmin:7.0f} {Hmax:7.0f}")

    # CONLY->LANDING motion (chassis stationarity check)
    print("\n"+"="*100)
    print("CONLY -> LANDING motion per job (chassis should be stationary)")
    print("="*100)
    byjob = {}
    for r in rows:
        byjob.setdefault(r['job'], {})[r['state']] = r
    print(f"{'job':56} {'dT':>6} {'dH':>6} {'dG':>6}")
    dTs, dHs, dGs = [], [], []
    for job, st in sorted(byjob.items()):
        if 'CONLY' in st and 'LANDING' in st:
            dT = st['LANDING']['T']-st['CONLY']['T']
            dH = st['LANDING']['H']-st['CONLY']['H']
            dG = st['LANDING']['G']-st['CONLY']['G']
            dTs.append(abs(dT)); dHs.append(abs(dH)); dGs.append(abs(dG))
            print(f"{job:56} {dT:6.0f} {dH:6.0f} {dG:6.0f}")
    if dTs:
        print(f"{'|median|':56} {np.median(dTs):6.0f} {np.median(dHs):6.0f} {np.median(dGs):6.0f}")
        print(f"{'|max|':56} {max(dTs):6.0f} {max(dHs):6.0f} {max(dGs):6.0f}")

if __name__ == '__main__':
    main()
