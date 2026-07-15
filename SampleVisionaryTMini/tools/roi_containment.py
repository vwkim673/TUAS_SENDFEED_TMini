"""
roi_containment.py - Does our no-VA ROI INCLUDE the target region?

Uses the human-segmented GT cone body as the reference "target region" and
measures, per frame, how well our preset crop / isolated cone bump cover it:

  crop_cov   % of GT-body points that fall inside our +-150 preset crop box
             -> "does the preset box include the target region?"  (want ~100)
  pt_in      is the GT target POINT inside the crop?  margin = mm to nearest
             crop face (negative => point is OUTSIDE the crop)  (want >0, large)
  bump_prec  % of our cone-bump points that lie on the GT body (within TOL mm)
             -> "is what we isolated actually the target cone?" (want high)
  bump_cov   % of GT-body points within TOL mm of a bump point
             -> how much of the cone our (corner-biased) bump spans
  cen_d      distance between bump centroid and GT-body centroid (mm)

Flags frames where the target region is NOT safely included:
  crop_cov<90, pt outside/near edge (margin<30), or bump_prec<60 (wrong object).

Usage: python roi_containment.py [--tol 25] [--state CONLY] [--csv out.csv]
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R

STATE_DIR = {'CONLY':'Chassis','LANDING':'Separated','LANDED':'Landed'}
REGION_R = 75.0   # radius (mm) around the GT point defining the "necessary target region"

def gt_body(gt_root, job, frame, state):
    d = os.path.join(gt_root, job, STATE_DIR.get(state,''))
    if not os.path.isdir(d): return None
    for fn in os.listdir(d):
        if fn.lower().startswith(frame.lower()) and fn.lower().endswith('.pcd'):
            pts = R.read_pcd(os.path.join(d, fn))
            return pts[~((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))]
    return None

def nn_frac(a, b, tol):
    """fraction of points in `a` within tol of any point in `b` (chunked)."""
    if len(a)==0 or len(b)==0: return 0.0
    hit = 0
    for i in range(0, len(a), 1024):
        chunk = a[i:i+1024]
        d2 = ((chunk[:,None,:]-b[None,:,:])**2).sum(-1).min(1)
        hit += int((d2 <= tol*tol).sum())
    return 100.0*hit/len(a)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=R.GT_ROOT)
    ap.add_argument('--orig', default=R.ORIG_ROOT)
    ap.add_argument('--tol', type=float, default=25.0)
    ap.add_argument('--region-r', type=float, default=75.0)
    ap.add_argument('--state', default='')
    ap.add_argument('--csv', default='')
    a = ap.parse_args()
    global REGION_R; REGION_R = a.region_r

    rows = []
    for g in R.load_gt(a.gt):
        if g['state'] not in ('CONLY','LANDING'): continue
        if a.state and g['state']!=a.state: continue
        frame, job = g['frame'], g['job']
        bp = os.path.join(a.orig, job, frame, frame+'_base_converted.pcd')
        body = gt_body(a.gt, job, frame, g['state'])
        if not os.path.isfile(bp) or body is None: continue
        jp,chs,sprd,spos = R.parse_job(job); sensor = R.frame_sensor(frame)
        pc = R.preset_for(job, frame)
        if pc is None: continue
        cg,ct,cy = pc; key = R.config_key(sensor, jp, chs)
        pts = R.read_pcd(bp); pts = pts[~((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))]
        # generous preset-crop bounds. col0=Gantry(X), col1=Height(Y), col2=Trolley(Z).
        X0,X1 = cg-R.GANTRY_HALF, cg+R.GANTRY_HALF
        Z0,Z1 = ct-R.TROLLEY_BACK, ct+R.TROLLEY_FWD
        def incrop(p):
            return ((p[:,0]>=X0)&(p[:,0]<=X1)&(p[:,1]>=R.HEIGHT_MIN)&
                    (p[:,1]<=R.HEIGHT_MAX)&(p[:,2]>=Z0)&(p[:,2]<=Z1))
        crop_cov = 100.0*incrop(body).mean()
        # target point margin to crop faces. g['T']=col0(gantry), g['G']=col2(trolley)
        gX,gH,gZ = g['T'],g['H'],g['G']
        margin = min(gX-X0, X1-gX, gZ-Z0, Z1-gZ, gH-R.HEIGHT_MIN, R.HEIGHT_MAX-gH)
        # NECESSARY TARGET REGION = GT-body points within REGION_R of the GT point.
        gp = np.array([gX,gH,gZ])
        near = body[((body-gp)**2).sum(1) <= REGION_R*REGION_R]
        regN = len(near)
        reg_in_crop = 100.0*incrop(near).mean() if regN else float('nan')
        # our extraction
        det = R.detect(R.preset_crop(pts, cg, ct), R.target_on_right(sensor,jp,chs), cg)
        bump = det.get('bump') if det else None
        if bump is None or len(bump)==0:
            bp_prec=cen_d=float('nan')
        else:
            bp_prec = nn_frac(bump, body, a.tol)
            cen_d   = float(np.linalg.norm(bump.mean(0)-near.mean(0))) if regN else float('nan')
        # does our bump cover the necessary target region (corner neighbourhood)?
        reg_bump_cov = nn_frac(near, bump, a.tol) if (regN and bump is not None and len(bump)) else float('nan')
        cfg = f"{sensor} {jp} CHS{chs}"
        flag = ''
        if not np.isnan(reg_in_crop) and reg_in_crop<99: flag+='REGION! '   # target region clipped by crop
        if margin<30:   flag+='EDGE '
        if not np.isnan(bp_prec) and bp_prec<60: flag+='OBJ? '              # bump off the cone
        rows.append((key,cfg,g['state'],frame,len(body),crop_cov,margin,reg_in_crop,reg_bump_cov,bp_prec,cen_d,flag))

    # report
    print(f"tol={a.tol}mm  region=GT-body within {REGION_R:.0f}mm of GT point (the corner neighbourhood)\n"
          f"  margin   = mm from GT point to nearest crop face   (>0 => point inside crop)\n"
          f"  reg_crop%= % of target REGION inside our preset crop   (want 100 => region included)\n"
          f"  reg_bcov%= % of target REGION captured by our cone bump\n"
          f"  bprec%   = % of our cone bump that sits ON the GT cone body (is it the right object)\n"
          f"  body_crop%/gtN = full (loose) GT-body coverage + its size (context only)\n")
    print(f"{'cfg':16} {'st':8} {'margin':>6} {'reg_crop%':>9} {'reg_bcov%':>9} {'bprec%':>6} {'body_crop%':>10} {'gtN':>5}  flag")
    order={'CONLY':0,'LANDING':1}
    rows.sort(key=lambda r:(r[1],order.get(r[2],9),r[3]))
    cur=None
    for key,cfg,st,frame,gtN,cc,mg,rc,rbc,bp,cd,flag in rows:
        if cur!=cfg: print('  '+'-'*92); cur=cfg
        print(f"{cfg:16} {st:8} {mg:6.0f} {rc:9.0f} {rbc:9.0f} {bp:6.0f} {cc:10.0f} {gtN:5d}  {flag}")
    print("\n=== medians by state ===")
    print(f"{'state':8} {'n':>3} {'margin':>6} {'reg_crop%':>9} {'reg_bcov%':>9} {'bprec%':>6}   min_margin  min_reg_crop  #flagged")
    def col(i,st): return np.array([r[i] for r in rows if r[2]==st and not (isinstance(r[i],float) and np.isnan(r[i]))])
    for st in ('CONLY','LANDING'):
        sel=[r for r in rows if r[2]==st]
        if not sel: continue
        mg=col(6,st); rc=col(7,st); rbc=col(8,st); bp=col(9,st)
        nflag=sum(1 for r in sel if r[11])
        print(f"{st:8} {len(sel):3d} {np.median(mg):6.0f} {np.median(rc):9.0f} {np.median(rbc):9.0f} {np.median(bp):6.0f}   "
              f"{mg.min():6.0f}     {rc.min():6.0f}        {nflag}")

    if a.csv:
        import csv
        with open(a.csv,'w',newline='') as f:
            w=csv.writer(f); w.writerow(['key','cfg','state','frame','gtN','body_crop','margin','reg_in_crop','reg_bump_cov','bump_prec','cen_d','flag'])
            for r in rows: w.writerow(r)
        print("\nwrote", a.csv)

if __name__ == '__main__':
    main()
