"""
roi_make_comparison.py - Build a comparison folder of OUR ROI vs GT ROI.

For every GT frame (CONLY, LANDING), keeping the JOB FOLDER NAME, writes:
  <dst>/<job>/<frame>_<STATE>_OUR_cone.pcd   our extracted TARGET cone (the answer; steps 1-5)
  <dst>/<job>/<frame>_<STATE>_OUR_crop.pcd   the wide preset INPUT crop (context only -- by
                                             design contains the whole scene incl. partner cone)
  <dst>/<job>/<frame>_<STATE>_GT_roi.pcd     the human ground-truth cone body (copied)

Overlay _OUR_cone vs _GT_roi to judge accuracy. _OUR_crop is just the input context --
it WILL include the partner/deck; that is not a cut-off failure.

Usage: python roi_make_comparison.py [--dst <folder>]
"""
import sys, os, shutil, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R
import roi_cone_extract as CE

STATE_DIR = {'CONLY':'Chassis','LANDING':'Separated','LANDED':'Landed'}

def gt_body_path(gt_root, job, frame, state):
    d = os.path.join(gt_root, job, STATE_DIR.get(state,''))
    if not os.path.isdir(d): return None
    for fn in os.listdir(d):
        if fn.lower().startswith(frame.lower()) and fn.lower().endswith('.pcd'):
            return os.path.join(d, fn)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=R.GT_ROOT)
    ap.add_argument('--orig', default=R.ORIG_ROOT)
    ap.add_argument('--dst', default=r"K:\TUAS\South_Endfeed\GroundTruth_Set\Comparison\XT")
    a = ap.parse_args()

    n_our = n_gt = 0
    for g in R.load_gt(a.gt):
        if g['state'] not in ('CONLY','LANDING'): continue
        job, frame, state = g['job'], g['frame'], g['state']
        pts = R.load_cloud(job, frame)   # resolves exact-frame OR nearest BatchResult frame
        if pts is None: continue
        jp,chs,sprd,spos = R.parse_job(job); sensor = R.frame_sensor(frame)
        pc = R.preset_for(job, frame)
        if pc is None: continue
        Tc,Gc,Yc = pc
        outdir = os.path.join(a.dst, job); os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, f"{frame}_{state}")

        # _OUR_crop = the generous preset INPUT crop (must INCLUDE the whole object;
        # contains the partner cone + deck on purpose -- it is context, NOT the result).
        crop = R.preset_crop(pts, Tc, Gc)
        if len(crop):
            R.write_pcd(base+"_OUR_crop.pcd", crop)
        # _OUR_cone = Stage-2 tight result: target cone only (tips -> side-rule -> plate).
        tor = R.target_on_right(sensor, jp, chs)
        cone = CE.extract_cone(crop, Tc, Gc, Yc, tor, plate_depth=200.0)
        if cone is not None and len(cone):
            R.write_pcd(base+"_OUR_cone.pcd", cone); n_our += 1

        # GT ROI = the human cone body (copy as-is)
        src = gt_body_path(a.gt, job, frame, state)
        if src: shutil.copy2(src, base+"_GT_roi.pcd"); n_gt += 1

    print(f"wrote {n_our} OUR_cone (+crop) + {n_gt} GT_roi pairs under {a.dst}")

if __name__ == '__main__':
    main()
