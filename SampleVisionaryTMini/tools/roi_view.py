"""
roi_view.py - Interactive 3D viewer of the ROI extraction vs GT/VA (Plotly HTML).

For one frame (or every GT frame), renders color-coded layers in an orbitable
browser view:
  - base scene (light grey, downsampled)        context
  - preset crop (steel blue)                     the +-150 T/G box handed to the detector
  - near-Z slab (cyan)                           protruding structure
  - selected cone bump (orange)                  the isolated target cone
  - NEW cone point (red diamond)                 this method's result
  - VA cone point (green diamond)                the 2D-model reference
  - GT target (magenta diamond)                  human ground truth

Writes <out>/<frame>.html. Open in any browser; hover shows T,H,G mm.

Usage:
  python roi_view.py                         # all GT frames -> tools/roi_html/
  python roi_view.py --frame <name>          # one frame
  python roi_view.py --state CONLY           # only CONLY frames
  python roi_view.py --worst 8               # the N worst-|dT| frames
"""
import sys, os, argparse
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R

def downsample(a, n):
    if a is None or len(a) <= n: return a
    idx = np.linspace(0, len(a)-1, n).astype(int)
    return a[idx]

def scatter(pts, name, color, size=2, opacity=0.8):
    pts = np.asarray(pts).reshape(-1,3)
    return go.Scatter3d(x=pts[:,0], y=pts[:,2], z=pts[:,1], mode='markers',
                        name=name, marker=dict(size=size, color=color, opacity=opacity),
                        hovertemplate='T=%{x:.0f}<br>G=%{y:.0f}<br>H=%{z:.0f}<extra>'+name+'</extra>')

def marker(p, name, color):
    p = np.asarray(p).reshape(3)
    return go.Scatter3d(x=[p[0]], y=[p[2]], z=[p[1]], mode='markers',
                        name=name, marker=dict(size=7, color=color, symbol='diamond',
                                               line=dict(width=1,color='black')),
                        hovertemplate='T=%{x:.0f}<br>G=%{y:.0f}<br>H=%{z:.0f}<extra>'+name+'</extra>')

def render_frame(job, frame, gT,gH,gG, Tc,Gc, orig, outdir, half_t=150, half_g=150):
    bp = os.path.join(orig, job, frame, frame+'_base_converted.pcd')
    if not os.path.isfile(bp): return None
    pts = R.read_pcd(bp); pts = pts[~((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))]
    crop = R.preset_crop(pts, Tc, Gc)
    sensor = R.frame_sensor(frame); jp,chs,sprd,spos = R.parse_job(job)
    tor = R.target_on_right(sensor, jp, chs)
    det = R.detect(crop, tor, Tc)
    if det is None: return None
    cone = det['cor_protr']
    # base scene limited to a neighbourhood for context (not all 217k)
    nb = pts[(pts[:,0]>=Tc-half_t-250)&(pts[:,0]<=Tc+half_t+250)&
             (pts[:,2]>=Gc-half_g-250)&(pts[:,2]<=Gc+half_g+250)]
    va = None
    vp = os.path.join(orig, job, frame, frame+'_AB_va_conePos.pcd')
    if os.path.isfile(vp): va = R.read_pcd(vp)[0]

    data = [scatter(downsample(nb,4000),'base scene','lightgrey',1,0.35),
            scatter(downsample(crop,3000),'crop (preset box)','steelblue',2,0.5),
            scatter(downsample(det.get('slab'),1500),'near-Z slab','cyan',2,0.7)]
    if det.get('bump') is not None:
        data.append(scatter(downsample(det['bump'],800),'cone bump','orange',3,0.9))
    data.append(marker(cone,'NEW cone','red'))
    if va is not None: data.append(marker(va,'VA cone','green'))
    data.append(marker([gT,gH,gG],'GT target','magenta'))

    dT = cone[0]-gT; dG = cone[2]-gG; dH = cone[1]-gH
    title = (f"{frame}<br><sub>{sensor} {jp} CHS{chs} | NEW dT={dT:+.0f} dH={dH:+.0f} dG={dG:+.0f} mm"
             + (f" | VA dT={va[0]-gT:+.0f}" if va is not None else "") + "</sub>")
    fig = go.Figure(data=data)
    fig.update_layout(title=title, scene=dict(
        xaxis_title='Trolley T (mm)', yaxis_title='Gantry G (mm)', zaxis_title='Height H (mm)',
        aspectmode='data'), legend=dict(itemsizing='constant'), margin=dict(l=0,r=0,t=60,b=0))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, frame+'.html')
    fig.write_html(out, include_plotlyjs='cdn')
    return out, abs(dT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=R.GT_ROOT)
    ap.add_argument('--orig', default=R.ORIG_ROOT)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'roi_html'))
    ap.add_argument('--frame', default='')
    ap.add_argument('--state', default='')      # CONLY / LANDING
    ap.add_argument('--worst', type=int, default=0)
    a = ap.parse_args()

    gt = R.load_gt(a.gt)
    results = []
    for g in gt:
        if g['state'] not in ('CONLY','LANDING'): continue
        if a.state and g['state']!=a.state: continue
        if a.frame and a.frame not in g['frame']: continue
        pc = R.preset_for(g['job'], g['frame'])
        if pc is None: continue
        Tc,Gc,Yc = pc
        r = render_frame(g['job'], g['frame'], g['T'],g['H'],g['G'], Tc,Gc, a.orig, a.out)
        if r: results.append((r[1], r[0], g['frame']))
    results.sort(reverse=True)
    if a.worst: results = results[:a.worst]
    print(f"wrote {len(results)} html files to {a.out}")
    for dt,path,frame in results[:a.worst or 12]:
        print(f"  |dT|={dt:5.0f}  {os.path.basename(path)}")
    if results:
        print(f"\nopen e.g.:  start \"\" \"{results[0][1]}\"")

if __name__ == '__main__':
    main()
