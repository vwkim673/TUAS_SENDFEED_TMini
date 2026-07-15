"""
roi_compare.py - Side-by-side comparison DOC: our ROI extraction vs GT ROI.

Produces ONE self-contained HTML report:
  - a summary table (every frame: our cone T/H/G, GT T/H/G, dT/dH/dG, region-in-crop)
  - per frame, a side-by-side 3D view:
       LEFT  = OUR ROI  (preset crop grey, cone bump orange, OUR cone = red diamond)
       RIGHT = GT  ROI  (human GT cone body green, GT target = magenta diamond)
    both panels share the same axes and both show BOTH markers, so the red(ours)
    vs magenta(GT) offset is directly visible in either panel.
  - grouped by job-position config (sensor x jobPos x CHS); all configs covered.

Open the HTML in any browser; orbit/zoom each panel; hover shows T,H,G mm.

Usage:
  python roi_compare.py                       # CONLY (chassis state) -> roi_html/roi_compare.html
  python roi_compare.py --state ALL           # CONLY + LANDING
  python roi_compare.py --out path.html
"""
import sys, os, argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_fullpipe_eval as R

STATE_DIR = {'CONLY':'Chassis','LANDING':'Separated','LANDED':'Landed'}

def gt_body(gt_root, job, frame, state):
    d = os.path.join(gt_root, job, STATE_DIR.get(state,''))
    if not os.path.isdir(d): return None
    for fn in os.listdir(d):
        if fn.lower().startswith(frame.lower()) and fn.lower().endswith('.pcd'):
            pts = R.read_pcd(os.path.join(d, fn))
            return pts[~((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))]
    return None

def ds(a, n):
    if a is None or len(a) <= n: return a
    return a[np.linspace(0, len(a)-1, n).astype(int)]

def sc(pts, name, color, size=2, op=0.6, scene=None):
    pts = np.asarray(pts).reshape(-1,3)
    return go.Scatter3d(x=pts[:,0], y=pts[:,2], z=pts[:,1], mode='markers', name=name,
                        marker=dict(size=size, color=color, opacity=op), scene=scene,
                        showlegend=False,
                        hovertemplate='T=%{x:.0f} G=%{y:.0f} H=%{z:.0f}<extra>'+name+'</extra>')

def mk(p, name, color, scene=None):
    p = np.asarray(p).reshape(3)
    return go.Scatter3d(x=[p[0]], y=[p[2]], z=[p[1]], mode='markers', name=name, scene=scene,
                        marker=dict(size=6, color=color, symbol='diamond', line=dict(width=1,color='black')),
                        showlegend=False,
                        hovertemplate='T=%{x:.0f} G=%{y:.0f} H=%{z:.0f}<extra>'+name+'</extra>')

def frame_fig(job, frame, gT,gH,gG, Tc,Gc, orig, gt_root, state):
    bp = os.path.join(orig, job, frame, frame+'_base_converted.pcd')
    if not os.path.isfile(bp): return None
    pts = R.read_pcd(bp); pts = pts[~((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))]
    crop = R.preset_crop(pts, Tc, Gc)
    sensor=R.frame_sensor(frame); jp,chs,sprd,spos=R.parse_job(job)
    det = R.detect(crop, R.target_on_right(sensor,jp,chs), Tc)
    if det is None: return None
    cone = det['cor_protr']; bump = det.get('bump')
    body = gt_body(gt_root, job, frame, state)
    gp = [gT,gH,gG]

    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'scene'},{'type':'scene'}]],
                        subplot_titles=('OUR ROI (red=our cone)','GT ROI (magenta=GT)'),
                        horizontal_spacing=0.02)
    # LEFT: our extraction
    fig.add_trace(sc(ds(crop,2500),'crop','lightsteelblue',2,0.35,scene='scene'), 1,1)
    if bump is not None: fig.add_trace(sc(ds(bump,700),'bump','orange',3,0.85,scene='scene'),1,1)
    fig.add_trace(mk(cone,'OUR cone','red',scene='scene'),1,1)
    fig.add_trace(mk(gp,'GT','magenta',scene='scene'),1,1)
    # RIGHT: GT body
    if body is not None: fig.add_trace(sc(ds(body,2500),'GT body','seagreen',2,0.5,scene='scene2'),1,2)
    fig.add_trace(mk(cone,'OUR cone','red',scene='scene2'),1,2)
    fig.add_trace(mk(gp,'GT','magenta',scene='scene2'),1,2)

    # shared axis ranges around the cone for comparability
    rng=200
    sx=[min(cone[0],gT)-rng, max(cone[0],gT)+rng]
    sy=[min(cone[2],gG)-rng, max(cone[2],gG)+rng]
    sz=[min(cone[1],gH)-rng, max(cone[1],gH)+rng]
    axes=dict(xaxis=dict(title='T',range=sx), yaxis=dict(title='G',range=sy),
              zaxis=dict(title='H',range=sz), aspectmode='cube')
    cam=dict(eye=dict(x=1.4,y=1.4,z=1.0))
    fig.update_layout(scene=dict(**axes,camera=cam), scene2=dict(**axes,camera=cam),
                      height=420, margin=dict(l=0,r=0,t=24,b=0))
    dT,dH,dG = cone[0]-gT, cone[1]-gH, cone[2]-gG
    return fig, (dT,dH,dG)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gt', default=R.GT_ROOT); ap.add_argument('--orig', default=R.ORIG_ROOT)
    ap.add_argument('--state', default='CONLY', help='CONLY | LANDING | ALL')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'roi_html','roi_compare.html'))
    a=ap.parse_args()
    states = ('CONLY','LANDING') if a.state.upper()=='ALL' else (a.state.upper(),)

    gt=R.load_gt(a.gt)
    items=[]
    for g in gt:
        if g['state'] not in states: continue
        jp,chs,sprd,spos=R.parse_job(g['job'])
        pc=R.preset_for(g['job'], g['frame'])
        if pc is None: continue
        key=R.config_key(R.frame_sensor(g['frame']), jp, chs)
        cfg=f"{R.frame_sensor(g['frame'])} {jp} CHS{chs}"
        items.append((cfg,key,g))
    items.sort(key=lambda x:(x[0], x[2]['state']))

    figs_html=[]; table=[]; first=True
    cur=None
    for cfg,key,g in items:
        r=frame_fig(g['job'],g['frame'],g['T'],g['H'],g['G'],*R.PRESETS[key][:2],a.orig,a.gt,g['state'])
        if r is None: continue
        fig,(dT,dH,dG)=r
        if cfg!=cur:
            figs_html.append(f'<h2 style="margin:18px 0 4px">{cfg}</h2>'); cur=cfg
        figs_html.append(f'<div style="font:13px monospace;color:#444">{g["state"]} &nbsp; '
                         f'{g["frame"]} &nbsp; <b>dT={dT:+.0f} dH={dH:+.0f} dG={dG:+.0f} mm</b></div>')
        figs_html.append(fig.to_html(full_html=False, include_plotlyjs=('cdn' if first else False),
                                     default_height=420))
        first=False
        flag = 'OK' if abs(dT)<=80 else ('CHECK' if abs(dT)<=150 else 'MISS')
        table.append((cfg,g['state'],g['frame'][-26:],dT,dH,dG,flag))

    # summary table
    rows=''.join(
        f'<tr style="background:{"#fff" if f=="OK" else "#fff5e6" if f=="CHECK" else "#ffe6e6"}">'
        f'<td>{c}</td><td>{s}</td><td style="font:11px monospace">{fr}</td>'
        f'<td align=right>{dT:+.0f}</td><td align=right>{dH:+.0f}</td><td align=right>{dG:+.0f}</td>'
        f'<td align=center>{f}</td></tr>'
        for c,s,fr,dT,dH,dG,f in table)
    thead=('<tr><th>config</th><th>state</th><th>frame</th><th>dT</th><th>dH</th><th>dG</th><th>flag</th></tr>')
    # medians
    arr=np.array([[t[3],t[4],t[5]] for t in table],float); med=np.median(np.abs(arr),0)
    summary=(f'<p style="font:14px sans-serif">Frames: {len(table)} &nbsp; '
             f'median |dT|={med[0]:.0f} |dH|={med[1]:.0f} |dG|={med[2]:.0f} mm &nbsp; '
             f'(errors = OUR cone − GT, mm). Red=our cone, Magenta=GT, Orange=isolated cone bump.</p>')
    html=(f'<!doctype html><meta charset=utf-8><title>ROI vs GT comparison</title>'
          f'<body style="font-family:sans-serif;max-width:1500px;margin:auto;padding:12px">'
          f'<h1>ROI extraction vs Ground-Truth — side by side</h1>{summary}'
          f'<table border=1 cellpadding=4 style="border-collapse:collapse;font:13px sans-serif">'
          f'{thead}{rows}</table>'+''.join(figs_html)+'</body>')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out,'w',encoding='utf-8').write(html)
    print('wrote', a.out, f'({len(table)} frames, median |dT|={med[0]:.0f}mm)')

if __name__=='__main__':
    main()
