import sys, os, glob
import numpy as np
sys.argv=['x']
exec(open(os.path.join(os.path.dirname(__file__),'pcd_inspect.py')).read().split('if __name__')[0])

# For each FRONT job: take a few roiCrop PCDs, compute the near-Z cone-height
# structure's dense min-X and max-X, compare to that job's VA cone median.
ROI_ROOT = r"K:\TUAS\South_Endfeed\ActualJob\XT_Batch\FRONT\Batch_result_ROI2"
VA_ROOT  = r"K:\TUAS\South_Endfeed\ActualJob\XT_Batch\FRONT\Batch_result"

def dense_extreme(vals, win=40.0, minpts=20, find_max=False):
    v=np.sort(vals)
    if len(v)<minpts: return v[-1] if find_max else v[0]
    if find_max:
        hi=len(v)-1; lo=hi
        while hi>=0:
            while lo>0 and v[hi]-v[lo-1]<=win: lo-=1
            if hi-lo+1>=minpts: return float(v[lo:hi+1].mean())
            hi-=1; lo=min(lo,hi)
        return float(v[-1])
    else:
        lo=0; hi=0
        while lo<len(v):
            while hi<len(v)-1 and v[hi+1]-v[lo]<=win: hi+=1
            if hi-lo+1>=minpts: return float(v[lo:hi+1].mean())
            lo+=1; hi=max(hi,lo)
        return float(v[0])

def va_cone_med(job):
    res=os.path.join(VA_ROOT,job,job+"_res.txt")
    if not os.path.exists(res): return None
    L=open(res).read().splitlines(); h=L[0].split(';')
    ip=h.index('XT_prob'); ix=h.index('cone_x')
    xs=[]
    for ln in L[1:]:
        c=ln.split(';')
        if len(c)<=ix: continue
        try:
            if float(c[ip])>=70 and float(c[ix])!=-10000: xs.append(float(c[ix]))
        except: pass
    return np.median(xs) if xs else None

for job in sorted(os.listdir(ROI_ROOT)):
    jd=os.path.join(ROI_ROOT,job)
    if not os.path.isdir(jd) or 'TMINI' in job: continue
    vmed=va_cone_med(job)
    crops=glob.glob(os.path.join(jd,'*','*_roiCrop.pcd'))
    if not crops: continue
    minxs=[]; maxxs=[]
    for cp in crops[::max(1,len(crops)//8)][:8]:
        pts=read_pcd(cp)
        v=pts[~((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))]
        if len(v)<50: continue
        # near-Z slab
        zmin=dense_extreme(v[:,2],40,20,False)
        s=v[(v[:,2]>=zmin-50)&(v[:,2]<=zmin+300)]
        if len(s)<50: continue
        minxs.append(dense_extreme(s[:,0],40,20,False))
        maxxs.append(dense_extreme(s[:,0],40,20,True))
    if not minxs: continue
    mn=np.median(minxs); mx=np.median(maxxs)
    tag=' '.join(job.split('_')[4:9])
    if vmed is None:
        print(f"{tag:34s} VA=None  nearZ minX~{mn:6.0f} maxX~{mx:6.0f}")
    else:
        dmin=abs(mn-vmed); dmax=abs(mx-vmed)
        pick='MIN-X' if dmin<dmax else 'MAX-X'
        print(f"{tag:34s} VA~{vmed:6.0f}  minX~{mn:6.0f}(d{dmin:4.0f})  maxX~{mx:6.0f}(d{dmax:4.0f})  -> {pick}")
