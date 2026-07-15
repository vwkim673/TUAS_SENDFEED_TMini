import numpy as np, os, collections
import roi_fullpipe_eval as R, roi_containment as C, roi_cone_extract as E

def dz(v,w=40,m=30,mx=False): return R.dense_extreme(np.asarray(v),w,m,find_max=mx)

def target_position(cone, tor):
    if cone is None or len(cone)<30: return None
    zNear=dz(cone[:,2],40,30,False)
    zs=cone[(cone[:,2]>=zNear-100)&(cone[:,2]<=zNear+300)]
    if len(zs)<5: zs=cone
    if not tor:
        mx=dz(zs[:,0],40,30,True); filt=zs[(zs[:,0]>=mx-300)&(zs[:,0]<=mx+100)]
    else:
        mn=dz(zs[:,0],40,30,False); filt=zs[(zs[:,0]>=mn-100)&(zs[:,0]<=mn+300)]
    if len(filt)<5: filt=zs
    # densest 30mm Y slice
    ylo,yhi=filt[:,1].min(),filt[:,1].max(); best=None;bn=0
    cyv=yhi
    while cyv>ylo:
        sl=filt[(filt[:,1]>=cyv-30)&(filt[:,1]<cyv)]
        if len(sl)>bn: bn=len(sl);best=sl
        cyv-=30
    mpy=best if best is not None and len(best)>0 else filt
    # densest 20mm X slice and Z slice within mpy
    def densest(arr,ax):
        lo,hi=arr[:,ax].min(),arr[:,ax].max(); b=None;bb=0;c=lo
        while c<hi:
            sl=arr[(arr[:,ax]>=c)&(arr[:,ax]<c+20)]
            if len(sl)>bb: bb=len(sl);b=sl
            c+=20
        return b if b is not None else arr
    mpx=densest(mpy,0); mpz=densest(mpy,2)
    mpx_minZ=mpx[:,2].min(); mpz_minX=mpz[:,0].min(); mpz_maxX=mpz[:,0].max(); mpz_minY=mpz[:,1].min()
    refx = mpz_minX if tor else mpz_maxX
    d=(mpy[:,0]-refx)**2+(mpy[:,2]-mpx_minZ)**2
    pt=mpy[int(np.argmin(d))].copy(); pt[1]=mpz_minY
    return pt

res=collections.defaultdict(list)
for g in R.load_gt(R.GT_ROOT):
    if g['state']!='CONLY': continue
    jp,chs,sprd,spos=R.parse_job(g['job']); sensor=R.frame_sensor(g['frame']); key=R.config_key(sensor,jp,chs)
    pts=R.load_cloud(g['job'],g['frame'])
    if pts is None: continue
    cg,ct,cy=R.PRESETS[key]; tor=R.target_on_right(sensor,jp,chs)
    cone=E.extract_cone(R.preset_crop(pts,cg,ct),cg,ct,cy,tor)
    tp=target_position(cone,tor)
    if tp is None: continue
    gtx,gty,gtz=g['T'],g['H'],g['G']   # T->x col0, H->y col1, G->z col2
    res[key].append((tp[0]-gtx, tp[1]-gty, tp[2]-gtz, tor))
print('%-16s %4s %4s %6s %6s %6s   (err = OUR target - GT, mm; T=x H=y G=z)'%('config','n','tor','dT','dH','dG'))
aT=[];aH=[];aG=[]
for k in sorted(res):
    v=res[k]; dT=np.median([x[0] for x in v]); dH=np.median([x[1] for x in v]); dG=np.median([x[2] for x in v]); tor=v[0][3]
    aT+=[abs(x[0]) for x in v]; aH+=[abs(x[1]) for x in v]; aG+=[abs(x[2]) for x in v]
    print('%-16s %4d %4s %6.0f %6.0f %6.0f'%(k,len(v),str(tor),dT,dH,dG))
print('%-16s %4d %4s |%5.0f| |%5.0f| |%5.0f|  (median |err|)'%('-- ALL',len(aT),'',np.median(aT),np.median(aH),np.median(aG)))
