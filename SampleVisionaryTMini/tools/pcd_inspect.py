import sys, struct, numpy as np

def lzf_decompress(data, expected_len):
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        ctrl = data[i]; i += 1
        if ctrl < 32:
            length = ctrl + 1
            out += data[i:i+length]; i += length
        else:
            length = ctrl >> 5
            ref = len(out) - ((ctrl & 0x1f) << 8) - 1
            if length == 7:
                length += data[i]; i += 1
            ref -= data[i]; i += 1
            for _ in range(length + 2):
                out.append(out[ref]); ref += 1
    return bytes(out[:expected_len])

def read_pcd(path):
    with open(path, 'rb') as f:
        raw = f.read()
    # parse header
    hdr_end = raw.index(b'DATA ')
    line_end = raw.index(b'\n', hdr_end)
    header = raw[:line_end].decode('ascii', 'replace')
    fields, sizes, counts, types = [], [], [], []
    npoints = 0
    datafmt = 'ascii'
    for ln in header.splitlines():
        p = ln.split()
        if not p: continue
        if p[0] == 'FIELDS': fields = p[1:]
        elif p[0] == 'SIZE': sizes = list(map(int, p[1:]))
        elif p[0] == 'TYPE': types = p[1:]
        elif p[0] == 'COUNT': counts = list(map(int, p[1:]))
        elif p[0] == 'POINTS': npoints = int(p[1])
        elif p[0] == 'WIDTH': width = int(p[1])
        elif p[0] == 'HEIGHT': height = int(p[1])
        elif p[0] == 'DATA': datafmt = p[1]
    if npoints == 0:
        try: npoints = width * height
        except: pass
    body = raw[line_end+1:]
    stride = sum(sizes)
    if datafmt == 'binary_compressed':
        comp_size, uncomp_size = struct.unpack('<II', body[:8])
        dec = lzf_decompress(body[8:8+comp_size], uncomp_size)
        # SoA: field-by-field
        arrs = {}
        off = 0
        for fn, sz, ct in zip(fields, sizes, counts):
            cnt = npoints * ct
            blocklen = cnt * sz
            arrs[fn] = np.frombuffer(dec[off:off+blocklen], dtype='<f4').copy()
            off += blocklen
        return np.stack([arrs['x'], arrs['y'], arrs['z']], axis=1)
    elif datafmt == 'binary':
        a = np.frombuffer(body[:npoints*stride], dtype='<f4')
        a = a.reshape(npoints, stride//4)
        return a[:, :3]
    else:
        pts = []
        for ln in body.decode('ascii','replace').splitlines():
            p = ln.split()
            if len(p) >= 3: pts.append([float(p[0]),float(p[1]),float(p[2])])
        return np.array(pts)

def stats(path, label=""):
    pts = read_pcd(path)
    # drop invalid (0,0,0)
    valid = pts[~((pts[:,0]==0)&(pts[:,1]==0)&(pts[:,2]==0))]
    v = valid if len(valid) else pts
    print(f"--- {label or path}")
    print(f"  points: {len(pts)} (valid {len(valid)})")
    if len(v):
        for ax,nm in zip(range(3), ['T(x)','H(y)','G(z)']):
            print(f"  {nm}: min {v[:,ax].min():8.1f}  med {np.median(v[:,ax]):8.1f}  max {v[:,ax].max():8.1f}")
    return v

if __name__ == '__main__':
    for p in sys.argv[1:]:
        lbl = p.split('\\')[-1]
        stats(p, lbl)
