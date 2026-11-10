#!/usr/bin/env python3
"""
ck_lat_P3d.py -- dung lai truong d tren ba lat z (giua, z = 0, z = t) tu DIEM
KIEM cua benchmark_P3d.py, khong can dolfinx (chi numpy + h5py).

Diem kiem moi nhan: <ck>/r<i>of<n>_s<slot>.npz voi d (mang cuc bo, ca ghost) va
xyz = [toa do V_u (3 x nu) | toa do V_d (nd) | toa do V_H].  Lay khe co buoc
chung lon nhat cua moi nhan, gop d theo toa do, roi ghi tep XDMF/h5 2D (luoi tu
giac nx x nx) ma measure_kink.py doc duoc nhu dau ra cua benchmark.

Dung: python3 ck_lat_P3d.py ck_P3_cg1_n4 --nx 100 --nz 2 --out P3_cg1_n4_ck
  -> P3_cg1_n4_ck.xdmf/.h5 (lat giua), P3_cg1_n4_ck_z0.*, P3_cg1_n4_ck_zt.*
"""
import argparse, glob, os, re
import numpy as np
import h5py

p = argparse.ArgumentParser()
p.add_argument("ck")
p.add_argument("--nx", type=int, default=100)
p.add_argument("--nz", type=int, default=2)
p.add_argument("--out", required=True)
a = p.parse_args()
h = 1.0 / a.nx

files = glob.glob(os.path.join(a.ck, "r*of*_s*.npz"))
if not files:
    raise SystemExit(f"khong thay diem kiem trong {a.ck}")
n = int(re.search(r"of(\d+)_", files[0]).group(1))
steps = {}
for f in files:
    r = int(re.search(r"/r(\d+)of", "/" + os.path.basename(f)).group(1))
    with np.load(f) as z:
        steps.setdefault(r, {})[int(z["step"])] = f
common = set.intersection(*(set(v) for v in steps.values()))
if len(steps) != n or not common:
    raise SystemExit("thieu nhan hoac khong co buoc chung")
s = max(common)
X, D = [], []
for r in range(n):
    with np.load(steps[r][s]) as z:
        nu = len(z["u"]) // 3
        nd = len(z["d"])
        xyz = z["xyz"].reshape(-1, 3)
        X.append(xyz[nu:nu + nd])
        D.append(z["d"])
        hist = z["hist"]
X = np.concatenate(X)
D = np.concatenate(D)
print(f"  diem kiem buoc {s} ({n} nhan); F cuoi = {hist[-1,1]:.4f}  "
      f"F max = {np.abs(hist[:,1]).max():.4f} o buoc {int(np.argmax(np.abs(hist[:,1])))+1}")
ii = np.rint(X[:, 0] / h).astype(int)
jj = np.rint(X[:, 1] / h).astype(int)
kk = np.rint(X[:, 2] / h).astype(int)
N1 = a.nx + 1
gx, gy = np.meshgrid(np.arange(N1) * h, np.arange(N1) * h, indexing="ij")
geom = np.column_stack([gx.ravel(), gy.ravel()])          # chi so = i*N1 + j
q = []
for i in range(a.nx):
    for j in range(a.nx):
        q.append([i * N1 + j, (i + 1) * N1 + j, (i + 1) * N1 + j + 1, i * N1 + j + 1])
q = np.array(q, dtype=np.int64)
for k_sl, suf in ((a.nz // 2, ""), (0, "_z0"), (a.nz, "_zt")):
    sel = kk == k_sl
    g = np.full(N1 * N1, np.nan)
    g[ii[sel] * N1 + jj[sel]] = D[sel]                     # ghost trung toa do -> cung gia tri
    miss = int(np.isnan(g).sum())
    g[np.isnan(g)] = 0.0
    base = a.out + suf
    with h5py.File(base + ".h5", "w") as f:
        f["Mesh/mesh/geometry"] = geom
        f["Mesh/mesh/topology"] = q
        f["Function/damage/0"] = g
    nv, nc = len(geom), len(q)
    hb = os.path.basename(base) + ".h5"
    open(base + ".xdmf", "w").write(f"""<?xml version="1.0"?>
<Xdmf Version="3.0"><Domain><Grid Name="mesh" GridType="Uniform">
<Topology TopologyType="Quadrilateral" NumberOfElements="{nc}" NodesPerElement="4">
<DataItem Dimensions="{nc} 4" NumberType="Int" Precision="8" Format="HDF">{hb}:/Mesh/mesh/topology</DataItem></Topology>
<Geometry GeometryType="XY"><DataItem Dimensions="{nv} 2" Format="HDF">{hb}:/Mesh/mesh/geometry</DataItem></Geometry>
<Attribute Name="damage" AttributeType="Scalar" Center="Node">
<DataItem Dimensions="{nv} 1" Format="HDF">{hb}:/Function/damage/0</DataItem></Attribute>
</Grid></Domain></Xdmf>
""")
    print(f"  lat k = {k_sl} ({'giua' if not suf else suf}): {int(sel.sum())} bac tu do, "
          f"thieu {miss} nut; d>0.9: {int((g > 0.9).sum())} -> {base}.xdmf")
