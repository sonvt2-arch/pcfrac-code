"""
chan_doan_ghi_sau.py -- PHAN TU NAO GHI SAU CUNG o moi nut khi noi suy psi+ vao
khong gian lien tuc (nghien cuu E, 29/10/2026; mo ta, khong ghi truoc).

Y tuong
-------
Ban truoc khi sua (--hspace cg1 cua benchmark_A.py) goi

    psi_fn.interpolate(fem.Expression(psi_pos(u), V_d.element.interpolation_points))

DOLFINx tinh bieu thuc tren tung phan tu roi GAN gia tri vao cac bac tu do cua
phan tu do, lan luot theo thu tu phan tu cuc bo.  O nut dung chung, phan tu ghi
sau cung thang.  Script nay KHONG doan thu tu do tu ma nguon: no lam dung lenh
interpolate ay voi mot bieu thuc ma gia tri tren moi phan tu la CHI SO cua phan
tu (ham DG0 c -> c), nen gia tri nhan duoc o nut chinh la chi so phan tu thang.
Doi chieu them voi du doan "chi so phan tu ke lon nhat".

Roi xep vi tri phan tu thang so voi nut (tam phan tu tru toa do nut):
    tu giac : trên-phai (TP), tren-trai (TT), duoi-phai (DP), duoi-trai (DT)
    tam giac: 8 huong theo goc cua trong tam (D, DB, B, TB, T, TN, N, DN = dong,
              dong bac, bac, ...) cho nut goc o; nut tam o chi co 4 tam giac
              (B/N/D/T).
va dem tren toan luoi, tren hang nut y = 0,5 phia truoc mui khia, va trong dai
0,5 < x < 1, |y - 0,5| < 0,15 mm (3 ell).  In ban do ASCII "^" (thang o tren),
"v" (thang o duoi) cho dai do.

Quy uoc dau goc (measure_kink.py): goc = atan(dy/dx), y TANG len tren, nen goc
AM la vet nut re XUONG (ve phia y < 0,5).  Q1 nx = 100 1 nhan: -40,63 (xuong);
tam giac P1 crossed: +18,64 (len).

Chi mo ta mau hinh hoc cua "nguoi ghi sau"; khong suy ra quan he nhan qua.

Chay (1 nhan, trong anh dolfinx/dolfinx:stable):
    python3 chan_doan_ghi_sau.py --nx 100
"""
import argparse
import math
import numpy as np
from mpi4py import MPI
import ufl
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem import functionspace, Function

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=100)
a = p.parse_args()
comm = MPI.COMM_WORLD
if comm.size != 1:
    raise SystemExit("chi chay 1 nhan")
print(f"DOLFINx {dolfinx.__version__}  nx = {a.nx}")


def phan_tich(cell):
    if cell == "quad":
        m = mesh.create_unit_square(comm, a.nx, a.nx, mesh.CellType.quadrilateral)
    else:
        m = mesh.create_unit_square(comm, a.nx, a.nx, mesh.CellType.triangle,
                                    diagonal=mesh.DiagonalType.crossed)
    tdim = m.topology.dim
    nc = m.topology.index_map(tdim).size_local
    V = functionspace(m, ("Lagrange", 1))
    V0 = functionspace(m, ("DG", 0))
    cid = Function(V0)
    cid.x.array[np.asarray(V0.dofmap.list)[:, 0]] = np.arange(nc, dtype=float)
    ip = V.element.interpolation_points
    if callable(ip):
        ip = ip()
    f = Function(V)
    f.interpolate(fem.Expression(cid, ip))          # CUNG lenh voi --hspace cg1
    win = np.rint(f.x.array).astype(np.int64)

    # Du doan: phan tu ke co chi so cuc bo lon nhat.
    dl = np.asarray(V.dofmap.list)
    mx = np.full(len(f.x.array), -1, dtype=np.int64)
    np.maximum.at(mx, dl.ravel(), np.repeat(np.arange(nc), dl.shape[1]))
    print(f"\n=== {cell}: so phan tu {nc}, so nut {len(win)}; "
          f"nut ma phan tu thang = phan tu ke chi so lon nhat: "
          f"{int((win == mx).sum())}/{len(win)}")

    X = V.tabulate_dof_coordinates()[:, :2]
    mid = mesh.compute_midpoints(m, tdim, np.arange(nc, dtype=np.int32))[:, :2]
    r = mid[win] - X                                  # tam phan tu thang - nut
    h = 1.0 / a.nx
    # Thu tu phan tu cuc bo theo khong gian: tuong quan hang (Spearman) giua
    # chi so va toa do tam.
    def sp(u, v):
        ru = np.argsort(np.argsort(u)); rv = np.argsort(np.argsort(v))
        return float(np.corrcoef(ru, rv)[0, 1])
    print(f"  tuong quan hang chi so phan tu ~ x tam: {sp(np.arange(nc), mid[:, 0]):+.3f}; "
          f"~ y tam: {sp(np.arange(nc), mid[:, 1]):+.3f}")

    quad4 = np.where(r[:, 1] > 0, np.where(r[:, 0] > 0, "TP", "TT"),
                     np.where(r[:, 0] > 0, "DP", "DT"))
    if cell == "quad":
        lop = quad4
        ten = ["TP", "TT", "DP", "DT"]
        goc_o = np.ones(len(win), dtype=bool)
    else:
        # Nut goc o (toa do i h): 8 tam giac ke, hai trong moi o vuong; ghi o
        # vuong (TP/TT/DP/DT) va tam giac "n" (nam ngang, sat canh ngang cua o)
        # hay "d" (dung, sat canh doc).  Nut tam o ((i + 1/2) h): 4 tam giac
        # cua chinh o do: B (tren), N (duoi), D (phai), T (trai).
        frac = np.abs(X / h - np.rint(X / h)).max(axis=1)
        goc_o = frac < 1e-6
        nd = np.where(np.abs(r[:, 1]) < np.abs(r[:, 0]), "n", "d")
        tam4 = np.where(np.abs(r[:, 1]) > np.abs(r[:, 0]),
                        np.where(r[:, 1] > 0, "B", "N"), np.where(r[:, 0] > 0, "D", "T"))
        lop = np.where(goc_o, np.char.add(np.char.add(quad4, "-"), nd), tam4)
        ten = [q + "-" + t for q in ("TP", "TT", "DP", "DT") for t in ("n", "d")] + ["B", "N", "D", "T"]
    tren = r[:, 1] > 1e-12
    duoi = r[:, 1] < -1e-12

    def dem(mask, nhan):
        n = int(mask.sum())
        if n == 0:
            print(f"  {nhan}: 0 nut"); return
        s = "  ".join(f"{t} {int((lop[mask] == t).sum()):5d}" for t in ten)
        print(f"  {nhan}: {n} nut | {s} | thang o TREN {int((tren & mask).sum())}, "
              f"DUOI {int((duoi & mask).sum())}, ngang {int((~tren & ~duoi & mask).sum())}")

    noi = (X[:, 0] > 1e-9) & (X[:, 0] < 1 - 1e-9) & (X[:, 1] > 1e-9) & (X[:, 1] < 1 - 1e-9)
    hang = noi & (np.abs(X[:, 1] - 0.5) < 1e-9) & (X[:, 0] > 0.5 + 1e-9)
    dai = noi & (X[:, 0] > 0.5 + 1e-9) & (np.abs(X[:, 1] - 0.5) < 0.15 + 1e-9)
    for nm, mk in (("toan luoi (nut trong)", noi), ("hang y = 0,5, x > 0,5", hang),
                   ("dai 0,5<x<1, |y-0,5|<=0,15", dai)):
        dem(mk & goc_o, nm + (" -- nut goc o" if cell == "tri" else ""))
        if cell == "tri":
            dem(mk & ~goc_o, nm + " -- nut tam o")

    # Ban do: nut goc o trong dai, x tang sang phai, y tang len tren.
    print("  BAN DO nguoi ghi sau (nut goc o; ^ = phan tu thang o tren nut, "
          "v = o duoi, - = ngang; hang giua la y = 0,5; cot trai x = 0,5)")
    ii = np.rint(X[:, 0] / h).astype(int); jj = np.rint(X[:, 1] / h).astype(int)
    j0, j1 = int(round(0.35 / h)), int(round(0.65 / h))
    i0, i1 = int(round(0.5 / h)), a.nx
    G = {}
    for k in np.nonzero(goc_o & (ii >= i0) & (ii <= i1) & (jj >= j0) & (jj <= j1))[0]:
        G[(jj[k], ii[k])] = "^" if tren[k] else ("v" if duoi[k] else "-")
    for j in range(j1, j0 - 1, -1):
        print(f"  y={j*h:4.2f} " + "".join(G.get((j, i), " ") for i in range(i0, i1 + 1)))
    # Hang y = 0,5: tron chuoi thang tren/duoi theo x, de nhin co chu ky khong.
    kk = np.nonzero(hang & goc_o)[0]
    kk = kk[np.argsort(X[kk, 0])]
    print("  hang y = 0,5 (x tu 0,51 toi 0,99): "
          + "".join("^" if tren[k] else ("v" if duoi[k] else "-") for k in kk))
    if cell == "tri":
        kk = np.nonzero(dai & ~goc_o & (np.abs(X[:, 1] - 0.5) < 0.6 * h))[0]
        kk = kk[np.argsort(X[kk, 0] + 10 * X[kk, 1])]
        for yy in sorted(set(np.round(X[kk, 1], 6))):
            s = [k for k in kk if abs(X[k, 1] - yy) < 1e-9]
            print(f"  nut tam o y = {yy:.3f} (x tang): " + "".join(lop[k] for k in s))


for c in ("quad", "tri"):
    phan_tich(c)
