#!/usr/bin/env python3
"""
measure_kink.py -- do DO PHAN GIAI GOC cua phep do huong vet nut phase-field.

Y tuong:
  Benchmark A la keo thuan mode I: khia nam ngang, tai thang dung, vet nut chay
  THANG theo phuong x.  Goc that da biet va bang 0.  Vay moi do lech cua goc
  khop duoc tu truong hu hong la SAI SO DO thuan tuy, khong lan sai so mo hinh.

  Script khop duong tam dai hu hong tren nhung doan dai ngan khac nhau, roi bao
  do lech chuan cua goc quanh 0 theo tung do dai doan.  Do chinh la do phan giai
  goc, do chu khong uoc luong.  Tu do suy nguoc sai so dinh vi duong tam delta
  theo quan he  sigma_theta ~ 2 delta / Lambda.

Dung:
  python measure_kink.py damage_WU_ell0.05.xdmf --ell 0.05
  python measure_kink.py --tu-sinh          # tu sinh du lieu gia de thu duong ong
"""
import argparse, math, os, sys
import numpy as np


# ----------------------------------------------------------------- doc du lieu
def doc_xdmf(path):
    """Tra ve (xy, d).  Thu meshio truoc, roi den h5py."""
    try:
        import meshio
        m = meshio.read(path)
        xy = np.asarray(m.points)[:, :2]
        for name, arr in (m.point_data or {}).items():
            a = np.asarray(arr).reshape(len(xy), -1)[:, 0]
            if a.min() >= -1e-6 and a.max() <= 1.0 + 1e-6:
                print(f"  meshio: dung truong '{name}'")
                return xy, a
        sys.exit("meshio doc duoc luoi nhung khong thay truong nao trong [0,1]")
    except ImportError:
        pass
    except BaseException as e:
        # meshio thoat bang sys.exit voi XDMF cua DOLFINx, nen phai bat ca
        # SystemExit chu khong chi Exception.
        print(f"  meshio khong doc duoc ({type(e).__name__}); chuyen sang h5py")

    try:
        import h5py
    except ImportError:
        sys.exit("khong co meshio lan h5py.  pip install meshio  hoac  pip install h5py")

    h5 = os.path.splitext(path)[0] + ".h5"
    if not os.path.exists(h5):
        sys.exit(f"khong thay {h5} di kem {path}")
    geom = {}; vals = {}
    with h5py.File(h5, "r") as f:
        def tham(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            a = obj[()]
            if a.ndim == 2 and a.shape[1] in (2, 3) and a.dtype.kind == "f":
                geom[name] = a
            elif a.dtype.kind == "f" and a.ndim in (1, 2):
                vals[name] = np.ravel(a)
        f.visititems(tham)
    if not geom:
        print("  cay du lieu trong tep h5:")
        with h5py.File(h5, "r") as f:
            f.visititems(lambda n, o: print("   ", n, getattr(o, "shape", ""),
                                            getattr(o, "dtype", "")))
        sys.exit("khong nhan ra toa do nut trong tep h5")
    gname = max(geom, key=lambda k: len(geom[k]))
    xy = geom[gname][:, :2]
    ung = {k: v for k, v in vals.items()
           if len(v) == len(xy) and v.min() >= -1e-6 and v.max() <= 1.0 + 1e-6}
    if not ung:
        sys.exit("khong thay truong nao cung so nut va nam trong [0,1]")
    vname = sorted(ung)[0]
    print(f"  h5py: luoi '{gname}' ({len(xy)} nut), truong '{vname}'")
    return xy, ung[vname]


def tu_sinh(ell=0.05, h=None, nghieng_do=0.0, nhieu=0.0, seed=0):
    """Sinh mot dai hu hong tong hop de thu duong ong khi chua co tep that."""
    h = h or ell / 5.0
    rng = np.random.default_rng(seed)
    n = int(1.0 / h) + 1
    g = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(g, g)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    t = math.tan(math.radians(nghieng_do))
    yc = 0.5 + t * (xy[:, 0] - 0.5)
    r = np.abs(xy[:, 1] - yc)
    D = math.pi * ell / 2.0                       # nua be rong gia do ho tro
    d = np.clip(1.0 - r / D, 0.0, 1.0) ** 2
    d[xy[:, 0] < 0.5] = 1.0                       # khia
    if nhieu:
        d = np.clip(d + rng.normal(0, nhieu, d.shape), 0, 1)
    return xy, d


# ------------------------------------------------------------------ duong tam
def duong_tam(xy, d, x0, h, dnguong=0.5, dmin=0.2):
    """Tam khoi luong theo y cua truong d, trong tung cot x rong h.

    dmin dat SAN cho trong so: nut co d < dmin khong gop vao tam.  Khong co san
    nay thi dam hu hong muc thap trai rong quanh vet nut se keo duong tam di
    lang thang, va do lech chuan cua goc bi thoi len (do duoc 12-13 do o ca
    grip clamp, 25/9/2026)."""
    m = xy[:, 0] >= x0
    x, y, dd = xy[m, 0], xy[m, 1], np.where(d[m] >= dmin, d[m], 0.0)
    if len(x) == 0:
        return np.empty((0, 2))
    canh = np.arange(x.min(), x.max() + h, h)
    idx = np.digitize(x, canh) - 1
    ra = []
    for i in range(len(canh) - 1):
        s = idx == i
        if not s.any():
            continue
        w = dd[s]
        if w.max() < dnguong:          # cot chua co vet nut di qua
            continue
        if w.sum() <= 0:
            continue
        ra.append((0.5 * (canh[i] + canh[i + 1]), float(np.average(y[s], weights=w))))
    return np.asarray(ra)


def goc_theo_doan(tam, Lam, buoc=0.5):
    """Truot cua so dai Lam doc theo x, khop bac nhat, tra ve (gocs, so_doc_lap).

    buoc = 1.0 cho cua so roi nhau; 0.5 cho chong nhau mot nua.  Cua so chong
    nhau sinh mau tuong quan, nen ham tra ve them so cua so ROI NHAU de biet co
    bao nhieu mau thuc su doc lap."""
    if len(tam) < 3:
        return np.array([]), 0
    gocs = []
    x = tam[:, 0]
    x_bd = x[0]
    while True:
        i = np.searchsorted(x, x_bd)
        j = np.searchsorted(x, x_bd + Lam)
        if j - i < 3 or j > len(tam):
            break
        seg = tam[i:j]
        if seg[-1, 0] - seg[0, 0] < 0.8 * Lam:
            break
        a = np.polyfit(seg[:, 0], seg[:, 1], 1)[0]
        gocs.append(math.degrees(math.atan(a)))
        x_bd += buoc * Lam
    doc_lap = int((x[-1] - x[0]) // Lam) if Lam > 0 else 0
    return np.asarray(gocs), doc_lap


def ve_ascii(xy, d, tam=None, nc=104, nr=34):
    """In mot ban do ASCII cua truong d de NHIN truoc khi tin bat ky con so nao."""
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    G = np.full((nr, nc), np.nan)
    ix = np.clip(((xy[:, 0] - x0) / (x1 - x0 + 1e-15) * (nc - 1)).astype(int), 0, nc - 1)
    iy = np.clip(((xy[:, 1] - y0) / (y1 - y0 + 1e-15) * (nr - 1)).astype(int), 0, nr - 1)
    for a, b, v in zip(iy, ix, d):
        G[a, b] = v if np.isnan(G[a, b]) else max(G[a, b], v)
    bang = " .:-=+*#%@"
    mark = {}
    if tam is not None and len(tam):
        for xx, yy in tam:
            a = int(np.clip((yy - y0) / (y1 - y0 + 1e-15) * (nr - 1), 0, nr - 1))
            b = int(np.clip((xx - x0) / (x1 - x0 + 1e-15) * (nc - 1), 0, nc - 1))
            mark[(a, b)] = "o"
    print()
    print("  BAN DO TRUONG HU HONG  (khoang trang = d nho, @ = d ~ 1, o = diem duong tam)")
    print(f"  truc x tu {x0:.2f} den {x1:.2f} mm ; truc y tu {y0:.2f} den {y1:.2f} mm ; y TANG len tren")
    print("  +" + "-" * nc + "+")
    for a in range(nr - 1, -1, -1):
        hang = []
        for b in range(nc):
            if (a, b) in mark:
                hang.append("o"); continue
            v = G[a, b]
            hang.append(" " if np.isnan(v) else bang[min(9, int(v * 9.999))])
        print("  |" + "".join(hang) + "|")
    print("  +" + "-" * nc + "+")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xdmf", nargs="?")
    p.add_argument("--ell", type=float, required=True)
    p.add_argument("--x0", type=float, default=0.55, help="bat dau sau mui khia")
    p.add_argument("--goc-that", type=float, default=0.0)
    p.add_argument("--tu-sinh", action="store_true")
    p.add_argument("--ve", action="store_true", help="in ban do ASCII cua truong d")
    p.add_argument("--dmin", type=float, default=0.2,
                   help="san trong so: nut co d duoi muc nay khong gop vao duong tam")
    p.add_argument("--nghieng", type=float, default=0.0, help="chi cho --tu-sinh")
    a = p.parse_args()

    if a.tu_sinh:
        print("  CHE DO TU SINH -- du lieu gia, chi de thu duong ong")
        xy, d = tu_sinh(ell=a.ell, nghieng_do=a.nghieng)
    else:
        if not a.xdmf:
            sys.exit("can duong dan .xdmf, hoac --tu-sinh")
        xy, d = doc_xdmf(a.xdmf)

    h = a.ell / 5.0
    tam = duong_tam(xy, d, a.x0, h, dmin=a.dmin)
    if a.ve:
        ve_ascii(xy, d, tam)
        print(f"  thong ke d: min {d.min():.4f}  trung binh {d.mean():.4f}  "
              f"max {d.max():.4f}  ; so nut d>0.9: {(d>0.9).sum()} / {len(d)}")
    print(f"  ell = {a.ell} mm ; h chia cot = {h:.5f} mm ; san trong so d >= {a.dmin}")
    print(f"  so diem duong tam: {len(tam)}"
          + (f" ; x tu {tam[0,0]:.3f} den {tam[-1,0]:.3f} mm" if len(tam) else ""))
    if len(tam) < 4:
        sys.exit("qua it diem duong tam -- kiem lai --x0 hoac tep dau vao")

    print()
    print("  " + "=" * 68)
    print(f"  DO PHAN GIAI GOC  (goc that = {a.goc_that:.1f} do)")
    print("  " + "=" * 68)
    print(f"  {'Lambda':>9} {'/ell':>5} {'cua so':>7} {'roi nhau':>9} "
          f"{'goc TB':>9} {'sai so':>9} {'nguon':>8} {'delta':>11}")
    for k in (2, 4, 6, 10, 20, 40):
        Lam = k * a.ell
        g, dl = goc_theo_doan(tam, Lam)
        if len(g) == 0:
            print(f"  {Lam:>9.4f} {k:>5d} {0:>7d} {dl:>9d} "
                  f"{'--':>9} {'--':>9} {'khong du':>8} {'--':>11}")
            continue
        if len(g) >= 2:
            err = float(np.std(g, ddof=1)); nguon = "do lech"
        else:
            err = abs(float(g[0]) - a.goc_that); nguon = "1 cua so"
        delta = math.radians(err) * Lam / 2.0
        print(f"  {Lam:>9.4f} {k:>5d} {len(g):>7d} {dl:>9d} "
              f"{np.mean(g):>8.2f}o {err:>8.2f}o {nguon:>8} {delta/a.ell:>7.3f} ell")
    print("  " + "=" * 68)
    print("  Cot cuoi la sai so dinh vi duong tam suy nguoc tu  sigma ~ 2 delta / Lambda.")
    print("  Neu cot do gan nhu khong doi theo Lambda thi mo hinh sai so la dung,")
    print("  va doan do toi thieu cho mot nguong goc  T  la  Lambda >= 2 delta / tan(T).")


if __name__ == "__main__":
    main()
