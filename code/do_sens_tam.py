#!/usr/bin/env python3
"""
do_sens_tam.py -- thuoc do vong 5-C cho SENS Miehe (ghi truoc: Pervious-Concrete
doc/CA-VONG5-C-2026-11-02.md).  Chi numpy/h5py; matplotlib tuy chon (--png).

  (a) duong tam BAM VET tu dinh khia x0 = 0.50 (y0 = 0.50), cot rong ell/5, tach cum
      theo y (khe > 1.5h), chon cum gan diem truoc; dung khi y < 2 ell; goc TB theo
      cua so 2/4/6 ell (goc_theo_doan cua measure_kink.py).
  (b) x cham canh duoi: thanh phan lien thong d>=0.2 chua nut gan dinh khia, nut canh
      duoi gan diem tam cuoi nhat.
  (c) Hausdorff hai duong tam.

Dung:
  python do_sens_tam.py A.xdmf B.xdmf --nhan Q1 DG0 --ell 0.01 [--png hinh.png] [--ve]
"""
import argparse, math, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_kink import goc_theo_doan, ve_ascii


def doc(path):
    import h5py
    h5 = os.path.splitext(path)[0] + ".h5"
    with h5py.File(h5, "r") as f:
        xy = f["Mesh/mesh/geometry"][()][:, :2]
        g = f["Function/damage"]
        k = sorted(g.keys(), key=float)[-1]          # buoc cuoi
        d = np.ravel(g[k][()])
    return xy, d


def kich_thuoc_luoi(xy):
    u = np.unique(np.round(xy[:, 0], 12))
    return float(np.min(np.diff(u)))


def duong_tam_bam(xy, d, ell, hm, x0=0.50, y0=0.50, dmin=0.2, dnguong=0.5):
    h = ell / 5.0
    m = (xy[:, 0] >= x0) & (d >= dmin)
    x, y, w = xy[m, 0], xy[m, 1], d[m]
    canh = np.arange(x0, xy[:, 0].max() + h, h)
    idx = np.digitize(x, canh) - 1
    ra, ytr, ly_do = [], y0, "het mien"
    for i in range(len(canh) - 1):
        s = idx == i
        if not s.any():
            continue
        ys, ws = y[s], w[s]
        o = np.argsort(ys); ys, ws = ys[o], ws[o]
        cat = np.where(np.diff(ys) > 1.5 * hm)[0] + 1
        cums = [(a, b) for a, b in zip(np.split(ys, cat), np.split(ws, cat)) if b.max() >= dnguong]
        if not cums:
            continue
        tt = [float(np.average(a, weights=b)) for a, b in cums]
        yc = min(tt, key=lambda t: abs(t - ytr))
        if abs(yc - ytr) > 10 * ell:
            ly_do = f"mat vet tai x={0.5*(canh[i]+canh[i+1]):.4f}"
            break
        if yc < 2 * ell:
            ly_do = f"y<2ell tai x={0.5*(canh[i]+canh[i+1]):.4f}"
            break
        ra.append((0.5 * (canh[i] + canh[i + 1]), yc)); ytr = yc
    return np.asarray(ra), ly_do


def cham_canh_duoi(xy, d, hm, tam, dmin=0.2, xt=(0.50, 0.50)):
    """Thanh phan lien thong (luoi deu: lan can <= 1.5h) chua nut gan dinh khia."""
    m = np.where(d >= dmin)[0]
    I = np.round(xy[m, 0] / hm).astype(int); J = np.round(xy[m, 1] / hm).astype(int)
    vi = {(a, b): k for k, (a, b) in enumerate(zip(I, J))}
    goc = int(np.argmin(np.hypot(xy[m, 0] - xt[0], xy[m, 1] - xt[1])))
    thay = np.zeros(len(m), bool); thay[goc] = True; st = [goc]
    while st:
        k = st.pop()
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                q = vi.get((I[k] + da, J[k] + db))
                if q is not None and not thay[q]:
                    thay[q] = True; st.append(q)
    tp = m[thay]
    day = tp[xy[tp, 1] <= 1e-8]
    tat_ca_day = m[xy[m, 1] <= 1e-8]
    goc_mau = tat_ca_day[(xy[tat_ca_day, 0] <= 0.05) | (xy[tat_ca_day, 0] >= 0.95)]
    goc_ngoai = np.setdiff1d(goc_mau, tp)
    tt = {"so_nut_thanh_phan": int(len(tp)),
          "nut_day_trong_thanh_phan": int(len(day)),
          "khoang_x_day_trong_thanh_phan": [float(xy[day, 0].min()), float(xy[day, 0].max())] if len(day) else None,
          "nut_day_goc_mau_ngoai_thanh_phan_bo": int(len(goc_ngoai)),
          "x_cham": None}
    if len(day) and len(tam):
        P = tam[-1]
        k = day[np.argmin(np.hypot(xy[day, 0] - P[0], xy[day, 1] - P[1]))]
        tt["x_cham"] = float(xy[k, 0])
    return tt


def tam_theo_hang(xy, d, ell, hm, xmin=0.45, xmax=0.95, ymax=0.48, dmin=0.2, dnguong=0.5):
    """BO SUNG SAU KHI THAY SO (khong vao phan quyet): tam theo HANG y (cho vet doc > 45 do),
    trong hop x in [xmin, xmax], 2ell <= y <= ymax; cum theo x nhu duong_tam_bam, lay cum
    trai nhat (vet tu khia). Tra ve (x, y) xep theo x tang."""
    m = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= 2 * ell) & (xy[:, 1] <= ymax) & (d >= dmin)
    ra = []
    for yv in np.unique(np.round(xy[m, 1], 9)):
        s = m & (np.abs(xy[:, 1] - yv) < 1e-9)
        xs, ws = xy[s, 0], d[s]; o = np.argsort(xs); xs, ws = xs[o], ws[o]
        cat = np.where(np.diff(xs) > 1.5 * hm)[0] + 1
        cums = [(a, b) for a, b in zip(np.split(xs, cat), np.split(ws, cat)) if b.max() >= dnguong]
        if cums:
            a, b = cums[0]; ra.append((float(np.average(a, weights=b)), float(yv)))
    ra = np.asarray(ra)
    return ra[np.argsort(ra[:, 0])] if len(ra) else ra


def goc_hang(tam, Lam):
    """Goc TB (do, so voi truc x) tu cua so dai Lam theo y, buoc Lam/2, khop x(y)."""
    if len(tam) < 3:
        return None, 0
    t = tam[np.argsort(tam[:, 1])]; y = t[:, 1]; gs = []; yb = y[0]
    while True:
        i, j = np.searchsorted(y, yb), np.searchsorted(y, yb + Lam)
        if j - i < 3 or y[j - 1] - y[i] < 0.8 * Lam:
            break
        a = np.polyfit(t[i:j, 1], t[i:j, 0], 1)[0]
        gs.append(math.degrees(math.atan(1.0 / a)) if a != 0 else -90.0)
        yb += 0.5 * Lam
    return (float(np.mean(gs)) if gs else None), len(gs)


def hausdorff(A, B):
    D = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
    return float(max(D.min(1).max(), D.min(0).max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xdmf", nargs=2)
    p.add_argument("--nhan", nargs=2, default=["Q1", "DG0"])
    p.add_argument("--ell", type=float, default=0.01)
    p.add_argument("--png")
    p.add_argument("--json")
    p.add_argument("--ve", action="store_true")
    a = p.parse_args()
    ell = a.ell
    kq, tams = {}, {}
    for nh, f in zip(a.nhan, a.xdmf):
        xy, d = doc(f); hm = kich_thuoc_luoi(xy)
        tam, ly = duong_tam_bam(xy, d, ell, hm)
        tams[nh] = (xy, d, tam)
        r = {"tep": os.path.basename(f), "h_luoi": hm, "so_diem_tam": int(len(tam)), "dung": ly,
             "x_tam": [float(tam[0, 0]), float(tam[-1, 0])] if len(tam) else None,
             "diem_cuoi": [float(v) for v in tam[-1]] if len(tam) else None}
        for k in (2, 4, 6):
            g, dl = goc_theo_doan(tam, k * ell)
            r[f"goc_{k}ell"] = float(np.mean(g)) if len(g) else None
            r[f"cuaso_{k}ell"] = int(len(g))
        r["day"] = cham_canh_duoi(xy, d, hm, tam)
        kq[nh] = r
        if a.ve:
            print(f"\n=== {nh}: {f}"); ve_ascii(xy, d, tam)
        print(f"{nh}: {len(tam)} diem tam, {ly}; goc 2/4/6ell = "
              + " / ".join("--" if r[f'goc_{k}ell'] is None else f"{r[f'goc_{k}ell']:.2f} ({r[f'cuaso_{k}ell']})"
                           for k in (2, 4, 6))
              + f"; x_cham = {r['day']['x_cham']}")
    print("BO SUNG sau khi thay so (tam theo hang y, khong vao phan quyet):")
    hang = {}
    for nh in a.nhan:
        xy, d, _ = tams[nh]; th = tam_theo_hang(xy, d, ell, kich_thuoc_luoi(xy)); hang[nh] = th
        r = {"so_diem": int(len(th)), "x": [float(th[0, 0]), float(th[-1, 0])], "y": [float(th[:, 1].min()), float(th[:, 1].max())]}
        for k in (2, 4, 6):
            g, n = goc_hang(th, k * ell); r[f"goc_{k}ell"] = g; r[f"cuaso_{k}ell"] = n
        kq[nh]["bo_sung_hang"] = r
        print(f"  {nh}: {len(th)} diem, x {r['x'][0]:.3f}-{r['x'][1]:.3f}, y {r['y'][0]:.3f}-{r['y'][1]:.3f}; goc 2/4/6ell = "
              + " / ".join(f"{r[f'goc_{k}ell']:.2f} ({r[f'cuaso_{k}ell']})" if r[f'goc_{k}ell'] is not None else "--" for k in (2, 4, 6)))
    dhh = hausdorff(*(hang[n] for n in a.nhan))
    kq["bo_sung_hausdorff_hang_mm"] = dhh
    print(f"  Hausdorff (tam theo hang) = {dhh:.4f} mm = {dhh/ell:.2f} ell")
    A, B = (tams[n][2] for n in a.nhan)
    dh = hausdorff(A, B) if len(A) and len(B) else None
    kq["hausdorff_mm"] = dh; kq["hausdorff_ell"] = dh / ell if dh is not None else None
    print(f"Hausdorff = {dh:.4f} mm = {dh/ell:.2f} ell" if dh is not None else "Hausdorff: --")
    if a.json:
        with open(a.json, "w") as fo:
            json.dump(kq, fo, indent=1, ensure_ascii=False)
    if a.png:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
        for axi, nh in zip(ax, a.nhan):
            xy, d, tam = tams[nh]
            n = int(round(math.sqrt(len(d))))
            o = np.lexsort((xy[:, 0], xy[:, 1]))
            axi.imshow(d[o].reshape(n, n), origin="lower", extent=[0, 1, 0, 1], cmap="Greys", vmin=0, vmax=1)
            for k2, c in ((nh, "r"), ([q for q in a.nhan if q != nh][0], "c")):
                th = hang[k2]
                axi.plot(th[:, 0], th[:, 1], c + "x", ms=2, lw=0, label=f"hang {k2} (bo sung)")
                t = tams[k2][2]
                if len(t):
                    axi.plot(t[:, 0], t[:, 1], c + ".-", ms=2, lw=0.8, label=f"tam {k2}")
            if kq[nh]["day"]["x_cham"] is not None:
                axi.plot(kq[nh]["day"]["x_cham"], 0, "g^", ms=8, label="cham day")
            axi.axhline(2 * ell, c="y", lw=0.6); axi.set_title(f"{nh}: d cuoi + duong tam")
            axi.legend(fontsize=7, loc="upper left")
        fig.tight_layout(); fig.savefig(a.png, dpi=130)


if __name__ == "__main__":
    main()
