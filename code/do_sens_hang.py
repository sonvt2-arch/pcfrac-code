#!/usr/bin/env python3
"""
do_sens_hang.py -- thuoc do vong 8-M cho SENS (ghi truoc: Pervious-Concrete
doc/CA-VONG8-M-2026-11-09.md).  Chi numpy/h5py; matplotlib tuy chon (--png).

  (a) duong tam theo HANG y: nut d >= 0.2 trong hop x in [0.45, 0.95], 2ell <= y <= 0.48;
      moi hang tach cum theo x (khe > 1.5h), bo cum co max d < 0.5, lay cum TRAI NHAT;
      diem tam = trong tam (trong so d) cua cum do.  Duong tam xep theo y giam (tu khia xuong).
  (b) goc: cua so dai Lam = 2/4/6 ell THEO CHIEU DAI DUONG TAM (do dai cung tich luy),
      buoc Lam/2 (cua so co < 3 diem hoac phu < 0.8 Lam thi bo qua), khop duong thang tong binh phuong toi thieu (PCA, khong phu thuoc
      huong), goc so voi truc x trong (-90, 90]; theta = trung binh cac cua so.  Chinh: 6 ell.
  (c) Hausdorff hai duong tam (mm va /ell).
  (d) x cham day lan dau: thanh phan lien thong d >= 0.2 (lan can <= 1.5h) chua nut gan
      dinh khia (0.50, 0.50); nut day (y <= 1e-8) cua thanh phan, bo dai goc mau
      (x <= 0.05 hoac x >= 0.95); x_cham = x nho nhat.  Truong d chi co buoc cuoi.

Dung:
  python do_sens_hang.py A.xdmf B.xdmf --nhan Q1 DG0 --ell 0.01 [--json kq.json] [--png h.png] [--ve]
"""
import argparse, math, os, sys, json
import numpy as np


def doc(path):
    import h5py
    h5 = os.path.splitext(path)[0] + ".h5"
    with h5py.File(h5, "r") as f:
        xy = f["Mesh/mesh/geometry"][()][:, :2]
        g = f["Function/damage"]
        k = sorted(g.keys(), key=float)[-1]
        d = np.ravel(g[k][()])
    if len(d) != len(xy):
        raise SystemExit(f"{path}: d co {len(d)} gia tri, luoi {len(xy)} nut -- khong phai truong nut")
    return xy, d, k


def kich_thuoc_luoi(xy):
    u = np.unique(np.round(xy[:, 0], 12))
    return float(np.min(np.diff(u)))


def tam_theo_hang(xy, d, ell, hm, xmin=0.45, xmax=0.95, ymax=0.48, dmin=0.2, dnguong=0.5):
    m = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= 2 * ell - 1e-12) \
        & (xy[:, 1] <= ymax + 1e-12) & (d >= dmin)
    yr = np.round(xy[:, 1], 9)
    ra = []
    for yv in np.unique(yr[m]):
        s = m & (yr == yv)
        xs, ws = xy[s, 0], d[s]
        o = np.argsort(xs); xs, ws = xs[o], ws[o]
        cat = np.where(np.diff(xs) > 1.5 * hm)[0] + 1
        cums = [(a, b) for a, b in zip(np.split(xs, cat), np.split(ws, cat)) if b.max() >= dnguong]
        if cums:
            a, b = cums[0]
            ra.append((float(np.average(a, weights=b)), float(yv)))
    ra = np.asarray(ra).reshape(-1, 2)
    return ra[np.argsort(-ra[:, 1])]


def goc_cung(tam, Lam):
    """Cua so dai Lam theo do dai cung, buoc Lam/2, khop PCA.  Tra ve (list goc, so cua so roi nhau)."""
    if len(tam) < 3:
        return [], 0
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(tam, axis=0).T))])
    gs, sb = [], 0.0
    while sb + Lam <= s[-1] + 1e-12:
        i, j = np.searchsorted(s, sb), np.searchsorted(s, sb + Lam, side="right")
        sb += 0.5 * Lam
        if j - i < 3 or s[j - 1] - s[i] < 0.8 * Lam:
            continue          # cua so thieu diem (khe tren duong tam): bo qua, khong dung
        P = tam[i:j] - tam[i:j].mean(0)
        v = np.linalg.svd(P, full_matrices=False)[2][0]
        if v[0] < 0 or (v[0] == 0 and v[1] < 0):
            v = -v
        g = math.degrees(math.atan2(v[1], v[0]))
        gs.append(90.0 if g <= -90.0 else g)
    return gs, int(s[-1] // Lam)


def hausdorff(A, B):
    D = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
    return float(max(D.min(1).max(), D.min(0).max()))


def cham_day(xy, d, hm, dmin=0.2, xt=(0.50, 0.50)):
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
    day = tp[(xy[tp, 1] <= 1e-8) & (xy[tp, 0] > 0.05) & (xy[tp, 0] < 0.95)]
    return {"nut_thanh_phan": int(len(tp)), "nut_day": int(len(day)),
            "x_cham_dau": float(xy[day, 0].min()) if len(day) else None,
            "khoang_x_day": [float(xy[day, 0].min()), float(xy[day, 0].max())] if len(day) else None}


def ve_ascii(xy, d, tam, nc=100, nr=40):
    G = np.zeros((nr, nc))
    ix = np.clip((xy[:, 0] * nc).astype(int), 0, nc - 1); iy = np.clip((xy[:, 1] * nr).astype(int), 0, nr - 1)
    np.maximum.at(G, (iy, ix), d)
    C = np.where(G >= 0.9, "#", np.where(G >= 0.5, "+", np.where(G >= 0.2, ".", " ")))
    for x, y in tam:
        C[min(int(y * nr), nr - 1), min(int(x * nc), nc - 1)] = "o"
    return "\n".join("|" + "".join(r) + "|" for r in C[::-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xdmf", nargs=2)
    p.add_argument("--nhan", nargs=2, default=["Q1", "DG0"])
    p.add_argument("--ell", type=float, default=0.01)
    p.add_argument("--json"); p.add_argument("--png"); p.add_argument("--ve", action="store_true")
    a = p.parse_args(); ell = a.ell
    kq, du = {}, {}
    for nh, f in zip(a.nhan, a.xdmf):
        xy, d, buoc = doc(f); hm = kich_thuoc_luoi(xy)
        tam = tam_theo_hang(xy, d, ell, hm); du[nh] = (xy, d, tam)
        r = {"tep": os.path.basename(f), "khoa_buoc": buoc, "h_luoi": hm, "so_diem_tam": int(len(tam))}
        if len(tam):
            r["dau"] = [float(v) for v in tam[0]]; r["cuoi"] = [float(v) for v in tam[-1]]
        for k in (2, 4, 6):
            g, nrn = goc_cung(tam, k * ell)
            r[f"goc_{k}ell"] = float(np.mean(g)) if g else None
            r[f"cuaso_{k}ell"] = len(g); r[f"roinhau_{k}ell"] = nrn
        r["day"] = cham_day(xy, d, hm)
        kq[nh] = r
        print(f"{nh}: {r['tep']} (buoc {buoc}, h={hm:.4g}), {len(tam)} diem tam"
              + (f", dau ({r['dau'][0]:.3f};{r['dau'][1]:.3f}) cuoi ({r['cuoi'][0]:.3f};{r['cuoi'][1]:.3f})" if len(tam) else ""))
        print("   goc 2/4/6ell = " + " / ".join(
            "--" if r[f"goc_{k}ell"] is None else f"{r[f'goc_{k}ell']:.2f} ({r[f'cuaso_{k}ell']} cua so)" for k in (2, 4, 6)))
        print(f"   cham day: x dau = {r['day']['x_cham_dau']}, khoang {r['day']['khoang_x_day']}, nut day {r['day']['nut_day']}")
        if a.ve:
            print(ve_ascii(xy, d, tam))
    A, B = (du[n][2] for n in a.nhan)
    dh = hausdorff(A, B) if len(A) and len(B) else None
    kq["hausdorff_mm"] = dh; kq["hausdorff_ell"] = dh / ell if dh is not None else None
    print(f"Hausdorff = {dh:.4f} mm = {dh/ell:.2f} ell" if dh is not None else "Hausdorff: --")
    g1, g2 = (kq[n]["goc_6ell"] for n in a.nhan)
    if g1 is not None and g2 is not None:
        kq["delta_6ell"] = abs(g1 - g2); print(f"|theta_{a.nhan[0]} - theta_{a.nhan[1]}| (6ell) = {abs(g1-g2):.2f} do")
    if a.json:
        with open(a.json, "w") as fo:
            json.dump(kq, fo, indent=1, ensure_ascii=False)
    if a.png:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        except ImportError:
            print("(khong co matplotlib -- bo qua --png)"); return
        fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
        for axi, nh in zip(ax, a.nhan):
            xy, d, tam = du[nh]
            axi.tricontourf(xy[:, 0], xy[:, 1], d, levels=np.linspace(0, 1, 11), cmap="Greys")
            for k2, c in zip(a.nhan, ("r", "c")):
                t = du[k2][2]
                if len(t):
                    axi.plot(t[:, 0], t[:, 1], c + ".-", ms=2, lw=0.8, label=f"tam hang {k2}")
            x0 = kq[nh]["day"]["x_cham_dau"]
            if x0 is not None:
                axi.plot(x0, 0, "g^", ms=8, label="cham day dau")
            axi.set_aspect("equal"); axi.set_title(f"{nh}: d cuoi + duong tam theo hang"); axi.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(a.png, dpi=130)


if __name__ == "__main__":
    main()
