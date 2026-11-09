"""
Do duong tam vet nut cong cua benchmark O (tam chu L) va khoang cach Hausdorff.

Cach do (ghi truoc, doc/CA-VONG8-O-2026-11-09.md cua du an):
  Vong tron ban kinh r quanh goc lom (250, 250), r tu 2 ell, buoc h/2.  Tren moi
  cung (trong mien), lay mau d noi suy song tuyen tren luoi nut deu, buoc goc
  sao cho cung cach ~h/4.  Cac doan lien tiep co d >= DMIN (0.5) la ung vien;
  diem tam = trong tam co trong so d cua doan.  r dau: doan co tong d lon nhat;
  sau do: doan gan diem truoc nhat, chi nhan neu cach diem truoc <= 3 ell
  (theo doi lien tuc).  Dung khi khong con doan nao.  Ket qua: da tuyen tam.
  Hausdorff(A, B) = max( max_a min_b |a-b|, max_b min_a |a-b| ) tren cac diem tam,
  tinh tren khoang r CHUNG (r <= min(rmax_A, rmax_B)); bao them ban toan bo.

Dung:
  python3 do_O.py dfield_TAG.npz [--ell 10]                 # mot truong: duong tam
  python3 do_O.py dfield_A.npz --so dfield_B.npz            # them Hausdorff A-B
Chi can numpy.
"""
import argparse
import numpy as np

XC, YC = 250.0, 250.0


def load(path):
    z = np.load(path, allow_pickle=False)
    x, d, h = z["x"], z["d"], float(z["h"])
    n = int(round(500.0 / h))
    G = np.full((n + 1, n + 1), np.nan)
    i = np.rint(x[:, 0] / h).astype(int)
    j = np.rint(x[:, 1] / h).astype(int)
    G[j, i] = d
    return G, h, float(z["ell"]), z


def interp(G, h, X, Y):
    n = G.shape[0] - 1
    fx, fy = X / h, Y / h
    i0 = np.clip(np.floor(fx).astype(int), 0, n - 1)
    j0 = np.clip(np.floor(fy).astype(int), 0, n - 1)
    tx, ty = fx - i0, fy - j0
    v = ((1 - tx) * (1 - ty) * G[j0, i0] + tx * (1 - ty) * G[j0, i0 + 1]
         + (1 - tx) * ty * G[j0 + 1, i0] + tx * ty * G[j0 + 1, i0 + 1])
    inside = (X >= 0) & (X <= 500) & (Y >= 0) & (Y <= 500) & ~((X > 250) & (Y < 250))
    return np.where(inside, v, np.nan)


def centerline(G, h, ell, dmin=0.5):
    pts, prev = [], None
    r = 2.0 * ell
    while r < 700.0:
        nth = max(64, int(np.ceil(2 * np.pi * r / (h / 4))))
        th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
        X, Y = XC + r * np.cos(th), YC + r * np.sin(th)
        v = interp(G, h, X, Y)
        m = np.nan_to_num(v, nan=0.0) >= dmin
        if not m.any():
            break
        # doan lien tiep (vong tron: noi dau-cuoi)
        idx = np.nonzero(m)[0]
        runs, cur = [], [idx[0]]
        for a in idx[1:]:
            if a == cur[-1] + 1:
                cur.append(a)
            else:
                runs.append(cur); cur = [a]
        runs.append(cur)
        if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == nth - 1:
            runs[0] = runs[-1] + runs[0]; runs.pop()
        cand = []
        for rr in runs:
            rr = np.array(rr)
            wv = np.nan_to_num(v[rr])
            cx, cy = np.sum(wv * X[rr]) / wv.sum(), np.sum(wv * Y[rr]) / wv.sum()
            # trong tam cua cung: dua ve dung ban kinh r
            a = np.arctan2(cy - YC, cx - XC)
            cand.append((XC + r * np.cos(a), YC + r * np.sin(a), wv.sum()))
        if prev is None:
            c = max(cand, key=lambda t: t[2])
        else:
            c = min(cand, key=lambda t: np.hypot(t[0] - prev[0], t[1] - prev[1]))
            if np.hypot(c[0] - prev[0], c[1] - prev[1]) > 3.0 * ell:
                break
        pts.append((r, c[0], c[1]))
        prev = c
        r += h / 2.0
    return np.array(pts).reshape(-1, 3)


def hausdorff(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.nan
    D = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
    return max(D.min(axis=1).max(), D.min(axis=0).max())


def mo_ta(P, ell):
    if len(P) < 2:
        return "  (khong co duong tam)"
    x0, y0, x1, y1 = P[0, 1], P[0, 2], P[-1, 1], P[-1, 2]
    c = np.array([x1 - x0, y1 - y0]); L = np.hypot(*c)
    dev = np.abs((P[:, 1] - x0) * c[1] - (P[:, 2] - y0) * c[0]) / max(L, 1e-12)
    bien = min(x1, 500 - y1, y1, 500 - x1, abs(x1 - 250) if y1 < 250 else 1e9,
               abs(y1 - 250) if x1 > 250 else 1e9)
    return (f"  diem dau ({x0:.1f}, {y0:.1f})  diem cuoi ({x1:.1f}, {y1:.1f})  r_max = {P[-1,0]:.1f} mm\n"
            f"  goc huong day cung (do, tu +x) = {np.degrees(np.arctan2(c[1], c[0])):.1f}\n"
            f"  lech lon nhat khoi day cung = {dev.max():.2f} mm = {dev.max()/ell:.2f} ell  "
            f"(cong neu > 1 ell)\n"
            f"  khoang cach diem cuoi toi bien tu do gan nhat = {bien:.2f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("f")
    ap.add_argument("--so", default=None, help="truong thu hai (vd DG0) de tinh Hausdorff")
    ap.add_argument("--ell", type=float, default=None)
    ap.add_argument("--dmin", type=float, default=0.5)
    a = ap.parse_args()
    G, h, ell, z = load(a.f)
    ell = a.ell or ell
    P = centerline(G, h, ell, a.dmin)
    print(f"# {a.f}: hspace={z['hspace']} nhan={int(z['nprocs'])} F_pk={float(z['F_pk']):.4f} N/mm "
          f"u_pk={float(z['u_pk']):.4e} mm  ell={ell} h={h}")
    print(mo_ta(P, ell))
    print("# r_mm x_mm y_mm")
    for r, x, y in P:
        print(f"{r:8.2f} {x:9.3f} {y:9.3f}")
    if a.so:
        G2, h2, _, z2 = load(a.so)
        Q = centerline(G2, h2, ell, a.dmin)
        rc = min(P[-1, 0] if len(P) else 0, Q[-1, 0] if len(Q) else 0)
        Pc, Qc = P[P[:, 0] <= rc + 1e-9], Q[Q[:, 0] <= rc + 1e-9]
        hc, hf = hausdorff(Pc[:, 1:], Qc[:, 1:]), hausdorff(P[:, 1:], Q[:, 1:])
        print(f"# so voi {a.so}: hspace={z2['hspace']} nhan={int(z2['nprocs'])} "
              f"F_pk={float(z2['F_pk']):.4f}")
        print(mo_ta(Q, ell))
        print(f"HAUSDORFF_CHUNG = {hc:.3f} mm = {hc/ell:.3f} ell   (r <= {rc:.1f} mm)")
        print(f"HAUSDORFF_TOAN_BO = {hf:.3f} mm = {hf/ell:.3f} ell")


if __name__ == "__main__":
    main()
