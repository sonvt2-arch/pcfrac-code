#!/usr/bin/env python3
"""Vong 11-B: do ngoai tuyen tu thu muc B_<tag>/ (B_<tag>_buoc.txt, B_<tag>_dthua.npz) + loaddisp_<tag>.txt.
Quy tac ghi truoc: doc/CA-VONG11-B-2026-11-17.md (repo du an).  Khong can dolfinx.

Dung: python3 do_tip_B.py THU_MUC_KET_QUA [--ell 0.01]
  Tim moi B_<tag>/ trong THU_MUC_KET_QUA; h suy tu tag (nx400 -> 1/400, P3 -> 1/100, con lai 1/200).
In bang: F_tip(l), F_tip(2l) (d >= 0.95), F_tip(l) voi d >= 0.90 (do nhay), F_init (delta 0.10, loaddisp moi),
theta_tip (day cung toi dinh o buoc dau a_tip >= 10 l), A_H / A_psiS / A_psiU o buoc 1, 100, buoc cuoi a_tip < l,
dau A_H truoc khoi nut, va sai lech a_tip ngoai tuyen so voi trong luc chay.
"""
import glob
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dinh_nut_B import dinh_nut           # noqa: E402
from tai_khoi_nut import khoi_nut          # noqa: E402


def doc_buoc(p):
    rows = {}
    for ln in open(p):
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.split()
        rows[int(v[0])] = [float(x) for x in v[1:]]     # buoc lap lai sau resume: giu dong sau cung
    st = np.array(sorted(rows))
    return st, np.array([rows[s] for s in st])


def a_tip_thua(rows, step, h, dim, thr):
    R = rows[rows[:, 0] == step]
    if len(R) == 0:
        return 0.0, (math.nan, math.nan)
    ij = R[:, 1:1 + dim]
    dd = R[:, -1] / 1e4
    S = (dd >= thr) & (ij[:, 0] * h >= 0.5 - 0.4 * h)
    ijS = ij[S]
    ng = (np.abs(ijS[:, 0] * h - 0.5) < 0.4 * h) & (np.abs(ijS[:, 1] * h - 0.5) < 0.4 * h)
    a, n = dinh_nut(ijS, ng, h, dim)
    return a, ((ijS[n, 0] * h, ijS[n, 1] * h) if n >= 0 else (math.nan, math.nan))


def a_bien_the(rows, step, h, dim, thr=0.95):
    """BIEN THE (KHONG GHI TRUOC, them 17/11 sau khi thay 3D ghi nguoc nut tach tu suon khia):
    thanh phan d >= thr noi voi BAT KY nut khia nao (moi nut khia lam nguon, ke ca suon x < 0.5);
    a = khoang cach Euclid lon nhat tu dinh ranh (0.5, 0.5) toi cac nut x >= 0.5 - 0.4h cua thanh phan."""
    R = rows[rows[:, 0] == step]
    if len(R) == 0:
        return 0.0, (math.nan, math.nan)
    ij = R[:, 1:1 + dim]
    dd = R[:, -1] / 1e4
    S = dd >= thr
    ijS = ij[S]
    x, y = ijS[:, 0] * h, ijS[:, 1] * h
    ng = (np.abs(y - 0.5) < 0.4 * h) & (x < 0.5 + 0.4 * h)
    if not ng.any():
        return 0.0, (math.nan, math.nan)
    key = {tuple(r): n for n, r in enumerate(ijS)}
    seen = np.zeros(len(ijS), bool)
    st = list(np.nonzero(ng)[0])
    seen[st] = True
    import itertools
    offs = [o for o in itertools.product((-1, 0, 1), repeat=dim) if any(o)]
    while st:
        n = st.pop()
        for o in offs:
            m = key.get(tuple(ijS[n][c] + o[c] for c in range(dim)))
            if m is not None and not seen[m]:
                seen[m] = True
                st.append(m)
    ok = seen & (x >= 0.5 - 0.4 * h)
    if not ok.any():
        return 0.0, (math.nan, math.nan)
    r = np.hypot(x - 0.5, y - 0.5)
    r[~ok] = -1
    n = int(np.argmax(r))
    return float(r[n]), (x[n], y[n])


def main():
    root = sys.argv[1]
    ell = 0.01
    if "--ell" in sys.argv:
        ell = float(sys.argv[sys.argv.index("--ell") + 1])
    print("# tag | buoc tip(l) | u_tip | F_tip(l) | F_tip(2l) | F_tip(l) d>=0.90 | F_init(0.10) moi | u_init | "
          "theta_tip(10l) | A_H b1 | A_H b100 | A_H truoc | A_psiS b1 | A_psiU b1 | A_psiU truoc | "
          "dau A_H truoc (cung dau/tong) | max|a ngoai tuyen - a log|/l | buoc cuoi | a_tip cuoi/l | BIEN THE (khong ghi truoc): F_tip(l) | theta(10l)")
    for dB in sorted(glob.glob(os.path.join(root, "B_B_*"))):
        if not os.path.isdir(dB):
            continue
        tag = os.path.basename(dB)
        fb = os.path.join(dB, f"{tag}_buoc.txt")
        fz = os.path.join(dB, f"{tag}_dthua.npz")
        fl = os.path.join(root, f"loaddisp_{tag[2:]}.txt")     # thu muc B_<tag>
        if not os.path.exists(fb):
            continue
        dim = 3 if "_P3_" in tag else 2
        h = 1 / 100 if dim == 3 else (1 / 400 if "nx400" in tag else 1 / 200)
        st, B = doc_buoc(fb)
        u, F, AH, AS, AU = B[:, 0], np.abs(B[:, 1]), B[:, 2], B[:, 3], B[:, 4]
        a = B[:, 7] * ell
        xt, yt = B[:, 8], B[:, 9]

        def first(cond):
            k = np.nonzero(cond)[0]
            return int(k[0]) if len(k) else None
        k1, k2, k10 = first(a >= ell - 1e-12), first(a >= 2 * ell - 1e-12), first(a >= 10 * ell - 1e-12)
        th = (math.degrees(math.atan2(yt[k10] - 0.5, xt[k10] - 0.5)) if k10 is not None else math.nan)
        kp = (k1 - 1) if k1 else len(st) - 1
        pre = np.arange(0, kp + 1)
        sig = np.sign(AH[kp]) if abs(AH[kp]) >= 1e-6 else 0
        nz = pre[np.abs(AH[pre]) >= 1e-6]
        same = int(np.sum(np.sign(AH[nz]) == sig)) if sig else 0
        # ngoai tuyen
        dmax, f09, fV, thV = math.nan, math.nan, math.nan, math.nan
        if os.path.exists(fz):
            rows = np.load(fz)["rows"]
            errs = []
            a09 = []
            for i, s in enumerate(st):
                errs.append(abs(a_tip_thua(rows, s, h, dim, 0.95)[0] - a[i]) / ell)
                a09.append(a_tip_thua(rows, s, h, dim, 0.90)[0])
            dmax = max(errs)
            av = [a_bien_the(rows, s_, h, dim) for s_ in st]
            aV = np.array([q[0] for q in av])
            kv1 = first(aV >= ell - 1e-12)
            kv10 = first(aV >= 10 * ell - 1e-12)
            fV = F[kv1] if kv1 is not None else math.nan
            thV = (math.degrees(math.atan2(av[kv10][1][1] - 0.5, av[kv10][1][0] - 0.5))
                   if kv10 is not None else math.nan)
            k09 = first(np.array(a09) >= ell - 1e-12)
            f09 = F[k09] if k09 is not None else math.nan
        fi, ui = math.nan, math.nan
        if os.path.exists(fl):
            ld = np.loadtxt(fl, ndmin=2)
            fi, ui, _ = khoi_nut(ld[:, 0], np.abs(ld[:, 1]), 0.10)
        i100 = int(np.nonzero(st == 100)[0][0]) if 100 in st else None
        g = lambda arr, k: f"{arr[k]:+.3e}" if k is not None else "-"
        fF = lambda k: f"{F[k]:.2f}" if k is not None else "-"
        print(f"{tag} | {st[k1] if k1 is not None else '-'} | {u[k1] if k1 is not None else math.nan:.4e} | "
              f"{fF(k1)} | {fF(k2)} | {f09:.2f} | {fi:.2f} | {ui:.4e} | {th:+.2f} | {g(AH, 0)} | {g(AH, i100)} | "
              f"{g(AH, kp)} | {g(AS, 0)} | {g(AU, 0)} | {g(AU, kp)} | {'+' if sig > 0 else ('-' if sig < 0 else '0')}"
              f" ({same}/{len(nz)}) | {dmax:.3g} | {st[-1]} | {a[-1] / ell:.2f} | {fV:.2f} | {thV:+.2f}")


if __name__ == "__main__":
    main()
