#!/usr/bin/env python3
"""Nghien cuu G (vong 3, 30/10/2026): do lai goc nut voi MOT luat bien chung.

Luat chinh giu nguyen (measure_kink.py: x0, dmin, cot rong ell/5, cua so 2/4/6 ell,
so chinh = "goc TB" o 6 ell).  Them DUY NHAT mot phep cat bien, ghi truoc trong
doc/CA-VONG3-2026-10-30.md muc G:
  - chi dung diem duong tam (x, y) cach MOI canh mau >= 2 ell:
        2 ell <= y <= 1 - 2 ell  va  x <= 1 - 2 ell  (x >= x0 = 0.60 da du xa canh trai);
  - co "cham canh": dai hu hong (d >= dmin) cham canh duoi y = 0 hoac canh tren
    y = 1 o mot nut co x >= x0 tren truong cuoi.  Bao kem x nho nhat cham canh.

Dung:
  python3 goc_bien_chung.py --ell 0.05 nhan1=duong/dan.xdmf nhan2=ck_dir ...
In mot dong / ca: goc cu (luat chinh) 2/4/6 ell, goc moi 2/4/6 ell, so cua so, co.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_kink as mk  # noqa: E402


def doc(path):
    if os.path.isdir(path):
        import goc_tu_diem_kiem as gk
        return gk.doc_diem_kiem(path)
    return mk.doc_xdmf(path)


def goc(tam, ell, k):
    g, _ = mk.goc_theo_doan(tam, k * ell)
    return (float(np.mean(g)) if len(g) else None), len(g)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ell", type=float, required=True)
    p.add_argument("--x0", type=float, default=0.60)
    p.add_argument("--dmin", type=float, default=0.2)
    p.add_argument("--cat", type=float, default=2.0, help="khoang cach bien, don vi ell")
    p.add_argument("ca", nargs="+", help="nhan=duong_dan(.xdmf hoac thu muc diem kiem)")
    a = p.parse_args()
    ell, c = a.ell, a.cat * a.ell
    print(f"# ell={ell} x0={a.x0} dmin={a.dmin} cat bien={a.cat} ell={c:.4f} mm")
    print("# nhan | cu 2/4/6 ell (so cua so 6ell) | moi 2/4/6 ell (so cua so 6ell) | "
          "doi o 6ell | cham canh (x nho nhat) | so diem tam cu/moi")
    for item in a.ca:
        nhan, path = item.split("=", 1)
        xy, d = doc(path)
        tam = mk.duong_tam(xy, d, a.x0, ell / 5.0, dmin=a.dmin)
        if len(tam):
            giu = ((tam[:, 1] >= c) & (tam[:, 1] <= 1.0 - c) & (tam[:, 0] <= 1.0 - c)
                   & (tam[:, 0] >= c))
            # chi giu doan LIEN TUC tu dau: diem sau lan ra vung cat dau tien bi bo,
            # de duong tam khong noi hai khuc cach nhau qua vung cat.
            if (~giu).any():
                giu[np.argmax(~giu):] = False
            tam2 = tam[giu]
        else:
            tam2 = tam
        cu = [goc(tam, ell, k) for k in (2, 4, 6)]
        moi = [goc(tam2, ell, k) for k in (2, 4, 6)]
        m = (d >= a.dmin) & (xy[:, 0] >= a.x0) & ((xy[:, 1] < 1e-9) | (xy[:, 1] > 1 - 1e-9))
        co = f"CO (x={xy[m, 0].min():.3f})" if m.any() else "khong"

        def f(v):
            return "--" if v[0] is None else f"{v[0]:+.2f}"
        doi = ("--" if cu[2][0] is None or moi[2][0] is None
               else f"{moi[2][0] - cu[2][0]:+.2f}")
        print(f"{nhan} | {f(cu[0])} / {f(cu[1])} / {f(cu[2])} ({cu[2][1]}) | "
              f"{f(moi[0])} / {f(moi[1])} / {f(moi[2])} ({moi[2][1]}) | {doi} | {co} | "
              f"{len(tam)}/{len(tam2)}", flush=True)


if __name__ == "__main__":
    main()
