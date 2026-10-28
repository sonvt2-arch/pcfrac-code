#!/usr/bin/env python3
"""duong_nut.py -- so do phu (ghi truoc, MA-DIEN-DAN-2026-10-28.md) cho ca #19720.

Dung lai doc_xdmf/duong_tam cua code/measure_kink.py (cung luat: cot rong ell/5,
san trong so dmin, cot chi tinh khi max d >= 0,5) va in:
  - duong tam (x, y) tung cot tu x0;
  - y cua duong tam tai x = 0,6 / 0,7 / 0,8 / 0,9 va tai diem cuoi (dinh vet nut xa nhat);
  - max |y - 0,5| tren duong tam;
  - do dai ma vet nut chay toi theo x (x cuoi cua duong tam).
Dung:  python3 duong_nut.py damage_<tag>.xdmf --ell 0.015 [--x0 0.5] [--dmin 0.2]
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
from measure_kink import doc_xdmf, duong_tam  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xdmf")
    p.add_argument("--ell", type=float, required=True)
    p.add_argument("--x0", type=float, default=0.5)
    p.add_argument("--dmin", type=float, default=0.2)
    a = p.parse_args()
    xy, d = doc_xdmf(a.xdmf)
    tam = duong_tam(xy, d, a.x0, a.ell / 5.0, dmin=a.dmin)
    print(f"  DUONG TAM (x0 = {a.x0}, cot = {a.ell/5:.4f} mm, dmin = {a.dmin}): {len(tam)} diem")
    if len(tam) == 0:
        print("  khong co cot nao co d >= 0,5")
        return
    for xx, yy in tam:
        print(f"    x = {xx:.4f}  y = {yy:.4f}")
    print("  Y TAI CAC MOC X (noi suy tuyen tinh tren duong tam; '--' = ngoai doan co vet nut):")
    for xm in (0.6, 0.7, 0.8, 0.9):
        if tam[0, 0] <= xm <= tam[-1, 0]:
            print(f"    x = {xm:.1f}: y = {np.interp(xm, tam[:, 0], tam[:, 1]):.4f}")
        else:
            print(f"    x = {xm:.1f}: --")
    print(f"  DIEM CUOI: x = {tam[-1,0]:.4f}, y = {tam[-1,1]:.4f}")
    k = int(np.argmax(np.abs(tam[:, 1] - 0.5)))
    print(f"  MAX |y - 0,5| = {abs(tam[k,1]-0.5):.4f} mm tai x = {tam[k,0]:.4f}")


if __name__ == "__main__":
    main()
