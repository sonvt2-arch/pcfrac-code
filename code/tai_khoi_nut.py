#!/usr/bin/env python3
"""Vong 10: tai khoi nut F_init (dinh truoc lan sut dau >= delta) va cuc dai toan cuc F_max tu loaddisp_*.txt.
Quy tac ghi truoc: doc/CA-VONG10-TAI-2026-11-16.md (repo du an).
Dung: python3 tai_khoi_nut.py tep1.txt [tep2.txt ...] [--delta 0.10]
"""
import sys
import numpy as np


def doc(p):
    a = np.loadtxt(p, comments="#", ndmin=2)
    return a[:, 0], np.abs(a[:, 1])


def khoi_nut(u, F, delta):
    M, iM = -1.0, 0
    for i, f in enumerate(F):
        if f > M:
            M, iM = f, i
        elif f < (1 - delta) * M:
            return M, u[iM], True
    return F.max(), u[int(F.argmax())], False


def main():
    args = sys.argv[1:]
    delta = 0.10
    if "--delta" in args:
        k = args.index("--delta"); delta = float(args[k + 1]); del args[k:k + 2]
    print("# tep | F_init (d=%.2f) | u_init | sut | F_max | u_max | F_init d=0.05 | d=0.20" % delta)
    for p in args:
        u, F = doc(p)
        fi, ui, s = khoi_nut(u, F, delta)
        f5 = khoi_nut(u, F, 0.05)[0]; f20 = khoi_nut(u, F, 0.20)[0]
        im = int(F.argmax())
        print(f"{p.split('/')[-1]} | {fi:.2f} | {ui:.4e} | {'co' if s else 'khong'} | {F[im]:.2f} | {u[im]:.4e} | {f5:.2f} | {f20:.2f}")


if __name__ == "__main__":
    main()
