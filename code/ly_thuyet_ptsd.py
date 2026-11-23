#!/usr/bin/env python3
"""Ly thuyet huong 2 (23/11/2026): phuong trinh sua doi cua cac cach luu H.

Phan A (--mo-hinh): mo hinh so khong can FEM.  Lay mot truong chuyen vi biet
truoc, noi suy Q1 tren luoi deu, tinh psi+ (Amor) tu gradient Q1 tai TAM phan
tu (DG0, cg1avg) hay tai GOC phan tu (cg1, cg1nmax, cg1nmean), diem Gauss 2x2
(cg1l2 = chieu L2 len Q1), roi dung dung toan tu luu nhu benchmark_A.py.
  (A1) truong tron  -> sai so tai nut, kiem bac Taylor (1 hay 2).
  (A2) truong Williams mode I quanh dinh khe (khe hinh hoc: nut tren mat khe
       nhan doi) -> phiem ham J_P = int w H_P dx, w = exp(-r^2/l^2) (va
       exp(-r/l)), lech dJ = (J_P - J_DG0)/J_DG0, du doan e_F ~ -dJ/2;
       bat doi xung A_H cua ghi de (dinh nghia nhu bai, chu nhat 4l x +-4l).
Phan B (--do): bac hoi tu do tu so lieu vong 11-A (them sau khi da commit du doan).

Chi dung numpy/scipy.  Chay:  python3 ly_thuyet_ptsd.py --mo-hinh
"""
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

E, NU = 210000.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))     # bien dang phang
K2 = LAM + MU
ELL = 0.01


def psi_amor(ux_x, ux_y, uy_x, uy_y):
    tr = ux_x + uy_y
    trp = np.maximum(tr, 0.0)
    exx = ux_x - tr / 2
    eyy = uy_y - tr / 2
    exy = 0.5 * (ux_y + uy_x)
    return 0.5 * K2 * trp ** 2 + MU * (exx ** 2 + eyy ** 2 + 2 * exy ** 2)


def grad_q1(U, xi, eta, h):
    """U[..., a] gia tri nut goc a = 0:(0,0) 1:(1,0) 2:(1,1) 3:(0,1)."""
    dx = ((1 - eta) * (U[..., 1] - U[..., 0]) + eta * (U[..., 2] - U[..., 3])) / h
    dy = ((1 - xi) * (U[..., 3] - U[..., 0]) + xi * (U[..., 2] - U[..., 1])) / h
    return dx, dy


CORN = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
G2 = 0.5 + np.array([-1, 1]) / (2 * np.sqrt(3))
GP2 = np.array([[a, b] for b in G2 for a in G2])
W2 = np.full(4, 0.25)
_g3 = np.array([-np.sqrt(0.6), 0, np.sqrt(0.6)])
_w3 = np.array([5, 8, 5]) / 9.0
GP3 = np.array([[0.5 + a / 2, 0.5 + b / 2] for b in _g3 for a in _g3])
W3 = np.array([wa * wb / 4 for wb in _w3 for wa in _w3])


def q1_shape(xi, eta):
    return np.array([(1 - xi) * (1 - eta), xi * (1 - eta), xi * eta, (1 - xi) * eta])


class Luoi:
    """Luoi deu n x n o vuong [-R, R]^2, dinh khe tai goc toa do (la nut).
    khe=True: nut tren y = 0, x < 0 cua cac o PHIA TREN duoc nhan doi."""

    def __init__(self, R, h, khe):
        n = int(round(2 * R / h))
        self.n, self.h, self.R = n, h, R
        x = -R + h * np.arange(n + 1)
        I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")   # o (i, j), chi so o = j*n + i
        I, J = I.ravel(), J.ravel()
        self.cx = -R + h * (I + 0.5)
        self.cy = -R + h * (J + 0.5)
        ci = I[:, None] + CORN[None, :, 0].astype(int)
        cj = J[:, None] + CORN[None, :, 1].astype(int)
        self.X = x[ci]
        self.Y = x[cj]
        nid = cj * (n + 1) + ci
        if khe:
            j0 = n // 2
            up = (cj == j0) & (self.X < -1e-14) & (self.cy[:, None] > 0)
            nid = np.where(up, (n + 1) ** 2 + ci, nid)
        _, self.nid = np.unique(nid, return_inverse=True)
        self.nid = self.nid.reshape(nid.shape)
        self.nn = self.nid.max() + 1
        self.nc = len(I)


def williams(X, Y, cy):
    """Mode I, K = 1 (chi ty le), nhanh theta theo phia cua o."""
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    on = (np.abs(Y) < 1e-14) & (X < 0)
    th = np.where(on, np.where(cy[:, None] > 0, np.pi, -np.pi), th)
    kap = 3 - 4 * NU
    c = 1.0 / (2 * MU) * np.sqrt(r / (2 * np.pi))
    ux = c * np.cos(th / 2) * (kap - 1 + 2 * np.sin(th / 2) ** 2)
    uy = c * np.sin(th / 2) * (kap + 1 - 2 * np.cos(th / 2) ** 2)
    return ux, uy


def tron(X, Y, cy=None):
    k = 2 * np.pi / (8 * ELL)
    return (1e-3 * np.sin(k * X) * np.cos(0.7 * k * Y),
            1e-3 * np.cos(0.6 * k * X + 0.3) * np.sin(k * Y + 0.2))


def psi_tai(Ux, Uy, h, xi, eta):
    a, b = grad_q1(Ux, xi, eta, h)
    c, d = grad_q1(Uy, xi, eta, h)
    return psi_amor(a, b, c, d)


def cac_cach_luu(L, Ux, Uy):
    h = L.h
    psi_c = psi_tai(Ux, Uy, h, 0.5, 0.5)                                    # DG0
    psi_k = np.stack([psi_tai(Ux, Uy, h, *CORN[a]) for a in range(4)], 1)   # goc
    psi_g = np.stack([psi_tai(Ux, Uy, h, *GP2[q]) for q in range(4)], 1)    # Gauss 2x2
    nid = L.nid
    out = {"dg0": psi_c}
    a = np.full(L.nn, -np.inf)
    np.maximum.at(a, nid.ravel(), psi_k.ravel())
    out["nmax"] = a
    s = np.zeros(L.nn)
    cnt = np.zeros(L.nn)
    np.add.at(s, nid.ravel(), psi_k.ravel())
    np.add.at(cnt, nid.ravel(), 1.0)
    out["nmean"] = s / cnt
    s2 = np.zeros(L.nn)
    np.add.at(s2, nid.ravel(), np.repeat(psi_c, 4))
    out["avg"] = s2 / cnt
    ow = np.zeros(L.nn)
    ow[nid.ravel()] = psi_k.ravel()       # gan theo thu tu o tang dan: o sau thang (o tren-phai)
    out["ghide"] = ow
    # chieu L2 len Q1: M p = int psi_h q, ve phai Gauss 2x2 (nhu benchmark_A)
    Nq = np.stack([q1_shape(*GP2[q]) for q in range(4)], 0)                 # (4 diem, 4 ham)
    Me = h * h * np.einsum("q,qa,qb->ab", W2, Nq, Nq)
    rows = np.repeat(nid, 4, axis=1).ravel()
    cols = np.tile(nid, (1, 4)).ravel()
    M = sp.csr_matrix((np.tile(Me.ravel(), L.nc), (rows, cols)), shape=(L.nn, L.nn))
    b = np.zeros(L.nn)
    np.add.at(b, nid.ravel(), (h * h * np.einsum("q,eq,qa->ea", W2, psi_g, Nq)).ravel())
    p, info = spla.cg(M, b, rtol=1e-12, maxiter=5000)
    out["l2"] = p
    out["_psi_k"] = psi_k
    return out


def phiem_ham(L, H, w_fun):
    """int w H dx, Gauss 3x3; H theo o (DG0) hay theo nut (Q1)."""
    h = L.h
    J = 0.0
    for q in range(9):
        xi, eta = GP3[q]
        xq = L.cx + (xi - 0.5) * h
        yq = L.cy + (eta - 0.5) * h
        wq = w_fun(xq, yq)
        if H.shape[0] == L.nc:
            Hq = H
        else:
            Hq = q1_shape(xi, eta) @ H[L.nid].T
        J += W3[q] * np.sum(wq * Hq) * h * h
    return J


def bat_doi_xung(L, H):
    """A_H = (int_T H - int_B H)/(int_T + int_B), chu nhat 0 < x < 4l, |y| < 4l (Gauss 3x3)."""
    h = L.h
    T = B = 0.0
    for q in range(9):
        xi, eta = GP3[q]
        xq = L.cx + (xi - 0.5) * h
        yq = L.cy + (eta - 0.5) * h
        Hq = H if H.shape[0] == L.nc else q1_shape(xi, eta) @ H[L.nid].T
        m = (xq > 0) & (xq < 4 * ELL) & (np.abs(yq) < 4 * ELL)
        T += W3[q] * np.sum(Hq[m & (yq > 0)])
        B += W3[q] * np.sum(Hq[m & (yq < 0)])
    return (T - B) / (T + B)


def mo_hinh():
    TEN = ["nmax", "nmean", "l2", "avg", "ghide"]
    print("== A1: truong tron, sai so tai nut so voi psi(nut) chinh xac (chuan max, vung trong)")
    print("   h/l    " + "  ".join(f"{t:>10s}" for t in TEN))
    prev = None
    for m in (2, 4, 8, 16, 32):
        L = Luoi(4 * ELL, ELL / m, khe=False)
        Ux, Uy = tron(L.X, L.Y)
        out = cac_cach_luu(L, Ux, Uy)
        # psi chinh xac tai nut: gradient giai tich (sai phan trung tam rat nho)
        xn = np.zeros(L.nn); yn = np.zeros(L.nn)
        xn[L.nid.ravel()] = L.X.ravel(); yn[L.nid.ravel()] = L.Y.ravel()
        eps = 1e-7
        def g(f, x, y):
            return ((f(x + eps, y) - f(x - eps, y)) / (2 * eps), (f(x, y + eps) - f(x, y - eps)) / (2 * eps))
        fx = lambda x, y: tron(x, y)[0]
        fy = lambda x, y: tron(x, y)[1]
        a, b = g(fx, xn, yn); c, d = g(fy, xn, yn)
        ps = psi_amor(a, b, c, d)
        inner = (np.abs(xn) < 3 * ELL) & (np.abs(yn) < 3 * ELL)
        err = [np.max(np.abs(out[t] - ps)[inner]) / np.max(ps[inner]) for t in TEN]
        s = f"   {1/m:<7.4f}" + "  ".join(f"{e:10.3e}" for e in err)
        if prev is not None:
            s += "   p: " + " ".join(f"{np.log2(p0/e):5.2f}" for p0, e in zip(prev, err))
        print(s)
        prev = err

    for wten, w_fun in (("exp(-r^2/l^2)", lambda x, y: np.exp(-(x * x + y * y) / ELL ** 2)),
                        ("exp(-r/l)", lambda x, y: np.exp(-np.hypot(x, y) / ELL))):
        print(f"\n== A2: Williams mode I, khe hinh hoc, w = {wten}; dJ = (J_P - J_DG0)/J_DG0;"
              " du doan e_F ~ -dJ/2")
        print("   h/l    " + "  ".join(f"{t:>9s}" for t in TEN) + "   | J_DG0 (tuong doi J min h)")
        rows = []
        for m in (2, 4, 8, 16, 32, 64):
            L = Luoi(6 * ELL, ELL / m, khe=True)
            Ux, Uy = williams(L.X, L.Y, L.cy)
            out = cac_cach_luu(L, Ux, Uy)
            J0 = phiem_ham(L, out["dg0"], w_fun)
            dJ = [phiem_ham(L, out[t], w_fun) / J0 - 1 for t in TEN]
            AH = bat_doi_xung(L, out["ghide"]) if wten.startswith("exp(-r^2") else np.nan
            rows.append((1 / m, dJ, J0, AH))
        Jref = rows[-1][2]
        for i, (hl, dJ, J0, AH) in enumerate(rows):
            s = f"   {hl:<7.4f}" + "  ".join(f"{100*v:+8.3f}%" for v in dJ) + f"   | {J0/Jref:.5f}"
            if not np.isnan(AH):
                s += f"   A_H(ghide) = {AH:+.4f}"
            print(s)
        print("   bac p tung cap (log2 |dJ_h / dJ_h/2|):")
        for i in range(len(rows) - 1):
            p = [np.log2(abs(a / b)) for a, b in zip(rows[i][1], rows[i + 1][1])]
            pj = np.log2(abs((rows[i][2] - Jref) / (rows[i + 1][2] - Jref))) if i + 1 < len(rows) - 1 else np.nan
            s = f"   {rows[i][0]:.4f}->{rows[i+1][0]:.4f}: " + "  ".join(f"{t}={v:5.2f}" for t, v in zip(TEN, p))
            s += f"   | J_DG0 toi J(h/l=1/64): p={pj:5.2f}"
            if not np.isnan(rows[i][3]):
                s += f"  A_H p={np.log2(abs(rows[i][3]/rows[i+1][3])):5.2f}"
            print(s)


def do():
    raise SystemExit("Phan B them sau khi commit du doan.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mo-hinh", action="store_true")
    ap.add_argument("--do", action="store_true")
    a = ap.parse_args()
    if a.mo_hinh:
        mo_hinh()
    if a.do:
        do()
