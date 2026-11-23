#!/usr/bin/env python3
"""
ly_thuyet_dinh_ly.py -- kiem so cac menh de cua doc/LY-THUYET-3-DINH-LY.md
(Pervious-Concrete, huong 3, 23/11/2026).  Chi numpy, khong FEniCS.

Bay cach luu H (dung nhu benchmark_A.py --hspace):
  dg0     psi tai TAM phan tu, mot gia tri/phan tu
  quad    psi tai diem Gauss 2x2 (dong nhat: khong anh xa)
  cg1     ghi de: nut nhan gia tri cua phan tu duyet SAU CUNG (psi_e tai nut)
  nmax    cuc dai qua phan tu ke cua psi_e(x_i)
  nmean   trung binh so hoc qua phan tu ke cua psi_e(x_i)
  l2      chieu L2 len Q1 (khoi luong day du, ve phai Gauss 2x2)
  avg     trung binh theo dien tich cua psi tai tam phan tu ke
Gia tri dung trong phuong trinh pha: H tai diem Gauss (CG1 -> noi suy song tuyen).

Chay:  python3 ly_thuyet_dinh_ly.py  [> ket-qua/2026-11-23-ly-thuyet-3/ly_thuyet_dinh_ly.txt]
"""
import numpy as np

SCH = ["dg0", "quad", "cg1", "nmax", "nmean", "l2", "avg"]
G = 0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)
QP = np.array([(a, b) for b in G for a in G])          # (xi, eta) tren [0,1]^2
LAM, MU = 0.3 / (1.3 * 0.4), 1.0 / 2.6                  # E = 1, nu = 0.3 (bien dang phang)
KB = LAM + MU                                           # mo dun khoi 2D


# ----------------------------------------------------------------- luoi Q1
class Mesh:
    """Luoi tich ten (hinh chu nhat) tren [0,1]^2; xs, ys la toa do nut."""

    def __init__(self, xs, ys):
        self.xs, self.ys = np.asarray(xs, float), np.asarray(ys, float)
        nx, ny = len(xs) - 1, len(ys) - 1
        self.nx, self.ny = nx, ny
        I, J = np.meshgrid(np.arange(nx), np.arange(ny))          # J hang, I cot
        I, J = I.ravel(), J.ravel()                               # e = J*nx + I
        nid = lambda i, j: j * (nx + 1) + i
        self.cells = np.stack([nid(I, J), nid(I + 1, J), nid(I + 1, J + 1), nid(I, J + 1)], 1)
        self.x0, self.y0 = self.xs[I], self.ys[J]
        self.hx, self.hy = np.diff(self.xs)[I], np.diff(self.ys)[J]
        X, Y = np.meshgrid(self.xs, self.ys)
        self.nodes = np.stack([X.ravel(), Y.ravel()], 1)
        self.nn, self.ne = len(self.nodes), len(self.cells)
        self.area = self.hx * self.hy
        bx = np.isclose(self.nodes[:, 0], 0) | np.isclose(self.nodes[:, 0], 1)
        by = np.isclose(self.nodes[:, 1], 0) | np.isclose(self.nodes[:, 1], 1)
        self.bnd = bx | by

    def phys(self, xi, eta):
        return self.x0 + xi * self.hx, self.y0 + eta * self.hy

    @staticmethod
    def shape(xi, eta):
        return np.array([(1 - xi) * (1 - eta), xi * (1 - eta), xi * eta, (1 - xi) * eta])


def uniform(n):
    t = np.linspace(0, 1, n + 1)
    return Mesh(t, t)


def alternating(n):
    """Kich thuoc xen ke h, 2h (khong trơn): luoi khong deu nhung van chu nhat."""
    w = np.where(np.arange(n) % 2 == 0, 1.0, 2.0)
    t = np.concatenate([[0], np.cumsum(w)]) / w.sum()
    return Mesh(t, t)


# ---------------------------------------------------------- truong vat ly
def psi_pos(exx, eyy, gxy):
    """Amor (2D): K/2 <tr>+^2 + mu dev:dev.  Loi theo eps (bo de Jensen)."""
    tr = exx + eyy
    dxx, dyy = exx - tr / 2, eyy - tr / 2
    return 0.5 * KB * np.maximum(tr, 0) ** 2 + MU * (dxx ** 2 + dyy ** 2 + 0.5 * gxy ** 2)


def u_exact(x, y):
    ux = 0.02 * (x + 0.5 * x ** 2 + 0.3 * np.sin(3 * y) + x * y)
    uy = 0.02 * (0.4 * y + 0.5 * np.cos(2 * x) * y + 0.2 * y ** 2)
    return ux, uy


def eps_exact(x, y):
    exx = 0.02 * (1 + x + y)
    eyy = 0.02 * (0.4 + 0.5 * np.cos(2 * x) + 0.4 * y)
    gxy = 0.02 * (0.9 * np.cos(3 * y) + x) + 0.02 * (-1.0 * np.sin(2 * x) * y)
    return exx, eyy, gxy


def psi_elem(m, U, xi, eta):
    """psi_e tai diem (xi, eta) cua MOI phan tu (dung bieu thuc trong phan tu e)."""
    ux, uy = U[m.cells, 0], U[m.cells, 1]                         # (ne, 4)
    dNx = np.array([-(1 - eta), (1 - eta), eta, -eta])            # d/dxi
    dNy = np.array([-(1 - xi), -xi, xi, (1 - xi)])                # d/deta
    exx = ux @ dNx / m.hx
    eyy = uy @ dNy / m.hy
    gxy = ux @ dNy / m.hy + uy @ dNx / m.hx
    return psi_pos(exx, eyy, gxy)


VERT = [(0, 0), (1, 0), (1, 1), (0, 1)]


def data(m, U):
    """Du lieu tung phan tu: tai 4 dinh (theo thu tu cells), tai tam, tai Gauss."""
    pv = np.stack([psi_elem(m, U, a, b) for a, b in VERT], 1)
    pc = psi_elem(m, U, 0.5, 0.5)
    pq = np.stack([psi_elem(m, U, a, b) for a, b in QP], 1)
    return pv, pc, pq


# ------------------------------------------------------------ toan tu luu
def op_overwrite(m, pv, order=None):
    """Ghi de: nut nhan gia tri cua phan tu ke co thu hang LON NHAT trong `order`."""
    order = np.arange(m.ne) if order is None else np.asarray(order)
    rank = np.empty(m.ne, int)
    rank[order] = np.arange(m.ne)
    best = np.full(m.nn, -1)
    np.maximum.at(best, m.cells.ravel(), np.repeat(rank, 4))
    out = np.empty(m.nn)
    for a in range(4):                                   # nut a cua phan tu thang
        e = order[best[m.cells[:, a]]]
        sel = e == np.arange(m.ne)
        out[m.cells[sel, a]] = pv[sel, a]
    return out


def op_nmax(m, pv):
    a = np.full(m.nn, -np.inf)
    np.maximum.at(a, m.cells.ravel(), pv.ravel())
    return a


def op_nmean(m, pv, order=None):
    order = np.arange(m.ne) if order is None else np.asarray(order)
    s, c = np.zeros(m.nn), np.zeros(m.nn)
    np.add.at(s, m.cells[order].ravel(), pv[order].ravel())
    np.add.at(c, m.cells.ravel(), 1.0)
    return s / c


def op_avg(m, pc, order=None):
    order = np.arange(m.ne) if order is None else np.asarray(order)
    s, c = np.zeros(m.nn), np.zeros(m.nn)
    np.add.at(s, m.cells[order].ravel(), np.repeat((m.area * pc)[order], 4))
    np.add.at(c, m.cells.ravel(), np.repeat(m.area, 4))
    return s / c


_MLOC = np.array([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]]) / 36.0


def mass_mv(m, p, lumped=False):
    loc = p[m.cells] @ _MLOC.T * m.area[:, None]
    out = np.zeros(m.nn)
    np.add.at(out, m.cells.ravel(), loc.ravel())
    return out


def rhs_l2(m, pq):
    b = np.zeros(m.nn)
    N = np.stack([Mesh.shape(a, c) for a, c in QP], 0)            # (4q, 4a)
    loc = (pq @ N) * (m.area[:, None] / 4.0)
    np.add.at(b, m.cells.ravel(), loc.ravel())
    return b


class _Perm:
    """Ban sao luoi voi phan tu danh so lai theo `order` (thu tu lap rap)."""

    def __init__(self, m, order):
        self.__dict__.update(m.__dict__)
        for k in ("cells", "x0", "y0", "hx", "hy", "area"):
            setattr(self, k, getattr(m, k)[order])


def op_l2(m, pq, tol=1e-14, order=None):
    if order is not None:
        m, pq = _Perm(m, order), pq[order]
    b = rhs_l2(m, pq)
    dg = np.zeros(m.nn)
    np.add.at(dg, m.cells.ravel(), np.repeat(m.area * 4 / 36.0, 4))
    x = b / dg
    r = b - mass_mv(m, x)
    z = r / dg
    p = z.copy()
    rz = r @ z
    nb = np.linalg.norm(b) + 1e-300
    for _ in range(5000):
        Ap = mass_mv(m, p)
        al = rz / (p @ Ap)
        x += al * p
        r -= al * Ap
        if np.linalg.norm(r) < tol * nb:
            break
        z = r / dg
        rz2 = r @ z
        p = z + rz2 / rz * p
        rz = rz2
    return x


def op_dual_bilinear(m, pc):
    """(Chi cho menh de 3c) noi suy song tuyen tu 4 tam phan tu ke tai nut TRONG:
    trong so duong, tai tao ham afin tren MOI luoi tich ten.  Nut bien: avg."""
    out = op_avg(m, pc)
    nx = m.nx
    cx = 0.5 * (m.xs[:-1] + m.xs[1:])
    cy = 0.5 * (m.ys[:-1] + m.ys[1:])
    PC = pc.reshape(m.ny, nx)
    for J in range(1, m.ny):
        for I in range(1, nx):
            s = (m.xs[I] - cx[I - 1]) / (cx[I] - cx[I - 1])
            t = (m.ys[J] - cy[J - 1]) / (cy[J] - cy[J - 1])
            out[J * (nx + 1) + I] = ((1 - s) * (1 - t) * PC[J - 1, I - 1] + s * (1 - t) * PC[J - 1, I]
                                     + s * t * PC[J, I] + (1 - s) * t * PC[J, I - 1])
    return out


def nodal(m, sch, pv, pc, pq, order=None):
    if sch == "cg1":
        return op_overwrite(m, pv, order)
    if sch == "nmax":
        return op_nmax(m, pv)
    if sch == "nmean":
        return op_nmean(m, pv, order)
    if sch == "avg":
        return op_avg(m, pc, order)
    if sch == "l2":
        return op_l2(m, pq, order=order)
    raise ValueError(sch)


def at_gauss(m, sch, pv, pc, pq, order=None, Hn=None):
    """Gia tri luu tai 4 diem Gauss moi phan tu, (ne, 4)."""
    if sch == "dg0":
        return np.repeat(pc[:, None], 4, 1)
    if sch == "quad":
        return pq.copy()
    Hn = nodal(m, sch, pv, pc, pq, order) if Hn is None else Hn
    N = np.stack([Mesh.shape(a, c) for a, c in QP], 0)
    return Hn[m.cells] @ N.T


def interp_U(m, f):
    ux, uy = f(m.nodes[:, 0], m.nodes[:, 1])
    return np.stack([ux, uy], 1)


def rate(e):
    e = np.asarray(e, float)
    return np.log2(np.abs(e[:-1]) / np.abs(e[1:]))


def hr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# =================================================================== T1
def T1_consistency():
    hr("T1 (P2) nhat quan tren du lieu tron, luoi deu: sai so tai diem Gauss")
    ns = [8, 16, 32, 64, 128]
    w = lambda x, y: 1 + x * y
    res = {s: {"Linf_R2": [], "Linf_R1": [], "bias_R2": [], "weak_R2": []} for s in SCH}
    for n in ns:
        m = uniform(n)
        U = interp_U(m, u_exact)
        pv, pc, pq = data(m, U)
        xq = np.stack([m.phys(a, b)[0] for a, b in QP], 1)
        yq = np.stack([m.phys(a, b)[1] for a, b in QP], 1)
        ex = psi_pos(*eps_exact(xq, yq))                         # R2: psi(eps(u)) dung
        W = w(xq, yq)
        for s in SCH:
            Hq = at_gauss(m, s, pv, pc, pq)
            res[s]["Linf_R2"].append(np.abs(Hq - ex).max())
            res[s]["Linf_R1"].append(np.abs(Hq - pq).max())
            res[s]["bias_R2"].append(((Hq - ex) * m.area[:, None] / 4).sum())
            res[s]["weak_R2"].append(abs(((Hq - ex) * W * m.area[:, None] / 4).sum()))
    print("h = 1/n, n =", ns, "; psi ~ %.1e" % psi_pos(*eps_exact(0.5, 0.5)))
    for key, lab in [("Linf_R1", "max |H_h - psi_h| tai Gauss (R1 = psi cua loi giai roi rac)"),
                     ("Linf_R2", "max |H_h - psi(eps(u))| tai Gauss (R2 = psi dung)"),
                     ("weak_R2", "|int (H_h - psi) (1+xy)|  (chuan yeu)"),
                     ("bias_R2", "int (H_h - psi)  CO DAU (lech mot phia?)")]:
        print("\n" + lab)
        for s in SCH:
            e = res[s][key]
            rr = rate(e) if key != "bias_R2" and min(np.abs(e)) > 1e-15 else rate(np.abs(np.array(e)) + 1e-300)
            print(f"  {s:6s} " + " ".join(f"{v:+.3e}" for v in e) + "   bac: " + " ".join(f"{r:5.2f}" for r in rr))
    return res


# =================================================================== T1b
def T1b_boundary():
    hr("T1b (P2, menh de 3b) sai so NUT: nut trong vs nut bien; luoi deu vs luoi xen ke h/2h")
    for name, mk in [("deu", uniform), ("xen ke h,2h", alternating)]:
        print(f"\nluoi {name}: max |P_h psi_h (x_i) - psi(x_i)|")
        ns = [8, 16, 32, 64, 128]
        E = {s: ([], []) for s in ["cg1", "nmax", "nmean", "avg", "l2", "dual"]}
        for n in ns:
            m = mk(n)
            U = interp_U(m, u_exact)
            pv, pc, pq = data(m, U)
            ex = psi_pos(*eps_exact(m.nodes[:, 0], m.nodes[:, 1]))
            for s in E:
                Hn = op_dual_bilinear(m, pc) if s == "dual" else nodal(m, s, pv, pc, pq)
                d = np.abs(Hn - ex)
                E[s][0].append(d[~m.bnd].max())
                E[s][1].append(d[m.bnd].max())
        for s in E:
            print(f"  {s:6s} trong: " + " ".join(f"{v:.2e}" for v in E[s][0]) + "  bac " +
                  " ".join(f"{r:4.2f}" for r in rate(E[s][0])))
            print(f"  {'':6s} bien : " + " ".join(f"{v:.2e}" for v in E[s][1]) + "  bac " +
                  " ".join(f"{r:4.2f}" for r in rate(E[s][1])))


# =================================================================== T2
def T2_positivity():
    hr("T2 (P4) don dieu / khong vuot: du lieu 0 <= psi <= 1")
    rng = np.random.default_rng(1)
    m = uniform(20)
    tests = {}
    # (a) chi thi mot phan tu (psi = 1 tren phan tu giua, 0 noi khac)
    e0 = 10 * 20 + 10
    pv = np.zeros((m.ne, 4)); pc = np.zeros(m.ne); pq = np.zeros((m.ne, 4))
    pv[e0] = 1; pc[e0] = 1; pq[e0] = 1
    tests["chi thi 1 phan tu"] = (pv, pc, pq)
    # (b) dai mot phan tu (vet nut): cot I = 10
    I = np.arange(m.ne) % 20
    band = (I == 10).astype(float)
    tests["dai rong 1 phan tu"] = (np.repeat(band[:, None], 4, 1), band.copy(), np.repeat(band[:, None], 4, 1))
    # (c) ngau nhien trong [0,1]
    pr = rng.random((m.ne, 4))
    tests["ngau nhien [0,1]"] = (pr, pr.mean(1), pr)
    for k, (pv, pc, pq) in tests.items():
        print(f"\n{k}:  min/max cua anh tai nut (CG1) hoac tai Gauss (dg0, quad); du lieu trong [0,1]")
        for s in SCH:
            v = at_gauss(m, s, pv, pc, pq) if s in ("dg0", "quad") else nodal(m, s, pv, pc, pq)
            flag = "VI PHAM P4" if (v.min() < -1e-12 or v.max() > 1 + 1e-12) else "ok"
            print(f"  {s:6s} min {v.min():+.4f}  max {v.max():+.4f}   {flag}")
    # 1D: he so tat dan cua nghich dao ma tran khoi luong P1
    n = 40
    M = (np.diag(np.full(n + 1, 4.0)) + np.diag(np.ones(n), 1) + np.diag(np.ones(n), -1)) / 6 / n
    M[0, 0] = M[-1, -1] = 2 / 6 / n
    b = np.zeros(n + 1); b[20] = b[21] = 0.5 / n           # psi = 1 tren phan tu [20,21]
    p = np.linalg.solve(M, b)
    print("\n1D P1, chieu L2 cua chi thi phan tu [x20,x21]: nut 17..24 =",
          " ".join(f"{v:+.4f}" for v in p[17:25]))
    print("  ti so lien tiep p[23]/p[22] = %.4f  (ly thuyet -2+sqrt3 = %.4f)" % (p[23] / p[22], -2 + np.sqrt(3)))


# =================================================================== T3
def T3_invariance():
    hr("T3 (P3) bat bien voi danh so phan tu va dong bien voi doi xung")
    rng = np.random.default_rng(2)
    m = uniform(16)
    U = rng.normal(size=(m.nn, 2)) * 0.01
    pv, pc, pq = data(m, U)
    perms = {"xuoi": np.arange(m.ne), "nguoc": np.arange(m.ne)[::-1], "ngau nhien": rng.permutation(m.ne)}
    print("(a) max |P_h(thu tu khac) - P_h(xuoi)| tai nut  (l2: thu tu lap rap; CG rtol 1e-14)")
    for s in ["cg1", "nmax", "nmean", "avg", "l2"]:
        base = nodal(m, s, pv, pc, pq, perms["xuoi"])
        ds = []
        for k in ["nguoc", "ngau nhien"]:
            ds.append(np.abs(nodal(m, s, pv, pc, pq, perms[k]) - base).max())
        print(f"  {s:6s} nguoc {ds[0]:.2e}   ngau nhien {ds[1]:.2e}   (|psi| max {pv.max():.2e})")
    # (b) doi xung guong x -> 1-x: du lieu doi xung, anh co doi xung khong
    n = 16
    m = uniform(n)
    xx = m.nodes[:, 0]
    Us = np.stack([0.01 * (xx - 0.5) * (1 + 0.5 * m.nodes[:, 1]) + 0.004 * (xx - 0.5) ** 3,
                   0.004 * m.nodes[:, 1] ** 2 + 0.003 * np.cos(2 * np.pi * xx) * m.nodes[:, 1]], 1)
    # Us la doi xung guong (ux phan doi xung, uy doi xung qua x = 0.5)
    pv, pc, pq = data(m, Us)
    mir = (m.nodes[:, 1] * n).round().astype(int) * (n + 1) + (n - (m.nodes[:, 0] * n).round().astype(int))
    print("(b) do bat doi xung guong  max |H(x_i) - H(R x_i)| / max H   (du lieu doi xung)")
    for s in ["cg1", "nmax", "nmean", "avg", "l2"]:
        Hn = nodal(m, s, pv, pc, pq)
        print(f"  {s:6s} {np.abs(Hn - Hn[mir]).max() / Hn.max():.2e}")
    # chieu lech cua ghi de: tuong duong lay mau truot (+h/2,+h/2)?
    m = uniform(32)
    U = interp_U(m, u_exact)
    pv, pc, pq = data(m, U)
    ex = psi_pos(*eps_exact(m.nodes[:, 0], m.nodes[:, 1]))
    d = op_overwrite(m, pv) - ex
    print("(c) ghi de, luoi deu n=32, du lieu tron: lech trung binh nut trong %+.3e (nmean %+.3e)"
          % (d[~m.bnd].mean(), (op_nmean(m, pv) - ex)[~m.bnd].mean()))


# =================================================================== T4
def T4_domination():
    hr("T4 (P1*) troi tai diem Gauss: min (H_h(x_q) - psi_e(x_q)) tren 2000 truong ngau nhien")
    rng = np.random.default_rng(3)
    worst = {s: np.inf for s in SCH}
    for trial in range(2000):
        n = int(rng.integers(2, 7))
        m = uniform(n) if trial % 2 == 0 else alternating(n)
        U = rng.normal(size=(m.nn, 2)) * 10 ** rng.uniform(-3, 0)
        pv, pc, pq = data(m, U)
        sc = pq.max() + 1e-300
        for s in SCH:
            worst[s] = min(worst[s], (at_gauss(m, s, pv, pc, pq) - pq).min() / sc)
    for s in SCH:
        print(f"  {s:6s} min tuong doi {worst[s]:+.3e}   " + ("TROI (P1* dung)" if worst[s] > -1e-12 else "khong troi"))


# =================================================================== T5
def T5_history():
    hr("T5 (P1, menh de 4) lich su: dinh di chuyen; luu nut 'max roi noi suy' vs 'noi suy roi max'")
    m = uniform(20)
    xq = np.stack([m.phys(a, b)[0] for a, b in QP], 1)
    yq = np.stack([m.phys(a, b)[1] for a, b in QP], 1)
    N = np.stack([Mesh.shape(a, c) for a, c in QP], 0)
    Hn = {s: np.zeros(m.nn) for s in ["nmax", "nmean", "avg", "l2"]}
    Hq_true = np.zeros((m.ne, 4))
    Hq_int = {s: np.zeros((m.ne, 4)) for s in Hn}
    for k in range(41):                       # dinh Gauss rong 2h chay theo x
        x0 = 0.2 + 0.6 * k / 40
        f = lambda x, y: np.exp(-((x - x0) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))
        pv = np.stack([f(*m.phys(a, b)) for a, b in VERT], 1)
        pc = f(*m.phys(0.5, 0.5))
        pq = f(xq, yq)
        Hq_true = np.maximum(Hq_true, pq)
        for s in Hn:
            p = nodal(m, s, pv, pc, pq)
            Hn[s] = np.maximum(Hn[s], p)
            Hq_int[s] = np.maximum(Hq_int[s], p[m.cells] @ N.T)
    for s in Hn:
        Hq = Hn[s][m.cells] @ N.T
        print(f"  {s:6s} min(H_nut(x_q) - max_t noi suy) = {(Hq - Hq_int[s]).min():+.2e} (>= 0)   "
              f"lech TB so H dung: {(Hq - Hq_true).mean():+.3e}  ")
    # bien the: lay max TAI GAUSS roi chieu L2 lai ve nut  -> vi pham P1
    print("\n  bien the 'max tai Gauss roi chieu L2': H^{n+1} = Pi_L2( max(H^n(x_q), psi(x_q)) )")
    H = np.zeros(m.nn)
    worst = 0.0
    for k in range(41):
        x0 = 0.2 + 0.6 * k / 40
        f = lambda x, y: np.exp(-((x - x0) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))
        g = np.maximum(H[m.cells] @ N.T, f(xq, yq))
        Hnew = op_l2(m, g)
        worst = min(worst, (Hnew - H).min())
        H = Hnew
    print(f"  min_n min_i (H_i^(n+1) - H_i^n) = {worst:+.3e}  -> " + ("VI PHAM P1" if worst < -1e-12 else "ok"))


# =================================================================== T6
def T6_order_barrier():
    hr("T6 (menh de 3) rao can bac: toan tu duong tu tam phan tu khong chinh xac cho bac 2")
    for n in [8, 16, 32]:
        m = uniform(n)
        cx, cy = m.phys(0.5, 0.5)
        f = lambda x, y: x ** 2 + y ** 2
        pc = f(cx, cy)
        pq = np.stack([f(*m.phys(a, b)) for a, b in QP], 1)
        pv = np.stack([f(*m.phys(a, b)) for a, b in VERT], 1)
        ex = f(m.nodes[:, 0], m.nodes[:, 1])
        h = 1.0 / n
        e_avg = (op_avg(m, pc) - ex)[~m.bnd]
        e_l2 = (op_l2(m, pq) - ex)[~m.bnd]
        print(f"  n={n:3d}: avg (duong) sai so nut trong min {e_avg.min():.3e} max {e_avg.max():.3e}"
              f"  = h^2/2 = {h * h / 2:.3e};  l2 (khong duong) [{e_l2.min():+.2e}, {e_l2.max():+.2e}]")
    # trong so cua bo khop bac hai binh phuong toi thieu tren 16 tam (2 vong) quanh mot nut trong:
    m = uniform(8)
    cx, cy = m.phys(0.5, 0.5)
    xi = np.array([0.5, 0.5])
    sel = (np.abs(cx - xi[0]) < 0.25) & (np.abs(cy - xi[1]) < 0.25)
    X = np.stack([np.ones(sel.sum()), cx[sel] - xi[0], cy[sel] - xi[1], (cx[sel] - xi[0]) ** 2,
                  (cx[sel] - xi[0]) * (cy[sel] - xi[1]), (cy[sel] - xi[1]) ** 2], 1)
    wts = np.linalg.pinv(X)[0]                                # gia tri tai nut = hang 0
    fq = cx[sel] ** 2 + cy[sel] ** 2
    print(f"  khop bac hai 16 tam: tong trong so {wts.sum():.6f}, trong so nho nhat {wts.min():+.4f} (<0),"
          f" sai so tren x^2+y^2: {wts @ fq - 0.5:+.1e}")
    # nut bien: moi trong so duong deu lech hang so lan nhat bang min khoang cach toi bien
    m = uniform(8)
    i = 4                                                     # nut (0.5, 0) tren bien duoi
    ce = [e for e in range(m.ne) if i in m.cells[e]]
    cyv = m.phys(0.5, 0.5)[1][ce]
    print(f"  nut bien (0.5,0): tam ke co y = {cyv} > 0 => moi trong so duong: sum w*y >= {cyv.min():.4f} = h/2;"
          " khong tai tao duoc f(x,y)=y")


if __name__ == "__main__":
    np.set_printoptions(linewidth=150)
    T1_consistency()
    T1b_boundary()
    T2_positivity()
    T3_invariance()
    T4_domination()
    T5_history()
    T6_order_barrier()
