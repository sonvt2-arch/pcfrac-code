"""
Benchmark A -- single-edge-notched tension, phase-field, DOLFINx 0.11.

Purpose
-------
This is the first run, and its job is NOT to produce a nice crack picture. It
is to answer one question:

    is the peak load INSENSITIVE to the regularisation length?

A cohesive (Wu / Lorentz) degradation function is built so that the peak
stress equals the prescribed tensile strength f_t regardless of ell.  The
classical AT2 model is not: its peak stress scales roughly as 1/sqrt(ell).

So the acceptance test is a CONTRAST, which is much stronger than a single
number:

    MODEL = "AT2"  ->  peak load must MOVE when ell is halved
    MODEL = "WU"   ->  peak load must NOT move (within ~5 %)

If both move, the cohesive normalisation a1 is wrong.  If neither moves, the
damage field is probably not developing at all.  Either way you learn what is
broken before a campaign is built on it.

Units: mm, N, MPa.

Run:
    python benchmark_A.py --model WU  --ell 0.02
    python benchmark_A.py --model WU  --ell 0.01
    python benchmark_A.py --model AT2 --ell 0.02
    python benchmark_A.py --model AT2 --ell 0.01
"""
import argparse
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, locate_dofs_topological

# ---------------------------------------------------------------- API shims
# DOLFINx moved these between releases.  Fail loudly with the version if
# neither location works, rather than silently doing something else.
try:
    from dolfinx.fem.petsc import LinearProblem
except ImportError as e:                                       # pragma: no cover
    raise SystemExit(f"LinearProblem not found (dolfinx {dolfinx.__version__}): {e}")

from dolfinx.fem.petsc import NonlinearProblem
import inspect as _inspect
# In DOLFINx 0.11 NonlinearProblem takes petsc_options_prefix and solves itself
# through PETSc SNES; in earlier releases it was a plain problem driven by
# dolfinx.nls.petsc.NewtonSolver.  Detect which, rather than assume.
_SNES_STYLE = "petsc_options_prefix" in _inspect.signature(
    NonlinearProblem.__init__).parameters
if not _SNES_STYLE:
    from dolfinx.nls.petsc import NewtonSolver

try:
    from dolfinx.fem import set_bc as _set_bc
except ImportError:
    from dolfinx.fem.petsc import set_bc as _set_bc


# ---------------------------------------------------------------- parameters
p = argparse.ArgumentParser()
p.add_argument("--model", choices=["AT2", "WU"], default="WU")
p.add_argument("--ell", type=float, default=0.02, help="mm")
p.add_argument("--nx", type=int, default=0, help="0 = choose from ell")
p.add_argument("--steps", type=int, default=60)
p.add_argument("--umax", type=float, default=0.0,
               help="mm; 0 = 3 x the cohesive opening 2Gc/ft")
p.add_argument("--u1", type=float, default=0.0,
               help="end of the FINE loading stage; 0 = 4 x elastic limit")
p.add_argument("--n1", type=int, default=40,
               help="steps in the fine stage (resolves the peak)")
p.add_argument("--grip", choices=["clamp", "pin"], default="clamp",
               help="clamp: u_x = 0 tren ca hai canh ngam (mac dinh); "
                    "pin: mot nut goc, hanh vi truoc 25/9/2026")
p.add_argument("--notchw", type=float, default=0.05,
               help="notch HALF thickness in mm, held fixed across ell")
p.add_argument("--split", choices=["amor", "vol", "full", "spectral"], default="amor",
               help="phep tach nang luong; vol va full chi de chan doan")
p.add_argument("--pores", default=None,
               help="tep pores.npz; ap d = 1 tren vung lo rong")
p.add_argument("--sharpnotch", action="store_true",
               help="vet khia sac: mot hang nut thay cho dai notchw")
p.add_argument("--plane", choices=["strain", "stress"],
               default="strain")
p.add_argument("--hspace", choices=["dg0", "cg1"], default="dg0",
               help="khong gian cua H va psi+: dg0 la ban da sua; cg1 tai lap "
                    "dung ban TRUOC khi sua (chi de do tat, khong dung cho ket qua)")
p.add_argument("--load", choices=["tension", "shear"], default="tension",
               help="tension: SENT (keo u_y canh tren); shear: SENS (u_x canh tren, "
                    "u_y = 0 canh tren, canh duoi ngam hai phuong); --grip bo qua khi shear")
p.add_argument("--ckpt", default=None,
               help="thu muc diem kiem; ghi trang thai sau moi buoc tai")
p.add_argument("--resume", action="store_true",
               help="chay tiep tu diem kiem chung moi nhat (phai cung so nhan)")
p.add_argument("--jitter", type=float, default=0.0,
               help="nhieu luoi: dich nut trong toi da FRAC*h theo x va y; 0 = tat "
                    "(giu nguyen ket qua cu tung bit)")
p.add_argument("--seed", type=int, default=0,
               help="hat giong cua nhieu luoi (chi dung khi --jitter > 0)")
p.add_argument("--cell", choices=["quad", "tri"], default="quad",
               help="quad: tu giac Q1 (mac dinh, giu nguyen ket qua cu tung bit); "
                    "tri: tam giac P1, moi o vuong nx x nx chia cheo chu thap (crossed)")
p.add_argument("--out", default=None)
args = p.parse_args()

L = 1.0                 # plate side, mm
E0, nu = 25000.0, 0.20  # MPa
ft = 3.5                # MPa
Gc = 0.016              # N/mm  (= 16 J/m^2)
ell = args.ell

# h <= ell/5 is the community rule of thumb; ell/2 is documented as too coarse
nx = args.nx if args.nx else int(np.ceil(L / (ell / 5.0)))
if nx > 400:
    if args.nx:
        # --nx do NGUOI DUNG dat thi ton trong.  Truoc day tran 400 ap ca cho
        # --nx, nen "--nx 600" bi am tham ha xuong 400.
        if MPI.COMM_WORLD.rank == 0:
            print(f"[warn] nx = {nx} (nguoi dung dat, vuot tran cu 400); "
                  f"so nut = {(nx+1)**2} -- theo doi RAM")
    else:
        nx = 400            # keep it inside 4 GB of RAM
        if MPI.COMM_WORLD.rank == 0:
            print(f"[warn] mesh capped at {nx}; h/ell = {L/nx/ell:.2f} (want <= 0.2)")

# The displacement at which an UNNOTCHED plate would first reach f_t.  The
# notch concentrates stress, so the peak arrives earlier than this -- but it
# fixes the order of magnitude, which is what a hand-picked umax gets wrong.
# ---------------------------------------------------- the loading schedule
# The first campaign anchored umax on the elastic limit and loaded to 2e-3 mm.
# That was the wrong scale.  Full separation of a cohesive crack needs the
# COHESIVE OPENING
#         w_c = 2 Gc / ft = 0.00914 mm,
# which is 65 times the elastic limit, so the run stopped at roughly a fifth
# of the way and every reported peak was a wobble on a plateau: d_tip never
# passed 0.10 and the load ended at 92-97 % of its maximum.
#
# Two stages, because one uniform ramp to 3 w_c leaves two or three steps
# below the peak: a fine stage that resolves the peak, then a coarse stage
# that carries the crack across the ligament.
u_elastic = ft * L / E0
w_c = 2.0 * Gc / ft
if args.umax <= 0.0:
    args.umax = 3.0 * w_c
if args.u1 <= 0.0:
    args.u1 = 4.0 * u_elastic

comm = MPI.COMM_WORLD
n2 = max(1, args.steps - args.n1)
_ramp = np.concatenate([
    np.linspace(0.0, args.u1, args.n1 + 1)[1:],
    np.linspace(args.u1, args.umax, n2 + 1)[1:]])
if comm.rank == 0:
    print(f"  w_c = 2Gc/ft = {w_c:.5f} mm ; elastic limit = {u_elastic:.3e} mm "
          f"({w_c/u_elastic:.0f}x smaller)")
    print(f"  ramp: {args.n1} fine steps to {args.u1:.3e} "
          f"(step {args.u1/args.n1:.2e}), then {n2} steps to {args.umax:.3e} "
          f"(step {(args.umax-args.u1)/n2:.2e})")
if comm.rank == 0:
    print(f"model={args.model}  ell={ell} mm  nx={nx}  h={L/nx:.4f} mm  "
          f"h/ell={L/nx/ell:.2f}")

# --cell tri: cung hinh vuong, cung nx, nhung moi o vuong chia thanh BON tam
# giac boi hai duong cheo ("crossed"; DOLFINx them mot nut o tam moi o).  Chon
# crossed vi do la kieu duy nhat ma luoi bat bien qua phep doi xung guong
# y -> 1 - y cua SENT (va ca x -> 1 - x): "right" va "left" anh xa vao nhau
# qua phep doi xung do, nen moi duong cheo co huong san la mot do lech co san
# cua LUOI, chinh no co the be cong vet nut DG0 ma ta can lam doi chung thang.
# Luoi crossed van giu cac canh ngang tren y = 0.5, nen vet khia sac (hang nut
# y = 0.5) va duong nut thang giu nguyen nhu luoi tu giac.  h van la canh O
# VUONG, L/nx: nut tam o (i + 1/2, j + 1/2) h nen tieu chi |y - 0.5| < 0.4 h
# cua vet khia khong bat nut tam nao (51 nut o nx = 100, nhu Q1).
if args.cell == "tri":
    domain = mesh.create_unit_square(comm, nx, nx, mesh.CellType.triangle,
                                     diagonal=mesh.DiagonalType.crossed)
else:
    domain = mesh.create_unit_square(comm, nx, nx, mesh.CellType.quadrilateral)
if comm.rank == 0:
    _nc = domain.topology.index_map(domain.topology.dim).size_global
    _nv = domain.topology.index_map(0).size_global
    print(f"  LUOI: --cell {args.cell}"
          + ("  (tam giac P1, crossed)" if args.cell == "tri" else "  (tu giac Q1)")
          + f"  so phan tu = {_nc}  so dinh = {_nv}", flush=True)


# ------------------------------------------------------------- nhieu luoi
# --jitter FRAC --seed S: dich moi nut TRONG mot khoang gia ngau nhien toi da
# FRAC*h theo x va theo y.  Do dich tinh tu CHI SO LUOI (i, j) = round(x/h,
# y/h) cua toa do GOC va hat giong, qua ham bam splitmix64, nen nut nao cung
# nhan dung mot do dich du no nam o nhan nao, la nut so huu hay ghost: nhieu
# khong phu thuoc phan hoach.  Khong dich: nut bien (bon canh) va nut tren
# vet khia (khia sac: hang nut y = 0.5, x <= 0.5; khia day: ca dai notchw ke ca
# hai mat), dung tieu chi cua notch() ben duoi.  FRAC = 0 thi khong cham vao
# luoi, ket qua giu nguyen tung bit.
def _splitmix64(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _jitter_mesh(frac, seed):
    if args.pores:
        raise SystemExit("--jitter chua ho tro --pores")
    if args.cell != "quad":
        # Chi so (i, j) = round(x/h, y/h) gan nut tam o crossed vao nut goc o
        # ben canh: hai nut cung mot khoa bam.  Chua ho tro, dung han.
        raise SystemExit("--jitter chua ho tro --cell tri")
    h = L / nx
    X = domain.geometry.x
    ii = np.rint(X[:, 0] / h).astype(np.int64)
    jj = np.rint(X[:, 1] / h).astype(np.int64)
    xg, yg = ii * h, jj * h                    # toa do goc, dung tren luoi deu
    fixed = (ii <= 0) | (ii >= nx) | (jj <= 0) | (jj >= nx)
    if args.sharpnotch:
        fixed |= (np.abs(yg - 0.5) < 0.4 * h) & (xg < 0.5 + 0.4 * h)
    else:
        fixed |= ((np.abs(yg - 0.5) <= args.notchw + 0.4 * h)
                  & (xg < 0.5 + 0.4 * h))
    key = ((np.uint64(seed) << np.uint64(32))
           | (jj * (nx + 1) + ii).astype(np.uint64)) * np.uint64(2)
    _u = lambda z: (_splitmix64(z) >> np.uint64(11)).astype(np.float64) * 2.0 ** -53
    dx = (2.0 * _u(key) - 1.0) * frac * h
    dy = (2.0 * _u(key + np.uint64(1)) - 1.0) * frac * h
    dx[fixed] = 0.0
    dy[fixed] = 0.0
    X[:, 0] += dx                              # nut giu nguyen: dx = dy = 0, toa do khong doi
    X[:, 1] += dy
    # Tong kiem tren nut SO HUU: phai trung nhau o moi so nhan.
    try:
        nown = domain.geometry.index_map().size_local
    except Exception:                                      # pragma: no cover
        nown = len(X)
    s = comm.allreduce(np.array([float((~fixed[:nown]).sum()),
                                 float(dx[:nown].sum()), float(dy[:nown].sum()),
                                 float(np.abs(dx[:nown]).sum() + np.abs(dy[:nown]).sum())]),
                       op=MPI.SUM)
    mx = comm.allreduce(float(max(np.abs(dx).max(), np.abs(dy).max())) / h if len(X) else 0.0,
                        op=MPI.MAX)
    if comm.rank == 0:
        print(f"  NHIEU LUOI: so nut dich = {int(s[0])}  max|dich|/h = {mx:.4f}  "
              f"tong dx = {s[1]:.12e}  tong dy = {s[2]:.12e}  tong |dich| = {s[3]:.12e}",
              flush=True)


if comm.rank == 0:
    print(f"  NHIEU LUOI: --jitter {args.jitter:g} --seed {args.seed}"
          + ("   (tat)" if args.jitter <= 0.0 else
             "   *** LUOI BI DICH -- CA NGHIEN CUU NHIEU"), flush=True)
if args.jitter > 0.0:
    _jitter_mesh(args.jitter, args.seed)

V_u = functionspace(domain, ("Lagrange", 1, (2,)))
V_d = functionspace(domain, ("Lagrange", 1))
# Truong lich su H phai HANG TRONG TUNG PHAN TU.  psi_pos(u) chua grad(u), voi u
# bac 1 thi grad(u) gian doan qua bien phan tu; noi suy no vao mot khong gian
# lien tuc thi tai moi nut dung chung "phan tu nao ghi sau thi thang", va thu tu
# o phu thuoc phan hoach -> huong vet nut phu thuoc so nhan.  DG0 khong co bac
# tu do dung chung nen khong con cho cho su tuy tien do.
# --hspace cg1 dung lai V_d lam khong gian cua H, dung nhu ma truoc ban va DG0,
# de do tat ma moi thu khac giu nguyen.
V_H = functionspace(domain, ("DG", 0)) if args.hspace == "dg0" else V_d
if comm.rank == 0:
    print(f"  KHONG GIAN H: --hspace {args.hspace}"
          + ("   *** BAN TRUOC KHI SUA -- CHI DE DO TAT" if args.hspace == "cg1" else ""),
          flush=True)

u = Function(V_u, name="u")
d = Function(V_d, name="damage")
d_old = Function(V_d)
H = Function(V_H, name="history")

# Plane strain is the physical choice for a plate; plane stress is the right
# choice for a slender bar, where it also makes the 1D modulus exactly E0 and
# the predicted peak exactly f_t.
mu = E0 / (2 * (1 + nu))
if getattr(args, "plane", "strain") == "stress":
    lam = E0 * nu / (1.0 - nu ** 2)
else:
    lam = E0 * nu / ((1 + nu) * (1 - 2 * nu))
K2 = lam + mu          # 2D bulk modulus: K = lambda + 2 mu / d with d = 2


def eps(v):
    return ufl.sym(ufl.grad(v))


def sigma0(v):
    return 2.0 * mu * eps(v) + lam * ufl.tr(eps(v)) * ufl.Identity(2)


def psi_pos(v):
    """Amor volumetric-deviatoric split, tension-positive.

    The coefficient on the volumetric term is the BULK modulus K = lambda + mu
    (two dimensions), not lambda.  Writing lambda there, as the first version
    did, silently drops mu * tr(eps)^2 / 2 from the driving energy.  On a
    uniaxial bar in plane strain that leaves psi_pos at 0.7750 of the true
    strain energy, so damage starts late and the peak stress comes out 15.9 %
    above f_t.  The A0 bar measured 4.0176 MPa against f_t = 3.5, and the
    uncorrected prediction for that run is 4.0577 MPa, which is how the error
    was identified.

    With K correct, psi_pos equals the full strain energy whenever tr(eps) > 0,
    which is what makes the identity H_crit = f_t^2 / (2 E) hold.
    """
    e = eps(v)
    tr_p = 0.5 * (ufl.tr(e) + abs(ufl.tr(e)))
    e_dev = e - ufl.tr(e) / 2.0 * ufl.Identity(2)
    if args.split == "spectral":
        # Phan tich pho (Miehe, Hofacker, Welschinger 2010):
        #   psi+ = 0.5 lam <tr eps>+^2 + mu sum_i <eps_i>+^2
        # Tri rieng 2x2 dang dong kin.  +1e-30 trong sqrt de kha vi tai eps = 0.
        _q = ufl.sqrt(((e[0, 0] - e[1, 1]) / 2.0) ** 2 + e[0, 1] ** 2 + 1.0e-30)
        _e1 = ufl.tr(e) / 2.0 + _q
        _e2 = ufl.tr(e) / 2.0 - _q
        _p1 = 0.5 * (_e1 + abs(_e1))
        _p2 = 0.5 * (_e2 + abs(_e2))
        return 0.5 * lam * tr_p ** 2 + mu * (_p1 ** 2 + _p2 ** 2)
    if args.split == "amor":
        return 0.5 * K2 * tr_p ** 2 + mu * ufl.inner(e_dev, e_dev)
    if args.split == "vol":
        # BO so hang lech.  Chi de chan doan duong nut; dinh luc khong dung duoc.
        return 0.5 * K2 * tr_p ** 2
    # full: khong tach gi ca
    return 0.5 * lam * ufl.tr(e) ** 2 + mu * ufl.inner(e, e)


if comm.rank == 0:
    print(f"  PHEP TACH NANG LUONG: --split {args.split}"
          + ("" if args.split == "amor"
             else "   *** BIEN THE CHAN DOAN -- DINH LUC KHONG DUNG DUOC"))


# ------------------------------------------------------- degradation function
if args.model == "AT2":
    c_w = 2.0                                   # normalisation for w(d)=d^2
    def w(dd):
        return dd ** 2
    def g(dd):
        return (1.0 - dd) ** 2 + 1e-7
else:
    # Wu (2017) cohesive: g(d) = (1-d)^2 / [(1-d)^2 + a1 d (1 + a2 d)]
    # with a1 = 4 E Gc / (pi ell ft^2).  a2 = -0.5 gives linear softening.
    a1 = 4.0 * E0 * Gc / (np.pi * ell * ft ** 2)
    a2, a3 = -0.5, 0.0
    c_w = np.pi                                 # for w(d) = 2d - d^2
    if comm.rank == 0:
        print(f"  Wu cohesive: a1 = {a1:.2f}")
    def w(dd):
        return 2.0 * dd - dd ** 2
    def g(dd):
        num = (1.0 - dd) ** 2
        den = num + a1 * dd * (1.0 + a2 * dd + a2 * a3 * dd ** 2)
        return num / (den + 1e-9)

def _g_num(arr):
    """g(d) evaluated on a numpy array -- used only for the staggered
    convergence test, so it must match g() above exactly."""
    a = np.clip(arr, 0.0, 1.0)
    if args.model == "AT2":
        return (1.0 - a) ** 2 + 1e-7
    num = (1.0 - a) ** 2
    den = num + a1 * a * (1.0 + a2 * a + a2 * a3 * a ** 2)
    return num / (den + 1e-9)


# ------------------------------------------------------------- displacement BC
def bottom(x):
    return np.isclose(x[1], 0.0)


def top(x):
    return np.isclose(x[1], 1.0)


fdim = domain.topology.dim - 1
f_bot = mesh.locate_entities_boundary(domain, fdim, bottom)
f_top = mesh.locate_entities_boundary(domain, fdim, top)

V_uy, _ = V_u.sub(1).collapse()
V_ux, _ = V_u.sub(0).collapse()
zero = Function(V_uy)
zero.x.array[:] = 0.0
u_top = Function(V_uy)

zx = Function(V_ux)
zx.x.array[:] = 0.0
if args.load == "shear":
    # SENS: canh duoi ngam ca hai phuong; canh tren u_y = 0, u_x tang dan.
    u_top = Function(V_ux)
    bc_u = [
        dirichletbc(zero, locate_dofs_topological((V_u.sub(1), V_uy), fdim, f_bot), V_u.sub(1)),
        dirichletbc(zx, locate_dofs_topological((V_u.sub(0), V_ux), fdim, f_bot), V_u.sub(0)),
        dirichletbc(zero, locate_dofs_topological((V_u.sub(1), V_uy), fdim, f_top), V_u.sub(1)),
        dirichletbc(u_top, locate_dofs_topological((V_u.sub(0), V_ux), fdim, f_top), V_u.sub(0)),
    ]
else:
    bc_u = [
        dirichletbc(zero, locate_dofs_topological((V_u.sub(1), V_uy), fdim, f_bot),
                    V_u.sub(1)),
        dirichletbc(u_top, locate_dofs_topological((V_u.sub(1), V_uy), fdim, f_top),
                    V_u.sub(1)),
    ]
# Dieu kien ngam theo phuong x.  Xem patch_a_grip.py de biet vi sao co lua chon.
if args.load == "shear":
    pass
elif args.grip == "clamp":
    # u_x = 0 tren ca hai canh ngam: chan truot ngang, giu mau o mode I.
    bc_u.append(dirichletbc(zx, locate_dofs_topological((V_u.sub(0), V_ux), fdim, f_bot),
                            V_u.sub(0)))
    bc_u.append(dirichletbc(zx, locate_dofs_topological((V_u.sub(0), V_ux), fdim, f_top),
                            V_u.sub(0)))
else:
    # Mot nut goc: du de co nghiem duy nhat, nhung hai canh ngam truot ngang tu do.
    corner = mesh.locate_entities_boundary(
        domain, 0, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bc_u.append(dirichletbc(zx, locate_dofs_topological((V_u.sub(0), V_ux), 0, corner),
                            V_u.sub(0)))
if comm.rank == 0 and args.load == "shear":
    print("  TAI: --load shear (SENS)", flush=True)
elif comm.rank == 0:
    print(f"  DIEU KIEN NGAM: --grip {args.grip}"
          + ("  (u_x = 0 tren ca hai canh)" if args.grip == "clamp"
             else "  (mot nut goc -- canh ngam truot ngang tu do)"))

# ---------------------------------------------------------------- the notch
# The notch is imposed as d = 1 on a thin strip from the left edge to mid-span
# at mid-height.  This avoids meshing a slit and is standard practice.
# The notch half-thickness is a FIXED length, not 1.5 ell.  Tying it to ell
# made the specimen geometry change with the very parameter the acceptance
# test varies: halving ell also halved the notch thickness, so a peak that
# moved could not be attributed to the model rather than to the geometry.
_PORES = None


def notch(x):
    if getattr(args, "sharpnotch", False):
        # --n1 la SO BUOC TAI o chang min, khong phai so phan chia luoi.
        # Dung sai phai buoc vao luoi thuc: h = L / nx.
        _h = 1.0 / float(nx)
        _m = (np.abs(x[1] - 0.5) < 0.4 * _h) & (x[0] < 0.5 + 0.4 * _h)
    else:
        _m = ((x[1] > 0.5 - args.notchw) & (x[1] < 0.5 + args.notchw)
              & (x[0] < 0.5))
    if getattr(args, "pores", None):
        global _PORES
        if _PORES is None:
            _z = np.load(args.pores)
            _PORES = (_z["cx"], _z["cy"], _z["r"])
            if comm.rank == 0:
                print("  PORES: %d lo tu %s   do rong = %.2f%%"
                      % (len(_PORES[0]), args.pores, 100.0 * float(_z["por"])),
                      flush=True)
        _cx, _cy, _r = _PORES
        for _k in range(len(_cx)):
            _m = _m | ((x[0] - _cx[_k]) ** 2 + (x[1] - _cy[_k]) ** 2 < _r[_k] ** 2)
    try:
        if comm.rank == 0:
            print("  NOTCH DOFS (nhan 0, cuc bo) = %d   sharp=%s"
                  % (int(_m.sum()), getattr(args, "sharpnotch", False)), flush=True)
    except Exception:
        pass
    return _m
notch_dofs = fem.locate_dofs_geometrical(V_d, notch)

# Reporting masks.
#
# "max damage outside the notch" turned out to be useless: it is dominated by
# the boundary layer of the imposed d = 1 strip, which sits at a fixed value
# and never moves, so it reported a frozen crack while the tip was in fact
# advancing.  The same trap caught the H diagnostic, whose global maximum is
# the notch-tip singularity -- inside the region where d is imposed.
#
# So both are measured strictly AHEAD of the tip instead.
_xc = V_d.tabulate_dof_coordinates()
x_tip = 0.5
ahead = (_xc[:, 0] > x_tip + 3.0 * ell) & (np.abs(_xc[:, 1] - 0.5) < 6.0 * ell)
free_mask = np.ones(d.x.array.size, dtype=bool)
free_mask[notch_dofs] = False
ahead = ahead & free_mask
# Mat na tuong ung tren DG0, theo TAM PHAN TU.  Cua so da doi hoi
# x > 0.5 + 3 ell nen khong chong len vet khia (x < 0.5); khong can free_mask.
_xH = V_H.tabulate_dof_coordinates()
ahead_H = ((_xH[:, 0] > x_tip + 3.0 * ell)
           & (np.abs(_xH[:, 1] - 0.5) < 6.0 * ell))
_xs_ahead = _xc[:, 0]
one = Function(V_d)
one.x.array[:] = 1.0
bc_d = [dirichletbc(one, notch_dofs)]

# ------------------------------------------------------------- the two problems
v = ufl.TestFunction(V_u)
du = ufl.TrialFunction(V_u)
a_u = ufl.inner(g(d) * sigma0(du), eps(v)) * ufl.dx
L_u = ufl.inner(Constant(domain, PETSc.ScalarType((0.0, 0.0))), v) * ufl.dx
prob_u = LinearProblem(a_u, L_u, bcs=bc_u, u=u,
                       petsc_options_prefix="u_",
                       petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                                      "ksp_rtol": 1e-10})

q = ufl.TestFunction(V_d)
Gc_c = Constant(domain, PETSc.ScalarType(Gc))
ell_c = Constant(domain, PETSc.ScalarType(ell))

# Build the ENERGY as a form and differentiate the form.
#
# The previous version called ufl.derivative(g(d), d, q) on a bare expression.
# ufl.derivative is meant for forms; on an expression it does not raise but it
# does not give the Gateaux derivative of the functional either, so the damage
# equation lost its driving term and d never moved.  Symptom: d frozen at its
# notch boundary-layer value for every load step, with the staggered loop
# exiting after 2 iterations because nothing changed.
E_damage = (g(d) * H
            + Gc_c / c_w * (w(d) / ell_c
                            + ell_c * ufl.inner(ufl.grad(d), ufl.grad(d)))) * ufl.dx
F_d = ufl.derivative(E_damage, d, q)
J_d = ufl.derivative(F_d, d, ufl.TrialFunction(V_d))

if _SNES_STYLE:
    prob_d = NonlinearProblem(
        F_d, d, bcs=bc_d, J=J_d, petsc_options_prefix="d_",
        # CG with hypre, which is the configuration that ran without a single
        # SNES failure.  Switching to GMRES with a backtracking line search
        # was my own change, made on the theory that the Wu Jacobian is not
        # symmetric positive definite; it made things worse, not better, so
        # it is reverted.  Failures are still counted and reported below.
        petsc_options={"snes_type": "vinewtonrsls", "snes_rtol": 1e-8,
                       "snes_atol": 1e-10, "snes_max_it": 40,
                       "ksp_type": "cg", "pc_type": "hypre",
                       "ksp_rtol": 1e-10})

    _lb_d = Function(V_d)
    _ub_d = Function(V_d)
    _ub_d.x.array[:] = 1.0

    def solve_damage():
        # Rang buoc d_old <= d <= 1 dat THANG vao bo giai thay vi cat sau.
        # d hien tai bang d_old o thoi diem nay (vong so le vua copy), nen
        # no vua la diem xuat phat kha thi vua la chan duoi.
        _lb_d.x.array[:] = d.x.array
        prob_d.solver.setVariableBounds(_lb_d.x.petsc_vec, _ub_d.x.petsc_vec)
        out = prob_d.solve()
        # A silently diverging solve looks exactly like "nothing happened",
        # which is the failure mode that cost us the last run.  Surface it.
        try:
            reason = prob_d.solver.getConvergedReason()
        except Exception:
            reason = None
        return reason
else:
    prob_d = NonlinearProblem(F_d, d, bcs=bc_d, J=J_d)
    solver_d = NewtonSolver(comm, prob_d)
    solver_d.rtol, solver_d.atol, solver_d.max_it = 1e-8, 1e-10, 40
    solver_d.convergence_criterion = "incremental"

    def solve_damage():
        its, ok = solver_d.solve(d)
        return 1 if ok else -1

# ------------------------------------------------------------- reaction force
# Total vertical reaction on the driven edge, as internal virtual work against
# a virtual field that is 1 in y on that edge and 0 elsewhere.  This avoids
# indexing into the dof array of a blocked sub-space, which is the single most
# error-prone line in a script like this.
_ri, _Vr = (0, V_ux) if args.load == "shear" else (1, V_uy)
one_y = Function(_Vr)
one_y.x.array[:] = 1.0
bc_reac = dirichletbc(one_y,
                      locate_dofs_topological((V_u.sub(_ri), _Vr), fdim, f_top),
                      V_u.sub(_ri))
v_reac = Function(V_u)
v_reac.x.array[:] = 0.0
_set_bc(v_reac.x.array, [bc_reac])
v_reac.x.scatter_forward()
react_form = fem.form(ufl.inner(g(d) * sigma0(u), eps(v_reac)) * ufl.dx)

# ------------------------------------------ crack length by dissipated surface
# The threshold measure "furthest free dof with d > 0.5" is unusable.  The free
# dof adjacent to the imposed d = 1 notch strip sits above 0.5 from the very
# first step, so it reports a = h at step 1 and never moves -- exactly what the
# constant a = 0.0100 in the ell = 0.05 / nx = 100 run was.  The regularised
# crack surface
#       Gamma(d) = (1/c_w) * int ( w(d)/ell + ell |grad d|^2 ) dx
# is the phase field's own measure of crack area, and it is what the model
# actually dissipates.  Gamma at step 0 already contains the notch and its
# boundary layer, so Gamma - Gamma_0 is crack EXTENSION in mm, free of both the
# threshold artefact and the boundary layer.
Gamma_form = fem.form((1.0 / c_w) * (w(d) / ell_c
                      + ell_c * ufl.inner(ufl.grad(d), ufl.grad(d))) * ufl.dx)

# Gamma_0 is taken at the FIRST LOAD STEP, not from a relaxation solve at zero
# load.  The relaxation was tried and it destroyed the run: with H = 0 the
# functional loses convexity, because w(d) = 2d - d^2 is concave, so Newton
# walked to d = 1 over the whole plate.  Irreversibility then locked it there
# and every step reported d_tip = 1.000 at zero load before any loading had
# been applied.  Gamma_0 = 3.183 mm for a 0.5 mm notch was the symptom.
#
# Taking the baseline at step 1 measures crack EXTENSION from the initial
# state, which is what the quantity is for, and needs no extra solve.
# d is left at zero here.  Pre-setting d = 1 on the notch dofs looked
# harmless, since the Dirichlet condition imposes it during the solve anyway,
# but it hands Newton a discontinuous starting field and the line search then
# failed at every step (SNES -6 at steps 1 to 5, -3 at step 6).  Starting from
# d = 0 and letting the boundary condition do the work is the configuration
# that ran clean.
Gamma0 = None

# ---------------------------------------------------------------------- loop
# Damage initiates where the history H reaches this.  For the Wu model it
# should equal ft^2/(2E) exactly; printing both is a one-line check that the
# normalisation a1 and the constant c_w belong to the same model.
if args.model == "WU":
    H_crit = 2.0 * Gc / (np.pi * ell * a1)
else:
    H_crit = Gc / (2.0 * ell)
if comm.rank == 0:
    print(f"  damage threshold H_crit = {H_crit:.4e} ; "
          f"ft^2/(2E) = {ft**2/(2*E0):.4e}")

hist = []
n_bad = 0
start_step = 1

# Diem kiem.  Moi nhan ghi mang CUC BO cua u, d, d_old, H (ca ghost) vao hai
# khe luan phien theo chan le cua buoc, ghi ra tep tam roi doi ten, nen may
# chet giua luc ghi van con mot khe tron ven.  Toa do bac tu do luu kem: chay
# tiep voi so nhan khac hay phan hoach khac thi dung lai, khong chay tiep sai.
import os as _os
_ck_dir = args.ckpt
_ck_funcs = (("u", u), ("d", d), ("d_old", d_old), ("H", H))
def _ck_xyz():
    return np.concatenate([V_u.tabulate_dof_coordinates().ravel(),
                           V_d.tabulate_dof_coordinates().ravel(),
                           V_H.tabulate_dof_coordinates().ravel()])
def _ck_path(slot):
    return _os.path.join(_ck_dir, f"r{comm.rank}of{comm.size}_s{slot}.npz")
def _ck_save(step):
    if not _ck_dir:
        return
    if comm.rank == 0:
        _os.makedirs(_ck_dir, exist_ok=True)
    comm.barrier()
    f = _ck_path(step % 2)
    tmp = f + ".tmp.npz"
    np.savez(tmp, step=step, n_bad=n_bad,
             gamma0=np.nan if Gamma0 is None else Gamma0,
             hist=np.array(hist, dtype=float).reshape(-1, 5), xyz=_ck_xyz(),
             **{k: fn.x.array for k, fn in _ck_funcs})
    _os.replace(tmp, f)
def _ck_remap(z):
    """Hoan vi cuc bo old -> new theo toa do, hoac None neu tap toa do khac."""
    nu, nd, nH = len(u.x.array) // 2, len(d.x.array), len(H.x.array)
    if (len(z["u"]) != 2 * nu or len(z["d"]) != nd or len(z["H"]) != nH
            or z["xyz"].size != 3 * (nu + nd + nH)):
        return None
    xo = z["xyz"].reshape(-1, 3); xn = _ck_xyz().reshape(-1, 3)
    out = {}
    for ten, a, b in (("u", 0, nu), ("d", nu, nu + nd), ("H", nu + nd, nu + nd + nH)):
        ko = np.round(xo[a:b], 10); kn = np.round(xn[a:b], 10)
        io = np.lexsort(ko.T[::-1]); i_n = np.lexsort(kn.T[::-1])
        if not np.array_equal(ko[io], kn[i_n]) or (len(io) > 1 and
                np.any(np.all(ko[io][1:] == ko[io][:-1], axis=1))):
            return None
        pm = np.empty(len(i_n), dtype=np.int64); pm[i_n] = io
        out[ten] = pm
    return out
if args.resume:
    if not _ck_dir:
        raise SystemExit("--resume can --ckpt")
    _have = {}
    for slot in (0, 1):
        try:
            with np.load(_ck_path(slot)) as z:
                _have[int(z["step"])] = slot
        except (OSError, KeyError, ValueError):
            pass
    _steps = comm.allgather(sorted(_have))
    _common = set(_steps[0]).intersection(*map(set, _steps[1:])) if _steps else set()
    if not _common:
        raise SystemExit("KHONG CO DIEM KIEM CHUNG cho moi nhan -- chay lai tu dau")
    _s = max(_common)
    with np.load(_ck_path(_have[_s])) as z:
        _perm = None
        if z["xyz"].shape != _ck_xyz().shape or not np.array_equal(z["xyz"], _ck_xyz()):
            # Cung phan hoach nhung THU TU bac tu do cuc bo (nhat la ghost) co the
            # khac giua hai may.  Neu tap toa do trung khop thi anh xa lai theo toa
            # do; neu khac tap thi la phan hoach khac that -- dung.
            _perm = _ck_remap(z)
        # Cac phep allreduce phai goi tren MOI nhan (goi rieng nhan 0 la treo).
        _ok = comm.allreduce(0 if (_perm is None and not np.array_equal(z["xyz"], _ck_xyz()))
                             else 1, op=MPI.MIN)
        if not _ok:
            raise SystemExit("PHAN HOACH KHAC diem kiem -- khong chay tiep")
        _co_anh_xa = comm.allreduce(1 if _perm is not None else 0, op=MPI.MAX)
        if comm.rank == 0 and _co_anh_xa:
            print("  RESUME: thu tu bac tu do khac, anh xa lai theo toa do", flush=True)
        for k, fn in _ck_funcs:
            a = z[k]
            if _perm is not None:
                a = (a.reshape(-1, 2)[_perm["u"]].ravel() if k == "u"
                     else a[_perm["H" if k == "H" else "d"]])
            fn.x.array[:] = a
            fn.x.scatter_forward()
        n_bad = int(z["n_bad"])
        Gamma0 = None if np.isnan(float(z["gamma0"])) else float(z["gamma0"])
        hist = [tuple(r) for r in z["hist"]]
    start_step = _s + 1
    if comm.rank == 0:
        print(f"  RESUME: chay tiep tu buoc {start_step} (diem kiem buoc {_s})", flush=True)
V0 = functionspace(domain, ("DG", 0))
# Diem lay psi+ la diem noi suy cua V_H tren phan tu tham chieu, nen tu dung
# cho ca hai loai o: DG0 -> TAM phan tu (tu giac (1/2, 1/2), tam giac (1/3,
# 1/3)); P1/Q1 -> cac DINH (4 hay 3).  Tren tam giac P1, grad u hang trong
# phan tu nen psi+ hang tung phan tu: DG0 lay dung gia tri do, con P1 dua gia
# tri cua MOT phan tu vao moi dinh dung chung (phan tu ghi sau thang) -- cung
# lop tat noi suy nut nhu Q1.  Cap nhat H = max(H, psi+) la theo mang bac tu
# do, khong phu thuoc loai o.
_ip = V_H.element.interpolation_points
if callable(_ip):
    _ip = _ip()
psi_expr = fem.Expression(psi_pos(u), _ip)
psi_fn = Function(V_H)

for step in range(start_step, args.steps + 1):
    ubar = float(_ramp[step - 1])
    u_top.x.array[:] = ubar

    for k in range(200):                                   # staggered loop
        prob_u.solve()
        psi_fn.interpolate(psi_expr)
        psi_fn.x.scatter_forward()
        H.x.array[:] = np.maximum(H.x.array, psi_fn.x.array)
        H.x.scatter_forward()
        d_old.x.array[:] = d.x.array
        reason = solve_damage()
        d.x.scatter_forward()
        # Irreversibility AND the physical bound 0 <= d <= 1.  Only the
        # lower bound was applied before, so d drifted to 1.372 on the
        # ell = 0.1 run: past d = 1 the Wu degradation function and its
        # Jacobian lose meaning, SNES then returned -3 or -6 at every step,
        # and the staggered loop exited after one iteration because nothing
        # could change.  Every number printed after that point was noise.
        d.x.array[:] = np.clip(np.maximum(d.x.array, d_old.x.array), 0.0, 1.0)
        d.x.scatter_forward()
        # Converge on the DEGRADATION, not on d.  With a1 ~ 2000 the Wu model
        # loses 86 % of its stiffness by d = 0.007, so a tolerance of 1e-4 on
        # |dd| lets the staggered loop exit while the mechanics is still
        # changing completely.  g(d) is what the displacement problem sees.
        gn = _g_num(d.x.array)
        go = _g_num(d_old.x.array)
        dd = float(np.max(np.abs(gn - go))) if gn.size else 0.0
        dd = comm.allreduce(dd, op=MPI.MAX)
        if dd < 1e-6:
            break

    if step in (1, 40):
        # Chi cong cac bac tu do THUOC SO HUU: mang cuc bo con chua o ghost,
        # cong ca thi dem trung.
        _n = V_H.dofmap.index_map.size_local * V_H.dofmap.index_map_bs
        _seg = H.x.array[:_n]
        _hsum = comm.allreduce(float(_seg.sum()), op=MPI.SUM)
        _hmax = comm.allreduce(float(_seg.max()) if _n else 0.0, op=MPI.MAX)
        _hl2 = comm.allreduce(float((_seg ** 2).sum()), op=MPI.SUM) ** 0.5
        if comm.rank == 0:
            print(f"  H-INVARIANT step{step}: sum={_hsum:.12e} "
                  f"max={_hmax:.12e} L2={_hl2:.12e}", flush=True)

    if reason is not None and reason <= 0:
        n_bad += 1
    F = comm.allreduce(fem.assemble_scalar(react_form), op=MPI.SUM)
    # damage outside the notch: this measures CRACK GROWTH.  Reporting
    # d.x.array.max() would report the notch, which is 1 by construction.
    dah = d.x.array[ahead]
    dmax = comm.allreduce(float(dah.max()) if dah.size else 0.0, op=MPI.MAX)
    Hah = H.x.array[ahead_H]
    Hmax = comm.allreduce(float(Hah.max()) if Hah.size else 0.0, op=MPI.MAX)
    # crack extension, measured as dissipated surface minus the initial notch
    Gamma = comm.allreduce(fem.assemble_scalar(Gamma_form), op=MPI.SUM)
    if Gamma0 is None:
        Gamma0 = Gamma
    a_len = Gamma - Gamma0
    hist.append((ubar, F, dmax, a_len, k + 1))
    if comm.rank == 0:
        print(f"  step {step:3d}  u = {ubar:.3e}  F = {F:8.4f}  "
              f"d_tip = {dmax:.3f}  a = {a_len:.4f}  "
              f"H/Hc = {Hmax/H_crit:8.2f}  stagg = {k+1}"
              + ("" if reason is None or reason > 0 else f"  SNES={reason}"))
    _ck_save(step)
    Fs = np.abs(np.array([r[1] for r in hist]))
    if len(Fs) > 6 and Fs.max() > 0 and Fs[-1] < 0.25 * Fs.max():
        if comm.rank == 0:
            print("  load has fallen below 25 % of peak -- stopping")
        break
    if n_bad > 20:
        if comm.rank == 0:
            print(f"  {n_bad} failed damage solves -- stopping, the remaining")
            print("  steps would only add noise")
        break

if comm.rank == 0:
    h = np.array(hist)
    Fabs = np.abs(h[:, 1])
    peak = Fabs.max()
    i_peak = int(np.argmax(Fabs))
    rose = i_peak > 0
    fell = i_peak < len(Fabs) - 1
    tag = args.out or f"{args.model}_ell{ell:g}"
    np.savetxt(f"loaddisp_{tag}.txt", h,
               header="u_mm  F_N_per_mm  dmax  crack_extension_mm  staggered_its")
    print("\n" + "=" * 60)
    print(f"  model {args.model}   ell = {ell} mm")
    print(f"  PEAK LOAD = {peak:.4f} N/mm   at step {i_peak+1} of {len(Fabs)}")
    print(f"  nominal peak stress = {peak / L:.4f} MPa   (f_t = {ft} MPa)")
    print(f"  expected for a half-ligament notch: {0.5*ft:.3f} MPa")
    # A peak is only a peak if the specimen actually broke.  The first
    # campaign printed "peak is real" for curves that ended at 97 % of their
    # maximum with d_tip below 0.10, i.e. for a plateau.  Two hard gates now:
    # the load must fall to below 60 % of the maximum, and damage must have
    # grown past 0.9 somewhere ahead of the notch.
    tail = Fabs[-1] / peak if peak > 0 else 1.0
    dfin = h[-1, 2]
    afin = h[-1, 3]
    print(f"  final load = {100*tail:.1f} % of peak ; final d ahead of the "
          f"tip = {dfin:.3f} ; crack extension = {afin:.4f} mm "
          f"({afin/ell:.1f} ell)")
    # The gate asks whether a crack ran, and it accepts either witness.
    # Requiring d > 0.9 alone was self-contradictory: the run stops when the
    # load falls below 25 % of peak, which happens while d at the measuring
    # points is still near 0.8, so a perfectly good fracture run was being
    # failed by its own stopping rule.
    cracked = dfin > 0.5 or afin > 2.0 * ell
    if not rose:
        print("  *** NO RISING BRANCH: the first load step is already past the")
        print("      peak.  Reduce --u1.  The value above is NOT the peak.")
    elif tail > 0.6 or not cracked:
        print("  *** NOT A FRACTURE PEAK: the load never fell below 60 % of")
        print("      its maximum, or no crack ran (d stayed under 0.5 and the")
        print("      crack grew less than 2 ell).  The maximum above is a")
        print("      plateau value, which is noise.")
        if h[-1, 0] > 0.95 * args.umax:
            print(f"      Raise --umax above {args.umax:.4f} mm (w_c = {w_c:.5f}).")
        else:
            print(f"      NOTE: the run stopped at u = {h[-1, 0]:.3e} mm of {args.umax:.4f} mm,")
            print("      so it ended early -- look at the failed damage solves below,")
            print("      not at umax.")
    else:
        print("  fracture peak confirmed: load fell past 60 % and a crack ran")
    if n_bad:
        print(f"  *** {n_bad} of {len(Fabs)} steps had a failed damage solve")
    print(f"  wrote loaddisp_{tag}.txt")
    print("=" * 60)

with io.XDMFFile(comm, f"damage_{args.model}_ell{ell:g}.xdmf", "w") as xf:
    xf.write_mesh(domain)
    xf.write_function(d)
