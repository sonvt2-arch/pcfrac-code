"""
vong11B.py -- chan doan vong 11-B (17/11/2026), goi tu benchmark_A.py / benchmark_P3d.py
khi co --do-B DIR.  Khong co --do-B thi KHONG duoc nhap, ket qua cu giu nguyen tung bit.

Quy tac do ghi truoc: doc/CA-VONG11-B-2026-11-17.md (repo du an).

1. Ban do "nguoi ghi sau" (last writer) cua phep noi suy nut CG1: duyet cac o CUC BO
   (so huu + ghost) theo thu tu chi so (hoac nguoc lai voi --ghi-nguoc), nut dung chung
   nhan gia tri cua o duyet SAU CUNG.  Kiem o buoc 1 rang cach ghi tay nay tai lap dung
   psi_fn.interpolate(psi_expr) cua DOLFINx (sai lech tuong doi in ra log).
   Vung R: tam o x in [0.5, 0.5 + 4 ell], |y - 0.5| <= 4 ell (moi z).
   Voi moi nut (V_d, so huu) trong R, KHONG nam tren mat y = 0.5: nguoi ghi la o "phia
   gan" neu |y_c - 0.5| < |y_nut - 0.5|.  f_gan(T), f_gan(B): ti le nut tren / duoi
   co nguoi ghi phia gan.  W = f_gan(B) - f_gan(T).  Nut tren mat: ti le nguoi ghi o tren.
2. Moi buoc: A_X = (int_T X - int_B X) / (int_T X + int_B X) tren R (T: tam o y > 0.5,
   B: y < 0.5) cho X = H (khong gian luu), psiS = psi+ da luu vao khong gian H (psi_fn,
   vong so le cuoi), psiU = psi0+(u) dung (cau phuong bac 4).
3. Moi buoc: dinh nut theo do dai doc duong nut (Dijkstra tren luoi nut deu, lang gieng
   {-1,0,1}^dim) trong tap S = {d >= 0.95, x >= 0.5 - 0.4h}, nguon = nut dinh ranh
   (x = 0.5, y = 0.5, moi z).  a_tip = khoang cach lon nhat; dinh = nut dat no.
   Ghi them cac nut d >= 0.5, x >= 0.5 - 3 ell (chi so luoi + d) de do lai ngoai tuyen.
4. Dung: a_tip >= dung_tip * ell VA da co lan sut >= 10 % so voi cuc dai chay cua |F|.
"""
import os
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dinh_nut_B import dinh_nut as _dinh_nut


class ChanDoanB:
    def __init__(self, comm, domain, V_d, V_H, H, d, u, psi_fn, psi_pos, hspace, ell, h,
                 outdir, tag, qdeg, ghi_nguoc=False, dung_tip=12.0, dung_buoc=0, he_so_F=1.0,
                 psi_expr=None):
        self.comm, self.domain, self.V_d, self.H, self.d, self.psi_fn = comm, domain, V_d, H, d, psi_fn
        self.ell, self.h, self.hspace, self.dung_tip = ell, h, hspace, dung_tip
        self.dim = domain.topology.dim
        self.psi_expr = psi_expr
        self.ghi_nguoc = ghi_nguoc
        self.dung_buoc = dung_buoc
        self.out = outdir
        if comm.rank == 0:
            os.makedirs(outdir, exist_ok=True)
        comm.barrier()
        self.f_buoc = os.path.join(outdir, f"B_{tag}_buoc.txt")
        self.f_thua = os.path.join(outdir, f"B_{tag}_dthua.bin")
        tdim = self.dim
        tm = domain.topology.index_map(tdim)
        self.nc = tm.size_local + tm.num_ghosts
        self.cells = np.arange(self.nc, dtype=np.int32)
        self.mid = mesh.compute_midpoints(domain, tdim, self.cells)
        self.dl = np.asarray(V_d.dofmap.list)[:self.nc]
        self.nv = self.dl.shape[1]
        im = V_d.dofmap.index_map
        self.nown = im.size_local
        self.X = V_d.tabulate_dof_coordinates()
        # --- vung R va chi thi T/B (DG0, theo tam o)
        x0, x1, b = 0.5, 0.5 + 4.0 * ell, 4.0 * ell
        V0 = fem.functionspace(domain, ("DG", 0))
        inR = ((self.mid[:, 0] >= x0 - 1e-12) & (self.mid[:, 0] <= x1 + 1e-12)
               & (np.abs(self.mid[:, 1] - 0.5) <= b + 1e-12))
        c0 = np.asarray(V0.dofmap.list)[:self.nc, 0]
        self.chiT, self.chiB = fem.Function(V0), fem.Function(V0)
        self.chiT.x.array[:] = 0.0
        self.chiB.x.array[:] = 0.0
        self.chiT.x.array[c0[inR & (self.mid[:, 1] > 0.5)]] = 1.0
        self.chiB.x.array[c0[inR & (self.mid[:, 1] < 0.5)]] = 1.0
        dxH = ufl.dx(metadata={"quadrature_degree": qdeg if hspace == "quad" else 2})
        dx4 = ufl.dx(metadata={"quadrature_degree": 4})
        self.forms = {
            "H": (fem.form(H * self.chiT * dxH), fem.form(H * self.chiB * dxH)),
            "psiS": (fem.form(psi_fn * self.chiT * dxH), fem.form(psi_fn * self.chiB * dxH)),
            "psiU": (fem.form(psi_pos(u) * self.chiT * dx4), fem.form(psi_pos(u) * self.chiB * dx4)),
        }
        # --- nut cho dinh nut
        self.ijk = np.rint(self.X[:self.nown, :self.dim] / h).astype(np.int64)
        self.Fs = []
        self.da_sut = False
        self.ban_do_nguoi_ghi()
        if comm.rank == 0:
            with open(self.f_buoc, "a") as f:
                f.write("# buoc u F A_H A_psiS A_psiU intH_T intH_B a_tip/ell x_tip y_tip n_S\n")

    # ------------------------------------------------------------ nguoi ghi sau
    def _thu_tu(self, nguoc):
        return self.cells[::-1] if nguoc else self.cells

    def _vi_tri_cuoi(self, nguoc):
        order = self._thu_tu(nguoc)
        idx = self.dl[order].ravel()
        rev = idx[::-1]
        uq, pos = np.unique(rev, return_index=True)
        last = len(idx) - 1 - pos            # vi tri trong idx (thu tu duyet) cua lan ghi cuoi
        return order, uq, last

    def ghi(self, vals, nguoc):
        """Gia tri nut theo 'nguoi ghi sau' tren thu tu o (xuoi/nguoc).  vals: (nc, nv)."""
        order, uq, last = self._vi_tri_cuoi(nguoc)
        a = np.zeros(len(self.psi_fn.x.array))
        a[uq] = vals[order].ravel()[last]
        return a

    def ban_do_nguoi_ghi(self):
        comm = self.comm
        for nguoc in (False, True):
            order, uq, last = self._vi_tri_cuoi(nguoc)
            winner = np.full(len(self.X), -1, dtype=np.int64)
            winner[uq] = np.repeat(order, self.nv)[last]
            Xo = self.X[:self.nown]
            wo = winner[:self.nown]
            yc = self.mid[wo, 1]
            xc = self.mid[wo, 0]
            inR = ((Xo[:, 0] >= 0.5 - 1e-12) & (Xo[:, 0] <= 0.5 + 4 * self.ell + 1e-12)
                   & (np.abs(Xo[:, 1] - 0.5) <= 4 * self.ell + 1e-12))
            tren = inR & (Xo[:, 1] > 0.5 + 1e-9)
            duoi = inR & (Xo[:, 1] < 0.5 - 1e-9)
            mat = inR & (np.abs(Xo[:, 1] - 0.5) <= 1e-9)
            gan = np.abs(yc - 0.5) < np.abs(Xo[:, 1] - 0.5)
            s = comm.allreduce(np.array([tren.sum(), (tren & gan).sum(), duoi.sum(), (duoi & gan).sum(),
                                         mat.sum(), (mat & (yc > 0.5)).sum(),
                                         inR.sum(), (inR & (xc > Xo[:, 0])).sum()], dtype=float), op=MPI.SUM)
            if comm.rank == 0:
                fT, fB = s[1] / max(s[0], 1), s[3] / max(s[2], 1)
                msg = (f"  B NGUOI GHI ({'NGUOC' if nguoc else 'XUOI'}{' <- dung' if nguoc == self.ghi_nguoc else ''}): "
                       f"nut tren n={int(s[0])} f_gan(T)={fT:.3f}; nut duoi n={int(s[2])} f_gan(B)={fB:.3f}; "
                       f"W=f_gan(B)-f_gan(T)={fB - fT:+.3f}; nut tren mat n={int(s[4])} ti le nguoi ghi o tren="
                       f"{s[5] / max(s[4], 1):.3f}; ti le nguoi ghi phia x lon={s[7] / max(s[6], 1):.3f}")
                print(msg, flush=True)
                with open(os.path.join(self.out, "nguoi_ghi.txt"), "a") as f:
                    f.write(msg + "\n")

    def kiem_noi_suy(self):
        """So psi_fn (do DOLFINx interpolate) voi cach ghi tay xuoi, nut so huu."""
        if self.psi_expr is None:
            return
        vals = np.asarray(self.psi_expr.eval(self.domain, self.cells)).reshape(self.nc, -1)
        if vals.shape != self.dl.shape:
            return
        for nguoc in (False, True):
            a = self.ghi(vals, nguoc)[:self.nown]
            b = self.psi_fn.x.array[:self.nown]
            e = self.comm.allreduce(float(np.max(np.abs(a - b))) if self.nown else 0.0, op=MPI.MAX)
            m = self.comm.allreduce(float(np.max(np.abs(b))) if self.nown else 0.0, op=MPI.MAX)
            if self.comm.rank == 0:
                print(f"  B KIEM NOI SUY buoc 1: |ghi tay {'NGUOC' if nguoc else 'XUOI'} - psi_fn|max / max psi = "
                      f"{e / max(m, 1e-300):.3e}", flush=True)

    # ------------------------------------------------------------ moi buoc
    def _A(self, k):
        fT, fB = self.forms[k]
        T = self.comm.allreduce(fem.assemble_scalar(fT), op=MPI.SUM)
        B = self.comm.allreduce(fem.assemble_scalar(fB), op=MPI.SUM)
        return (T - B) / (T + B) if (T + B) != 0 else 0.0, T, B

    def buoc(self, step, ubar, F):
        comm = self.comm
        if step == 1 and self.hspace == "cg1":
            self.kiem_noi_suy()
        AH, T, B = self._A("H")
        AS = self._A("psiS")[0]
        AU = self._A("psiU")[0]
        dv = self.d.x.array[:self.nown]
        Xo = self.X[:self.nown]
        m5 = (dv >= 0.5) & (Xo[:, 0] >= 0.5 - 3 * self.ell)
        rows = np.concatenate([self.ijk[m5], np.rint(dv[m5] * 1e4)[:, None].astype(np.int64)], axis=1)
        allr = comm.gather(rows, root=0)
        stop = False
        if comm.rank == 0:
            R = np.concatenate(allr).astype(np.int32)
            dd = R[:, -1] / 1e4
            ij = R[:, :self.dim]
            hh = self.h
            S = (dd >= 0.95) & (ij[:, 0] * hh >= 0.5 - 0.4 * hh)
            ijS = ij[S]
            nguon = (ijS[:, 0] * hh > 0.5 - 0.4 * hh) & (ijS[:, 0] * hh < 0.5 + 0.4 * hh) \
                & (np.abs(ijS[:, 1] * hh - 0.5) < 0.4 * hh)
            a, n = _dinh_nut(ijS, nguon, hh, self.dim)
            xt, yt = (ijS[n, 0] * hh, ijS[n, 1] * hh) if n >= 0 else (np.nan, np.nan)
            with open(self.f_thua, "ab") as f:
                np.save(f, np.concatenate([np.full((len(R), 1), step, np.int32), R], axis=1))
            with open(self.f_buoc, "a") as f:
                f.write(f"{step} {ubar:.6e} {F:.6f} {AH:+.6e} {AS:+.6e} {AU:+.6e} {T:.6e} {B:.6e} "
                        f"{a / self.ell:.4f} {xt:.5f} {yt:.5f} {int(S.sum())}\n")
            print(f"  B buoc {step}: A_H={AH:+.3e} A_psiS={AS:+.3e} A_psiU={AU:+.3e} "
                  f"a_tip={a / self.ell:.2f} ell tip=({xt:.4f},{yt:.4f})", flush=True)
            Fa = abs(F)
            if self.Fs and Fa < 0.9 * max(self.Fs):
                self.da_sut = True
            self.Fs.append(Fa)
            stop = (self.da_sut and a >= self.dung_tip * self.ell) or (0 < self.dung_buoc <= step)
            if stop:
                print(f"  B DUNG: a_tip >= {self.dung_tip:g} ell va da sut >= 10 % -- dung chay", flush=True)
        return comm.bcast(stop, root=0)

    def khoi_phuc(self, hist):
        """Sau --resume: dung lai cuc dai chay va co 'da sut' tu hist (u, F, ...)."""
        for r in hist:
            Fa = abs(r[1])
            if self.Fs and Fa < 0.9 * max(self.Fs):
                self.da_sut = True
            self.Fs.append(Fa)
