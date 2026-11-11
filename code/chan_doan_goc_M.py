import sys, numpy as np
sys.path.insert(0, sys.argv[1]); import do_sens_hang as m
for nh, f in zip(("Q1","DG0"), sys.argv[2:4]):
    xy, d, _ = m.doc(f); hm = m.kich_thuoc_luoi(xy); t = m.tam_theo_hang(xy, d, 0.01, hm)
    st = np.hypot(*np.diff(t, axis=0).T); k = np.where(st > 0.02)[0]
    print(nh, "buoc nhay > 2ell:", [(round(t[i,0],3), round(t[i,1],3), round(t[i+1,0],3), round(t[i+1,1],3)) for i in k])
    g, _ = m.goc_cung(t, 0.06); print("  6ell cac cua so:", [round(v,1) for v in g])
    tc = t[:k[0]+1] if len(k) else t
    for L in (2,4,6):
        g,_ = m.goc_cung(tc, L*0.01); print(f"  cat tai buoc nhay dau: {L}ell = {np.mean(g):.2f} ({len(g)})" if g else f"  {L}ell --")
    print("  diem cuoi sau cat:", tc[-1])
print("--- quy uoc huong di xuong (dy<0, goc trong (-180,0)) -- CHAN DOAN SAU KHI THAY SO, khong vao phan quyet")
for nh, f in zip(("Q1","DG0"), sys.argv[2:4]):
    xy, d, _ = m.doc(f); t = m.tam_theo_hang(xy, d, 0.01, m.kich_thuoc_luoi(xy))
    for L in (2,4,6):
        g,_ = m.goc_cung(t, L*0.01); g2 = [v-180 if v > 0 else v for v in g]
        print(f"  {nh} {L}ell: {np.mean(g2):.2f} ({len(g2)} cua so; so cua so goc>0 theo quy uoc ghi truoc: {sum(v>0 for v in g)})")
