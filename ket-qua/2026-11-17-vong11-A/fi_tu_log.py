import sys,re
# F_init (quy tac vong 10) tu cac dong "step" cua log (dung khi loaddisp chua ghi)
for f in sys.argv[1:]:
    st={}
    for l in open(f):
        m=re.match(r"\s+step\s+(\d+)\s+u = (\S+)\s+F =\s+(\S+)",l)
        if m: st[int(m[1])]=(float(m[2]),abs(float(m[3])))
    out=[]
    for dl in (0.10,0.05,0.20):
        M=-1;uM=None;r=None
        for k in sorted(st):
            u,F=st[k]
            if F>M: M,uM=F,u
            elif F<(1-dl)*M: r=(M,uM);break
        out.append(r)
    Fm=max(F for u,F in st.values()) if st else None
    print(f.split('/')[-1], "buoc cuoi",max(st) if st else None, "F_init(.10,.05,.20)=",out,"Fmax_tam=",Fm)
