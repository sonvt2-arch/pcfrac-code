#!/usr/bin/env bash
# Vong 11-A bo sung 20/11: khe hinh hoc nx800 tren may tram (DG0, cg1nmax, cg1nmean), ghi truoc o
# Pervious-Concrete doc/CA-VONG11-A-2026-11-17.md muc "Bo sung 20/11 (may tram, nx800)".
# Can: Docker, git (quyen push repo pcfrac-code). Chay tu goc repo:  bash may-tram/chay_nx800_khe.sh
# Bien: NP (so nhan MOI ca, mac dinh 4; ca 3 ca dung CUNG NP), CA (mac dinh ca 3 ca).
# Bi ngat (mat dien...) thi chay lai dung lenh nay: tu chay tiep tu diem kiem.
set -u
NP=${NP:-4}
CA=${CA:-"dg0 cg1nmax cg1nmean"}
IMG=dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8
ARGS="--model AT2 --ell 0.01 --grip clamp --split amor --sharpnotch --E 210000 --nu 0.3 --Gc 2.7 --n1 100 --u1 0.001 --load tension --steps 800 --umax 0.008 --nx 800 --notch geom --maxbad 100000"
GOC=$(cd "$(dirname "$0")/.." && pwd)
OUT=$GOC/ket-qua/2026-11-20-vong11-A-nx800; mkdir -p $OUT/truong
docker pull $IMG
for h in $CA; do
  tag=KM_${h}_geo_nx800_n$NP; W=$GOC/may-tram/lam_$tag
  mkdir -p $W; cp -n $GOC/code/*.py $W/ 2>/dev/null
  ( docker run --rm -v $W:/w -w /w --shm-size=2g $IMG bash -c "
      rs=''; ls ck_$tag/r0of${NP}_s*.npz >/dev/null 2>&1 && rs='--resume'
      echo \"=== $(date -u +%FT%TZ) \${rs:-tu dau} (may tram) \$(nproc) nhan; \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2); \$(free -g | awk '/Mem/{print \$2}') GB\" >> A_$tag.log
      mpirun --allow-run-as-root -n $NP python3 -u benchmark_A.py $ARGS --hspace $h --ckpt ck_$tag \$rs --out $tag >> A_$tag.log 2>&1
      echo \"=== EXIT=\$? \$(date -u +%FT%TZ)\" >> A_$tag.log
      if grep -q 'PEAK LOAD' A_$tag.log && [ -s damage_AT2_ell0.01.h5 ]; then
        pip install -q h5py 2>/dev/null; cp -f damage_AT2_ell0.01.h5 $tag.h5; cp -f damage_AT2_ell0.01.xdmf $tag.xdmf
        sed -i 's/damage_AT2_ell0\.01\.h5/$tag.h5/g' $tag.xdmf
        python3 measure_kink.py $tag.xdmf --ell 0.01 --dmin 0.2 --x0 0.60 --ve > goc_$tag.txt 2>&1
      fi"
    cp -f $W/A_$tag.log $W/loaddisp_*$tag*.txt $W/goc_$tag.txt $OUT/ 2>/dev/null
    cp -f $W/$tag.h5 $W/$tag.xdmf $OUT/truong/ 2>/dev/null
    echo "xong $tag" ) &
done
wait
cd $GOC && git add ket-qua/2026-11-20-vong11-A-nx800 && git commit -m "Vong 11-A nx800 khe hinh hoc (may tram, NP=$NP)" && git pull --rebase && git push
echo "Da day ket qua len GitHub. Bao Claude: 'nx800 xong'."
