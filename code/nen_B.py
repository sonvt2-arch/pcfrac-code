"""Vong 11-B: gom tep thua B_<tag>_dthua.bin (chuoi np.save) thanh B_<tag>_dthua.npz (nen).
Dung: python3 nen_B.py THU_MUC_B"""
import glob
import os
import sys
import numpy as np

for p in glob.glob(os.path.join(sys.argv[1], "*_dthua.bin")):
    parts = []
    with open(p, "rb") as f:
        while True:
            try:
                parts.append(np.load(f))
            except (EOFError, ValueError, OSError):
                break
    if parts:
        np.savez_compressed(p[:-4] + ".npz", rows=np.concatenate(parts))
        print(f"{p}: {len(parts)} khoi, {sum(len(a) for a in parts)} dong")
