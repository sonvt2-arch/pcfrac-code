"""Vong 11-B: dinh nut theo do dai doc duong nut (Dijkstra tren luoi nut deu).
Dung chung cho vong11B.py (trong luc chay) va do_tip_B.py (ngoai tuyen); khong can dolfinx."""
import heapq
import itertools
import numpy as np


def dinh_nut(ijk, nguon, h, dim):
    """Dijkstra tren tap nut (chi so luoi nguyen).  Tra (a_max, chi so nut dat a_max)."""
    if len(ijk) == 0 or not nguon.any():
        return 0.0, -1
    key = {tuple(r): n for n, r in enumerate(ijk)}
    dist = np.full(len(ijk), np.inf)
    pq = []
    for n in np.nonzero(nguon)[0]:
        dist[n] = 0.0
        pq.append((0.0, int(n)))
    heapq.heapify(pq)
    offs = [o for o in itertools.product((-1, 0, 1), repeat=dim) if any(o)]
    lens = [h * float(np.sqrt(sum(c * c for c in o))) for o in offs]
    while pq:
        dv, n = heapq.heappop(pq)
        if dv > dist[n]:
            continue
        r = ijk[n]
        for o, ln in zip(offs, lens):
            m = key.get(tuple(r[c] + o[c] for c in range(dim)))
            if m is not None and dv + ln < dist[m]:
                dist[m] = dv + ln
                heapq.heappush(pq, (dist[m], m))
    ok = np.isfinite(dist)
    if not ok.any():
        return 0.0, -1
    n = int(np.argmax(np.where(ok, dist, -1.0)))
    return float(dist[n]), n
