# PC-Frac: partition dependence of the phase-field history field

Code, workflows and numerical logs accompanying a manuscript in preparation.

- `code/benchmark_A.py` — phase-field (PF-CZM, Wu) single-edge-notched specimens in
  DOLFINx 0.11; `--hspace cg1|dg0` selects where the history field H is stored,
  `--load tension|shear` selects SENT or SENS, `--ckpt/--resume` checkpoints.
- `code/measure_kink.py`, `code/goc_tu_diem_kiem.py` — crack-path angle measurement
  (from an `.xdmf` field or from a checkpoint).
- `code/phep_thu_R.py` — the partition test R = Δ(partition) / Δ(physics).
- `.github/workflows/kiem-bat-bien-phan-hoach.yml` — runs one case on 1, 2 and 4 MPI ranks
  and fails if the step-1 history-field sum differs by more than 1e-9.
- `.github/workflows/chay-noi.yml` + `.github/chay-noi.json` — long runs chained across
  6-hour jobs through checkpoints; logs are committed to `ket-qua/`.
- `ket-qua/` — logs, load–displacement curves and angle measurements.

Comments and log messages are in Vietnamese. Container: `dolfinx/dolfinx:stable`.

## License

MIT — see `LICENSE`.
