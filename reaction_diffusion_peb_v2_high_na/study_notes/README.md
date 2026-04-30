# Study Notes — High-NA EUV PEB v2

Per-phase learning records. A note is added at the end of each phase and merged into main.

Each note captures:

- Goal: what question this phase tried to answer
- Steps taken: what was actually done
- Problems encountered: what got stuck, why
- Resolutions: how it was unblocked, why that approach was chosen
- Decision log: at each fork in the road, which branch was taken and why
- Verified results: numbers left behind
- Follow-up work: homework to be picked up by the next phase

## Index

- [01_stage1_clean_geometry.md](01_stage1_clean_geometry.md) — Stage 1 + Stage 1A: clean-geometry baseline and σ-compatible budget calibration
- [02_stage2_dh_time_sweep.md](02_stage2_dh_time_sweep.md) — Stage 2: DH × time 25-grid sweep and best-operating-point selection
- [03_stage3_electron_blur.md](03_stage3_electron_blur.md) — Stage 3: σ isolation + 3-stage LER measurement convention + plan §Stage 3B added
- [04_stage4_weak_quencher.md](04_stage4_weak_quencher.md) — Stage 4: weak-quencher 52-run sweep, σ=3 LER recovery verified, PSD band metric introduced, Stage 4B (CD-locked LER) deferred
- [05_stage5_pitch_dose.md](05_stage5_pitch_dose.md) — Stage 5: pitch × dose 36+72 runs, process-window shape, pitch=16 closed finding, quencher's small-pitch trade-off
- [06_stage4B_cd_locked.md](06_stage4B_cd_locked.md) — Stage 4B: CD-locked LER + Stage 5B pitch-dependent quencher mini-sweep, 1 displacement-artefact case identified, real LER degradation diagnosed at pitch ≤ 20, σ=2 + quencher OP applies only at pitch ≥ 24
- [07_stage6_xz_standing_wave.md](07_stage6_xz_standing_wave.md) — Stage 6: x-z PEB solver (Neumann-z mirror FFT), 3 thickness × 4 amplitude sweep, PEB standing-wave absorption thin>thick, helper integrated with CD-locked + PSD mid-band
