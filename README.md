# Star Tracker–Inertial Fusion (Open Reproducible Pipeline)
Purpose: Demonstrate a **minimal, self-contained** pipeline for spacecraft attitude estimation:
- Simulate a star field and a moving camera with a given FOV.
- Generate noisy 2D star measurements from 3D unit vectors.
- **Lost-in-Space (LIS) identification (toy)** via triangle-invariant matching for an anchor triad.
- Estimate attitude by **Wahba (SVD)**.
- Fuse with synthetic gyro via a **MEKF-lite** (bias-only filter) for temporal smoothing.
- Save rich plots: sky projection with IDs, attitude error curves, timing, and noise sensitivity.

Run:
```bash
python main.py
```
Outputs:
- `outputs/scene_step0.png` : initial annotated star image (IDs, centroids)
- `outputs/attitude_error.png` : time series of attitude error (deg)
- `outputs/noise_sweep.png`   : effect of pixel noise on attitude error
- `outputs/pipeline_report.txt` : quick metrics dump
