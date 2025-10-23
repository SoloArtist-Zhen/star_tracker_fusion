# Star Tracker–Inertial Fusion (Open Reproducible Pipeline)
Purpose: Demonstrate a **minimal, self-contained** pipeline for spacecraft attitude estimation:
- Simulate a star field and a moving camera with a given FOV.
- Generate noisy 2D star measurements from 3D unit vectors.
- **Lost-in-Space (LIS) identification (toy)** via triangle-invariant matching for an anchor triad.
- Estimate attitude by **Wahba (SVD)**.
- Fuse with synthetic gyro via a **MEKF-lite** (bias-only filter) for temporal smoothing.
- Save plots: sky projection with IDs, attitude error curves, timing, 和 noise sensitivity.

Run:
```bash
python main.py
```
Outputs:
- `outputs/scene_step0.png` : initial annotated star image (IDs, centroids)
- `outputs/attitude_error.png` : time series of attitude error (deg)
- `outputs/noise_sweep.png`   : effect of pixel noise on attitude error
- `outputs/pipeline_report.txt` : quick metrics dump
星跟踪器-惯性融合（开放可复现流程）
目的：演示一个**最小、独立**的航天器姿态估计流程：
- 模拟星场和具有给定视场的移动摄像机。
- 从三维单位向量生成带噪声的二维星体测量值。
- 通过对锚点三元组进行三角不变匹配，实现**迷失空间 (LIS) 识别（小实验）**。
- 使用**Wahba (SVD)**估计姿态。
- 通过**MEKF-lite**（仅偏置滤波器）与合成陀螺仪融合，进行时间平滑。
- 保存图表：带 ID 的天空投影、姿态误差曲线、时间曲线和噪声灵敏度。<img width="960" height="720" alt="scene_step0" src="https://github.com/user-attachments/assets/9c64ca68-833a-4990-8053-981b29a2960d" />
<img width="884" height="544" alt="noise_sweep" src="https://github.com/user-attachments/assets/bd112702-ea99-4512-9945-a362838b9cfd" />
<img width="1020" height="544" alt="attitude_error" src="https://github.com/user-attachments/assets/66f8de5d-15a7-477a-8b76-84418485f11b" />
