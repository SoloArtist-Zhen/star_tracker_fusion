
import numpy as np, matplotlib.pyplot as plt, time, math, json
from pathlib import Path
from wahba import wahba_attitude
from mekf import MEKFBiasOnly, quat_to_R, quat_mul, quat_from_omega

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

def rand_unit_sphere(n):
    u = np.random.rand(n)
    v = np.random.rand(n)
    theta = 2*np.pi*u
    phi = np.arccos(2*v-1)
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    S = np.stack([x,y,z],axis=1)
    return S/np.linalg.norm(S,axis=1,keepdims=True)

def project_points(R, stars, f=800, W=1280, H=960, fov_deg=20):
    # camera looks along +Z (camera frame); R is inertial->camera
    cam = (R @ stars.T).T
    # Keep those in front and within FOV cone
    cos_fov = np.cos(np.deg2rad(fov_deg))
    vis = cam[:,2] > 0
    vis = np.logical_and(vis, cam[:,2]/np.linalg.norm(cam,axis=1) > cos_fov)
    vis_idx = np.where(vis)[0]
    camv = cam[vis_idx]
    u = f * (camv[:,0]/camv[:,2]) + W/2
    v = f * (camv[:,1]/camv[:,2]) + H/2
    return vis_idx, np.stack([u,v],axis=1), camv

def add_pixel_noise(px, sigma=0.7):
    return px + np.random.randn(*px.shape)*sigma

def build_triangle_db(stars, K=300):
    # Precompute a small triangle invariant DB (angles among triplets)
    N = stars.shape[0]
    idx = np.random.choice(N, size=K, replace=False)
    triplets = []
    for i in range(0, K-2, 3):
        a,b,c = idx[i], idx[i+1], idx[i+2]
        A = stars[[a,b,c]]
        # pairwise angles
        ang = []
        for p in range(3):
            for q in range(p+1,3):
                ang.append(np.arccos(np.clip(A[p]@A[q], -1,1)))
        ang = np.sort(np.array(ang))
        triplets.append({"ids":[int(a),int(b),int(c)], "angles":ang})
    return triplets

def match_triplet(obs_dirs, cat_triplets, tol=0.01):
    # Given 3 observed star directions in camera frame (rotated inertial unknown),
    # compare their pairwise angles to DB and return best match ids.
    ang = []
    for p in range(3):
        for q in range(p+1,3):
            ang.append(np.arccos(np.clip(obs_dirs[p]@obs_dirs[q], -1,1)))
    ang = np.sort(np.array(ang))
    best = None; best_err = 1e9
    for tri in cat_triplets:
        err = np.linalg.norm(ang - tri["angles"])
        if err < best_err:
            best_err = err; best = tri
    if best_err < tol:
        return best["ids"], best_err
    return None, best_err

def rot_from_euler(roll,pitch,yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz@Ry@Rx

def quat_from_R(R):
    # convert 3x3 to quaternion [w,x,y,z]
    K = np.array([
        [R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
        [R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], 0, 0],
        [R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], 0],
        [R[1,2]-R[2,1], R[2,0]-R[0,2], R[0,1]-R[1,0], R[0,0]+R[1,1]+R[2,2]]
    ])
    K = K/3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]
    if q[0] < 0: q = -q
    return np.array([q[3], q[0], q[1], q[2]])

def ang_err_deg(R_est, R_true):
    Rt = R_est @ R_true.T
    ang = np.arccos(np.clip((np.trace(Rt)-1)/2, -1, 1))
    return np.rad2deg(ang)

def run_sim():
    np.random.seed(42)
    # Catalog & DB
    stars = rand_unit_sphere(2000)
    tri_db = build_triangle_db(stars, K=300)

    # Trajectory (20 s)
    T, dt = 20.0, 0.05
    steps = int(T/dt)
    # True rotation evolves with small body rates
    roll = np.linspace(0, 5*np.pi/180, steps)
    pitch = 0.5*np.sin(np.linspace(0, 2*np.pi, steps))*np.pi/180
    yaw = np.linspace(0, 8*np.pi/180, steps)

    f=800; W=1280; H=960; fov=20
    pix_noise = 0.7

    # Gyro: true w + bias + noise
    w_true = np.vstack([
        np.gradient(roll, dt),
        np.gradient(pitch, dt),
        np.gradient(yaw, dt)
    ]).T
    b_true = np.array([0.02, -0.01, 0.015])  # rad/s bias
    w_meas = w_true + b_true + 0.002*np.random.randn(steps,3)

    mekf = MEKFBiasOnly()
    err_list = []

    for k in range(steps):
        Rtrue = rot_from_euler(roll[k], pitch[k], yaw[k])

        # Visible stars & noisy pixels
        vis_idx, uv, camv = project_points(Rtrue, stars, f=f, W=W, H=H, fov_deg=fov)
        if uv.shape[0] < 8:  # ensure enough stars
            continue
        uv_noisy = add_pixel_noise(uv, sigma=pix_noise)

        # Choose 12 brightest (synthetic: nearest to center)
        center = np.array([W/2, H/2])
        d = np.linalg.norm(uv_noisy - center, axis=1)
        order = np.argsort(d)[:12]
        sel = order[:12]
        obs_px = uv_noisy[sel]
        obs_dirs = camv[sel] / np.linalg.norm(camv[sel], axis=1, keepdims=True)

        # LIS toy: match a triangle (first 3 of obs) -> get catalog IDs (if close)
        ids, err = match_triplet(obs_dirs[:3], tri_db, tol=0.02)
        if ids is None:
            # Fallback: pick random ids of same size (demo only)
            ids = np.random.choice(stars.shape[0], size=3, replace=False).tolist()
        # Build correspondences for first 6 points by nearest dir to matched tri-rot
        # Estimate attitude from first 3 as seed:
        R_seed = wahba_attitude(obs_dirs[:3], stars[ids])
        # Use seed to predict other correspondences by nearest neighbor (demo)
        pred_inertial = (R_seed.T @ obs_dirs[3:].T).T  # back to inertial approx
        # nearest neighbors in catalog (coarse; demo-only)
        cat_idx = []
        for v in pred_inertial:
            dots = stars @ v
            cat_idx.append(int(np.argmax(dots)))
        ref = np.vstack([stars[ids], stars[cat_idx[:3]]])
        obc = np.vstack([obs_dirs[:3], obs_dirs[3:6]])

        # Wahba for measurement attitude
        R_meas = wahba_attitude(obc, ref)
        # MEKF predict/update
        mekf.predict(w_meas[k], dt)
        # Convert R_meas to quaternion for update
        def R_to_quat(R):
            t = np.trace(R)
            if t > 0:
                s = math.sqrt(t+1.0)*2
                w = 0.25*s
                x = (R[2,1]-R[1,2])/s
                y = (R[0,2]-R[2,0])/s
                z = (R[1,0]-R[0,1])/s
            else:
                i = np.argmax(np.diag(R))
                if i == 0:
                    s = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
                    w = (R[2,1]-R[1,2])/s; x = 0.25*s; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
                elif i == 1:
                    s = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
                    w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = 0.25*s; z = (R[1,2]+R[2,1])/s
                else:
                    s = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
                    w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = 0.25*s
            q = np.array([w,x,y,z])
            return q/np.linalg.norm(q)
        q_meas = R_to_quat(R_meas)
        mekf.update(q_meas)

        # Error
        Rest = quat_to_R(mekf.q)
        err_list.append(ang_err_deg(Rest, Rtrue))

        # Plot scene for the first step
        if k == 0:
            plt.figure(figsize=(6,4.5))
            plt.scatter(uv[:,0], uv[:,1], s=5, alpha=0.4, label="All stars")
            plt.scatter(obs_px[:,0], obs_px[:,1], s=25, label="Selected")
            for i,(x,y) in enumerate(obs_px):
                plt.text(x+3,y+3,str(i), fontsize=7)
            plt.gca().invert_yaxis()
            plt.title("Synthetic star image (step 0)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUT/"scene_step0.png", dpi=160)
            plt.close()

    # Plot attitude error
    t = np.arange(len(err_list))*dt
    plt.figure(figsize=(6,3.2))
    plt.plot(t, err_list)
    plt.xlabel("Time (s)"); plt.ylabel("Attitude error (deg)"); plt.title("MEKF-lite attitude error")
    plt.grid(True); plt.tight_layout()
    plt.savefig(OUT/"attitude_error.png", dpi=170); plt.close()

    # Noise sweep
    sigmas = [0.2,0.5,0.7,1.0,1.5]
    errs = []
    for s in sigmas:
        np.random.seed(1)
        # single frame replication
        Rtrue = rot_from_euler(roll[5], pitch[5], yaw[5])
        vis_idx, uv, camv = project_points(Rtrue, stars, f=800, W=1280, H=960, fov_deg=20)
        uv_noisy = add_pixel_noise(uv, sigma=s)
        center = np.array([1280/2, 960/2])
        d = np.linalg.norm(uv_noisy - center, axis=1)
        sel = np.argsort(d)[:12]
        obs_px = uv_noisy[sel]; obs_dirs = camv[sel]/np.linalg.norm(camv[sel],axis=1,keepdims=True)
        ids, err = match_triplet(obs_dirs[:3], build_triangle_db(stars, K=300), tol=0.02)
        if ids is None:
            ids = np.random.choice(stars.shape[0], size=3, replace=False).tolist()
        R_seed = wahba_attitude(obs_dirs[:3], stars[ids])
        pred_inertial = (R_seed.T @ obs_dirs[3:].T).T
        cat_idx = []
        for v in pred_inertial:
            dots = stars @ v
            cat_idx.append(int(np.argmax(dots)))
        ref = np.vstack([stars[ids], stars[cat_idx[:3]]])
        obc = np.vstack([obs_dirs[:3], obs_dirs[3:6]])
        R_meas = wahba_attitude(obc, ref)
        errs.append(ang_err_deg(R_meas, Rtrue))
    plt.figure(figsize=(5.2,3.2))
    plt.plot(sigmas, errs, marker="o")
    plt.xlabel("Pixel noise σ (px)"); plt.ylabel("Single-frame error (deg)")
    plt.title("Noise sensitivity at one frame")
    plt.grid(True); plt.tight_layout()
    plt.savefig(OUT/"noise_sweep.png", dpi=170); plt.close()

    (OUT/"pipeline_report.txt").write_text(
        "Frames processed: %d\nFinal bias estimate (rad/s): %s\nFinal attitude error (deg): %.4f\n" %(
            len(err_list), np.array2string(mekf.b, precision=5), err_list[-1] if err_list else float('nan')
        ), encoding="utf-8"
    )

if __name__ == "__main__":
    run_sim()
