
import numpy as np

def quat_mul(q1, q2):
    # Hamilton product q = q1 ⊗ q2 ; q=[w,x,y,z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_from_omega(omega, dt):
    # Small-angle quaternion from body rates (rad/s)
    th = np.linalg.norm(omega)*dt
    if th < 1e-12:
        return np.array([1,0,0,0], dtype=float)
    axis = omega/np.linalg.norm(omega)
    s = np.sin(th/2.0)
    return np.array([np.cos(th/2.0), *(axis*s)])

def quat_to_R(q):
    w,x,y,z = q
    # Normalize
    n = np.linalg.norm(q)
    if n == 0: return np.eye(3)
    w,x,y,z = q/n
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])
    return R

def quat_inv(q):
    w,x,y,z = q
    return np.array([w,-x,-y,-z]) / (np.dot(q,q)+1e-15)

def small_angle_from_quat(dq):
    # dq ~ [1, 0.5*delta], return delta (approx)
    dq = dq/np.linalg.norm(dq)
    return 2.0*dq[1:]

class MEKFBiasOnly:
    """
    Minimal MEKF-like attitude-bias estimator.
    State: gyro bias b_g (3,)
    Attitude carried separately as quaternion q (inertial->cam)
    Process: q_k+1 = q_k ⊗ exp((ω_m - b_g)*dt/2), b_g = const + noise
    Measurement: attitude q_meas from Wahba, innovation on small-angle approx
    """
    def __init__(self, q0=None, b0=None, P0=None, Q=1e-7*np.eye(3), R=1e-5*np.eye(3)):
        self.q = np.array([1,0,0,0],dtype=float) if q0 is None else q0/np.linalg.norm(q0)
        self.b = np.zeros(3) if b0 is None else b0.copy()
        self.P = (1e-6*np.eye(3) if P0 is None else P0.copy())
        self.Q = Q
        self.R = R

    def predict(self, omega_meas, dt):
        dq = quat_from_omega(omega_meas - self.b, dt)
        self.q = quat_mul(self.q, dq)
        self.q = self.q/np.linalg.norm(self.q)
        self.P = self.P + self.Q*dt

    def update(self, q_meas):
        # innovation: delta_theta such that q_meas ≈ dq ⊗ q_pred
        dq = quat_mul(q_meas, quat_inv(self.q))
        delta = small_angle_from_quat(dq)  # 3x1
        H = np.eye(3)  # simplified
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        db = K @ delta
        self.b += db
        self.P = (np.eye(3) - K@H) @ self.P
        # Correct attitude
        corr = np.array([1.0, *(0.5*db)])
        self.q = quat_mul(corr, self.q)
        self.q = self.q/np.linalg.norm(self.q)
        return delta, db
