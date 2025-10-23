
import numpy as np

def wahba_attitude(obs_cam, ref_inertial, w=None):
    """
    Wahba's problem via SVD (Kabsch):
    obs_cam:  (N,3) unit vectors in camera frame
    ref_inertial: (N,3) corresponding unit vectors in inertial frame
    w:        (N,) weights (optional)
    return:   R (3x3) inertial->camera rotation
    """
    assert obs_cam.shape == ref_inertial.shape
    N = obs_cam.shape[0]
    if w is None:
        w = np.ones(N)
    W = np.diag(w)
    B = obs_cam.T @ W @ ref_inertial
    U, S, Vt = np.linalg.svd(B)
    M = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        M[2,2] = -1.0
    R = U @ M @ Vt
    return R
