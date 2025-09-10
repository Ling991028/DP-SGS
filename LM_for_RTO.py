'''
LM_RTO: Use Levenberg-Marquardt nonlinear least squares optimization to solve RTO objective function (6.31)
'''

import numpy as np
import configs


lam = 10000
lam_sqrt = np.sqrt(lam)


n_el, h0 = configs.n_el, configs.h0
background = configs.background
margin = configs.margin
el_dist, step = configs.el_dist, configs.step
# seed = configs.seed
mesh_obj, el_pos = configs.mesh_obj, configs.el_pos
fwd = configs.fwd
ex_mat = configs.ex_mat


def AnJ(x, L_sqrt, k):
    f1 = fwd.solve_eit(ex_mat, step, perm=x, parser="std")
    f_x = f1.v
    f_x = f_x + np.random.randn(208) * (np.max(f_x) * 2e-4)
    f_x[f_x < 0] = 0
    f_x = f_x.reshape(-1, 1)

    jac = f1.jac
    A = np.vstack((k * lam_sqrt * f_x, 1/k * L_sqrt @ x.reshape(-1,1))).reshape(-1, 1)
    J = np.vstack((k * lam_sqrt * jac, 1/k * L_sqrt))
    return A, J

def B(y, x0, sigma, L_sqrt, k):
    B0 = np.vstack((k * lam_sqrt * y.reshape(-1,1), 1/k * L_sqrt @ x0.reshape(-1,1))).reshape(-1, 1)
    B1 = sigma
    return B0 + B1


def LM_RTO(y, x0, Q, x_initial, L_sqrt, k, maxiter=20, tol=1e-8):
    u0 = 0
    u_low = 0.25
    u_high = 0.75
    w_up = 2
    w_down = 0.5
    gamma_0 = 0.001
    gamma = 0.001
    sigma = 0.01 * np.random.randn(len(y)+len(x0),1)
    x = x_initial.reshape(-1,1)
    b = B(y, x0, sigma, L_sqrt, k)
    alpha = 1
    del_alpha = alpha / (maxiter + 1)
    eta = 0.005 # Convergence threshold value, adjust as needed. Smaller eta requires more iterations
    for i in range(maxiter):
        A, J = AnJ(x.flatten(), L_sqrt, k)
        res = A - b
        res[0:208] = - res[0:208]
        residual = Q.T @ res
        l = 1/2 * np.linalg.norm(residual) ** 2
        Jx = Q.T @ J
        JxT = Jx.T
        H = JxT @ Jx + gamma * np.eye(Jx.shape[1])
        g = JxT @ residual
        dx = - np.linalg.solve(H, g)
        x_new = x + alpha * dx
        A_new, J_new = AnJ(x_new.flatten(), L_sqrt, k)
        res_new = A_new - b
        res_new[0:208] = - res_new[0:208]
        residual_new = Q.T @ (res_new)
        l_new = 1/2 * np.linalg.norm(residual_new) ** 2
        print("l_new:", l_new)
        r = - 2 * (l - l_new) / (np.dot(dx.T, g))
        if r < u0:
            x = x
            gamma = max(w_up * gamma, gamma_0)
        elif r >= u0 and r < u_low:
            x = x_new
            gamma = max(w_up * gamma, gamma_0)
        elif r >= u_low and r <= u_high:
            x = x_new
            gamma = gamma
        elif r > u_high:
            x = x_new
            gamma = w_down * gamma
        elif gamma < gamma_0:
            x = x_new

        alpha = alpha - del_alpha

    x = np.clip(x, 0, 2.0)
    A, J = AnJ(x.flatten(), L_sqrt, k)
    res = A - b
    res[0:208] = - res[0:208]
    residual = Q.T @ res
    l = 1 / 2 * np.linalg.norm(residual) ** 2
    # print("l:", l)
    converge = int(l < eta)
    return x, converge