'''
LM_MAP: Use Levenberg-Marquardt nonlinear least squares optimization to find Maximum A Posteriori (MAP) estimate
'''

import numpy as np
import configs


lam = 10000
lam_sqrt = np.sqrt(lam) # Coefficient for adjusting the prior variance size


n_el, h0 = configs.n_el, configs.h0
background = configs.background
margin = configs.margin
el_dist, step = configs.el_dist, configs.step
mesh_obj, el_pos = configs.mesh_obj, configs.el_pos
fwd = configs.fwd
ex_mat = configs.ex_mat


def AnJ(x, L_sqrt, k):
    '''
    :param k: Coefficient for adjusting the relative weighting between prior and likelihood
    :return: The A term in equation (6.7) and its corresponding first derivative J,
         where internal derivatives are approximated using the Jacobian matrix
    '''
    f1 = fwd.solve_eit(ex_mat, step, perm=x, parser="std")
    f_x = f1.v
    f_x = f_x + np.random.randn(208) * (np.max(f_x) * 2e-4)
    f_x[f_x < 0] = 0
    f_x = f_x.reshape(-1, 1)
    jac = f1.jac
    A = np.vstack((k * lam_sqrt * f_x, 1 / k * L_sqrt @ x.reshape(-1, 1))).reshape(-1, 1)
    J = np.vstack((k * lam_sqrt * jac, 1 / k * L_sqrt))
    return A, J

def B(y, x0, L_sqrt, k):
    '''
    :return: The b term in equation (6.7)
    '''
    return np.vstack((k * lam_sqrt * y.reshape(-1,1), 1/k * L_sqrt @ x0.reshape(-1,1))).reshape(-1, 1)

def LM_MAP(y, x0, x_initial, L_sqrt, maxiter=10):
    '''
    Solve for MAP, algorithm reference corresponds to Section 6.2
    :return: x -- MAP solution
             Q -- Q matrix obtained from QR decomposition of the Jacobian matrix J corresponding to the forward process of x
             Thin-QR decomposition is used
    '''
    # Parameters required for LM
    u0 = 0
    u_low = 0.25
    u_high = 0.75
    w_up = 2
    w_down = 0.5
    gamma_0 = 0.001
    gamma = 0.001
    x = x_initial.reshape(-1, 1)
    b = B(y, x0, L_sqrt, 1)
    # Update step size adjustment
    eta = 1
    del_eta = eta / (maxiter + 1)
    for i in range(maxiter):
        if i == 0:
            A, J = AnJ(x.flatten(), L_sqrt, 1)
            residual = A - b
            residual[0:208] = - residual[0:208]
            r1 = residual[0:208]
            r2 = residual[208:]
            K = np.linalg.norm(r2) ** 2 / (np.linalg.norm(r1) ** 2)
            k = np.sqrt(3 * np.sqrt(K))
            A, J = AnJ(x.flatten(), L_sqrt, k)
            b = B(y, x0, L_sqrt, k)
            residual = A - b
            residual[0:208] = - residual[0:208]
            l = 1 / 2 * np.linalg.norm(residual) ** 2
            Jx = J
            JxT = Jx.T
        else:
            A, J = AnJ(x.flatten(), L_sqrt, k)
            b = B(y, x0, L_sqrt, k)
            residual = A - b
            residual[0:208] = - residual[0:208]
            l = 1 / 2 * np.linalg.norm(residual) ** 2
            Jx = J
            JxT = Jx.T
        H = JxT @ Jx + gamma * np.eye(Jx.shape[1])
        g = JxT @ residual
        dx = - np.linalg.solve(H, g)
        x_new = x + eta * dx
        A_new, J_new = AnJ(x_new.flatten(), L_sqrt, k)
        residual_new = A_new - b
        residual_new[0:208] = - residual_new[0:208]
        l_new = 1 / 2 * np.linalg.norm(residual_new) ** 2
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
            gamma = 0
        print("iter:", i, "l:", l, "l_new:", l_new, "r:", r, "gamma:", gamma)
        x = np.clip(x, 0, 2.0)
        eta = eta - del_eta

    # x = np.clip(x, 0, 2.0)
    A, J = AnJ(x.flatten(), L_sqrt, k)
    Q, R = np.linalg.qr(J)
    return x, Q, k
