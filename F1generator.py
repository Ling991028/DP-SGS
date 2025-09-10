import numpy as np
import torch
import pyeit.eit.jac as jac
from pyeit.eit.fem import Forward
from pyeit.mesh import create, set_perm
from pyeit.mesh.shape import thorax, circle
from pyeit.eit.utils import eit_scan_lines

n_el, h0 = 16, 0.03
seed = 2022
el_dist, step = 1, 1
mesh_obj, el_pos = create(seed, n_el=n_el, fd=circle, h0=h0)
fwd = Forward(mesh_obj, el_pos)
ex_mat = eit_scan_lines(n_el, el_dist)


def h_mat(jac, gamma, K):
    #calculate Hessian matrix
    j_w_j = np.dot(np.transpose(jac),jac)
    r_diag = np.eye(j_w_j.shape[-1])
    j_w_j = K * j_w_j + r_diag / (K * gamma**2)
    # j_w_j = 1e2 * j_w_j + r_diag / (1e6 * gamma ** 2)
    h_matrix = np.linalg.inv(j_w_j)

    return h_matrix

def F1(y, v, miu, lam, max_iters = 10, gtol = 1e-8):
    # y is the true y value, v is the return value of this round's F0 generator
    # x0 is the initial value for this optimization, eta is the step size
    x0 = v + 0.1 * torch.randn(v.size())
    x = x0
    eta = 1

    for i in range(max_iters):
        x_norm = np.linalg.norm(x.detach().numpy().flatten())
        # forward solver
        f1 = fwd.solve_eit(ex_mat, step, perm=x.detach().numpy().flatten(), parser="std")
        fx = f1.v
        fx = fx + np.random.randn(208) * (np.max(fx) * lam)  # 1%
        fx[fx < 0] = 0
        jac = f1.jac
        # residual
        # r0 = fx - y
        r0 = y - fx

        R0 = np.linalg.norm(r0) ** 2
        r1 = x-v
        R1 = np.linalg.norm(r1.detach().numpy().flatten()) ** 2 / (miu ** 2)

        if i == 0:
            K = 3 * np.sqrt(R1 / R0)

        R_average = K * R0
        if i == 0 or i == max_iters - 1:
            print("R0:",R0)
            print("R1:",R1)
            print("R_average:",R_average)
        h_matrix_inv = h_mat(jac, miu, K)
        # r_matrix = np.dot(np.transpose(jac), r0.reshape(-1, 1)) + (x-v).detach().numpy().reshape(-1,1) / miu**2
        r1_matrix = np.dot(np.transpose(jac), r0.reshape(-1, 1)) * K
        r2_matrix = (x - v).detach().numpy().reshape(-1, 1) / (K * miu ** 2)
        # r1_matrix = np.dot(np.transpose(jac), 1e2 * r0.reshape(-1, 1))
        # r2_matrix = (x-v).detach().numpy().reshape(-1,1) / (1e6 * miu ** 2)
        # print("max_r1_matrix:", np.max(r1_matrix))
        # print("max_r2_matrix:", np.max(r2_matrix))
        # update
        p_k_1 = -np.dot(h_matrix_inv, r1_matrix).flatten()
        # print("max_p_k_1:", np.max(p_k_1))
        # print("min_p_k_1:", np.min(p_k_1))
        p_k_2 = -np.dot(h_matrix_inv, r2_matrix).flatten()
        # print("max_p_k_2:", np.max(p_k_2))
        # print("min_p_k_2:", np.min(p_k_2))
        p_k = p_k_1 + p_k_2
        # print("max_p_k:", np.max(p_k))
        # print("min_p_k:", np.min(p_k))

        delta_x = eta * torch.from_numpy(p_k).to(torch.float32).reshape(-1, 1)
        x = x + delta_x
        x = torch.clamp(x, min=0.1, max=2.0)

        eta = eta - 0.1

        # convergence test
        c = np.linalg.norm(delta_x) / x_norm
        if c < gtol:
            print("real iters: ", i)
            break
    # x = torch.clamp(x, min=0, max=2.0)
    return x
