'''
RTO-MH Algorithm Posterior Sampling Example

Implementation based on:
Bardsley, J. M. (2018). Computational Uncertainty Quantification for Inverse Problems:
An Introduction to Singular Integrals. Society for Industrial and Applied Mathematics.

Specific reference to Chapter 6:
"Markov Chain Monte Carlo Methods for Nonlinear Inverse Problems", pp. [105-124].

LM_MAP: Use Levenberg-Marquardt nonlinear least squares optimization to find Maximum A Posteriori (MAP) estimate
LM_RTO: Use Levenberg-Marquardt nonlinear least squares optimization to solve RTO objective function (6.31)
'''

import numpy as np
import configs
import cv2

from LM_for_MAP import LM_MAP, AnJ, B
from LM_for_RTO import LM_RTO
import matplotlib.pyplot as plt

import time
from tqdm import tqdm
from pyeit.mesh import set_perm
from fem2pixel import tri2square
from pixel2fem import squ2triangle


def lnc(Q,J,r):
    '''
    Calculate the logarithm of the 'posterior probability' for each sample in RTO-MH, corresponding to the final unnumbered formula on page 111
    :param J: Jacobian matrix of the forward process corresponding to the MAP solution
    :param Q: Q matrix obtained from QR decomposition of J
    :param r: Upper block of the partitioned matrix obtained from Ax-b in equation (6.7)
    :return: Logarithm of the 'posterior probability'
    '''
    c1 = np.log(np.abs(np.linalg.det(Q.T @ J)))
    c2 = 1/2 * np.linalg.norm(r) ** 2
    c3 = 1/2 * np.linalg.norm(Q.T @ r) ** 2
    return c1 + c2 - c3

def accr(x,x_new):
    '''
    Calculate the acceptance rate for each iteration of RTO-MH
    '''
    A_x, J_x = AnJ(x.flatten(), L_sqrt, k_map)
    r_x = A_x - b
    r_x[0:208] = - r_x[0:208]
    lnc_x = lnc(Q_map, J_x, r_x)
    A_new, J_new = AnJ(x_new.flatten(), L_sqrt, k_map)
    r_new = A_new - b
    r_new[0:208] = - r_new[0:208]
    lnc_new = lnc(Q_map, J_new, r_new)
    rate = min(1, np.exp(lnc_x - lnc_new))
    return rate

def cov_for_x(mesh):
    '''
    Set the covariance matrix for the Gaussian prior
    '''
    k = len(mesh['perm'])
    L = np.empty((k, 2))
    D = []
    gam = 0.9
    d = 0.1
    for i in range(k):
        center = mesh['node'][mesh['element'][i]].mean(axis=0)
        L[i, :] = center
    for i in range(k):
        for j in range(k):
            point_1 = L[i, :].squeeze()
            point_2 = L[j, :].squeeze()
            D.append(np.linalg.norm(point_1 - point_2, ord=2))
    D = np.array(D).reshape((k, k))
    cov = gam * np.exp(-D / d)
    return cov

def get_anomaly(img, margin=0.05, num_inclusion=2, seed = None):
    if seed is not None:
        np.random.seed(seed)

    X = img.copy()
    x, y = np.meshgrid(np.arange(-64, 64), np.arange(-64, 64))
    x = x.astype(np.float32) / 64
    y = y.astype(np.float32) / 64


    if num_inclusion == 1:
        x1 = np.random.uniform(-0.55, 0.55, 1)
        y1 = np.random.uniform(-0.55, 0.55, 1)
        r = 0.15 + np.random.rand(1) * 0.1
        assert 0.15 <= r.mean() <= 0.25
        anomaly_circle = (x - x1) ** 2 + (y - y1) ** 2 <= r ** 2
        X[anomaly_circle] = 1.5
        is_intersect = False
    if num_inclusion == 2:
        x1, x2 = np.random.uniform(-0.55, 0.55, 2)
        y1, y2 = np.random.uniform(-0.55, 0.55, 2)
        r = 0.15 + np.random.rand(2) * 0.1
        assert 0.15 <= r.mean() <= 0.25
        anomaly_circle_1 = (x - x1) ** 2 + (y - y1) ** 2 <= r[0] ** 2
        anomaly_circle_2 = (x - x2) ** 2 + (y - y2) ** 2 <= r[1] ** 2
        X[anomaly_circle_1] = 1.5
        X[anomaly_circle_2] = 0.5
        is_intersect = (x1 - x2) ** 2 + (y1 - y2) ** 2 < (r.sum() + margin) ** 2

    elif num_inclusion == 3:
        x1, x2, x3 = np.random.uniform(-0.55, 0.55, 3)
        y1, y2, y3 = np.random.uniform(-0.55, 0.55, 3)
        r = 0.15 + np.random.rand(3) * 0.1
        assert 0.15 <= r.mean() <= 0.25
        anomaly_circle_1 = (x - x1) ** 2 + (y - y1) ** 2 <= r[0] ** 2
        anomaly_circle_2 = (x - x2) ** 2 + (y - y2) ** 2 <= r[1] ** 2
        anomaly_circle_3 = (x - x3) ** 2 + (y - y3) ** 2 <= r[2] ** 2
        X[anomaly_circle_1] = 1.5
        X[anomaly_circle_2] = 1.2
        X[anomaly_circle_3] = 0.5
        is_intersect = ((x1 - x2) ** 2 + (y1 - y2) ** 2 < (r[0] + r[1] + margin) ** 2) or \
                       ((x1 - x3) ** 2 + (y1 - y3) ** 2 < (r[0] + r[2] + margin) ** 2) or \
                       ((x3 - x2) ** 2 + (y3 - y2) ** 2 < (r[2] + r[1] + margin) ** 2)
    elif num_inclusion == 4:
        x1, x2, x3, x4 = np.random.uniform(-0.55, 0.55, 4)
        y1, y2, y3, y4 = np.random.uniform(-0.55, 0.55, 4)
        r = 0.15 + np.random.rand(4) * 0.1
        assert 0.15 <= r.mean() <= 0.25
        anomaly_circle_1 = (x - x1) ** 2 + (y - y1) ** 2 <= r[0] ** 2
        anomaly_circle_2 = (x - x2) ** 2 + (y - y2) ** 2 <= r[1] ** 2
        anomaly_circle_3 = (x - x3) ** 2 + (y - y3) ** 2 <= r[2] ** 2
        anomaly_circle_4 = (x - x4) ** 2 + (y - y4) ** 2 <= r[3] ** 2
        X[anomaly_circle_1] = 1.5
        X[anomaly_circle_2] = 1.2
        X[anomaly_circle_3] = 0.5
        X[anomaly_circle_4] = 0.2
        is_intersect = ((x1 - x2) ** 2 + (y1 - y2) ** 2 < (r[0] + r[1] + margin) ** 2) or \
                       ((x1 - x3) ** 2 + (y1 - y3) ** 2 < (r[0] + r[2] + margin) ** 2) or \
                       ((x3 - x2) ** 2 + (y3 - y2) ** 2 < (r[1] + r[2] + margin) ** 2) or \
                       ((x4 - x1) ** 2 + (y4 - y1) ** 2 < (r[3] + r[0] + margin) ** 2) or \
                       ((x4 - x2) ** 2 + (y4 - y2) ** 2 < (r[3] + r[1] + margin) ** 2) or \
                       ((x4 - x3) ** 2 + (y4 - y3) ** 2 < (r[3] + r[2] + margin) ** 2)
    return X,is_intersect

if __name__ == "__main__":
    n_el, h0 = configs.n_el, configs.h0
    background = configs.background
    margin = configs.margin
    el_dist, step = configs.el_dist, configs.step
    mesh_obj, el_pos = configs.mesh_obj, configs.el_pos
    fwd = configs.fwd
    ex_mat = configs.ex_mat

    img = tri2square(mesh_obj, 128)

    cov = cov_for_x(mesh_obj)
    L = np.linalg.inv(cov)

    eigenvalues, eigenvectors = np.linalg.eig(L)
    D_sqrt = np.diag(np.sqrt(eigenvalues))
    L_sqrt = np.dot(np.dot(eigenvectors, D_sqrt), np.linalg.inv(eigenvectors))

    # L_sqrt = np.linalg.cholesky(L) # Calculate the matrix square root in equation (6.7)

    while True:
        seed = int(time.time())
        num_inclusion = 4
        X_img, is_intersect = get_anomaly(img, margin, num_inclusion=num_inclusion, seed = seed)
        if is_intersect == 0:
            break

    kernel_size = (5, 5)
    sigma_x = 3
    circle_mask = X_img.copy()
    circle_mask[circle_mask != 0] = 1
    X_img[X_img == 0] = 1.0
    X_real_img = cv2.GaussianBlur(X_img, kernel_size, sigma_x)
    X_real_img[circle_mask == 0] = 0.0
    X_img[circle_mask == 0] = 0.0

    x_real = squ2triangle(mesh_obj, X_real_img)

    y = fwd.solve_eit(ex_mat, step, perm=x_real, parser="std").v
    y = y + np.random.randn(208) * (np.max(y) * 2e-4) # Observation with noise
    y[y < 0] = 0
    x0 = np.ones_like(x_real) # Mean of prior

    x_initial = np.ones_like(x_real) + 0.05 * np.random.randn(x_real.size)

    x_map, Q_map, k_map = LM_MAP(y, x0, x_initial, L_sqrt) # Calculate the value of MAP

    b = B(y, x0, L_sqrt, k_map) # The b term in equation (6.7) corresponding to MAP, used for calculating the acceptance rate at each step

    x = x_map

    N = 5
    acc_rate = 0 # Overall sample acceptance rate for the entire iteration
    X = np.zeros((len(x),N))

    # RT0-MH
    for n in tqdm(range(N), desc='Overall Iterations'):
        while True:
            x_new,converge = LM_RTO(y, x0, Q_map, x_map, L_sqrt, k_map, maxiter=20)
            if converge == 1:
                break
        alpha = np.random.rand()
        a = accr(x,x_new)
        if a > alpha:
            x = x_new
            acc_rate = acc_rate + 1
        else:
            x = x
        X[:,n] = x.flatten()

    acc_rate = acc_rate / N
    print('acc_rate:',acc_rate)

    x_recon = np.mean(X,axis=1)

    # Plot
    pts = mesh_obj['node']
    tri = mesh_obj['element']
    x_recon = x_recon.flatten()

    vmin, vmax = np.min(np.real(x_real)), np.max(np.real(x_real))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].tripcolor(pts[:, 0], pts[:, 1], tri, np.real(x_initial),
                    shading="flat", alpha=1.0, cmap=plt.cm.viridis,
                    vmin=vmin, vmax=vmax)
    ax[0].set_title('Initial')
    ax[1].tripcolor(pts[:, 0], pts[:, 1], tri, np.real(x_recon),
                    shading="flat", alpha=1.0, cmap=plt.cm.viridis,
                    vmin=vmin, vmax=vmax)
    ax[1].set_title('Reconstruction')
    ax[2].tripcolor(pts[:, 0], pts[:, 1], tri, np.real(x_real),
                    shading="flat", alpha=1.0, cmap=plt.cm.viridis,
                    vmin=vmin, vmax=vmax)
    ax[2].set_title('Ground Truth')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    im = ax[2].tripcolor(pts[:, 0], pts[:, 1], tri, np.real(x_real),
                         shading="flat", alpha=1.0, cmap=plt.cm.viridis,
                         vmin=vmin, vmax=vmax)
    colorbar = plt.colorbar(im, cax=cbar_ax)
    colorbar.set_label('Conductivity (S/m)')

    plt.show()




