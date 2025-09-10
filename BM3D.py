# GPnP-BM3D Algorithm Posterior Sampling Example

import cv2
import numpy as np
from tqdm import tqdm
import bm3d

from F1generator import F1

from pyeit.mesh import create, set_perm
import pyeit.eit.jac as jac
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import thorax, circle

import matplotlib.pyplot as plt


from fem2pixel import tri2square
from pixel2fem import squ2triangle


import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys


sys.path.append('../')
sys.path.append('../../')
import torch

device = torch.device('cpu')

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
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
    alpha = 1
    beta = 0.25
    sigma_max = 0.2
    sigma_min = 0.001
    N = 50
    lam = 2e-4
    K = 20
    sigma_delta = (sigma_max - sigma_min) / (N - K - 1)

    n_el, h0 = 16, 0.03
    background = 1.0
    margin = 0.05
    el_dist, step = 1, 1
    seed = 2022
    mesh_obj, el_pos = create(seed, n_el=n_el, fd=circle, h0=h0)
    fwd = Forward(mesh_obj, el_pos)
    ex_mat = eit_scan_lines(n_el, el_dist)
    img = tri2square(mesh_obj, 128)

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

    X_real = squ2triangle(mesh_obj, X_real_img)
    X_real = torch.from_numpy(X_real).to(torch.float32).reshape(-1, 1)

    X_initial = torch.ones_like(X_real) + 0.05 * torch.randn(X_real.size())

    f1 = fwd.solve_eit(ex_mat, step, perm=X_real.detach().numpy().flatten(), parser="std")
    Y = f1.v
    Y = Y + np.random.randn(208) * (np.max(Y) * lam)
    Y[Y < 0] = 0
    mesh_initial = mesh_obj.copy()
    mesh_initial["perm"] = X_initial.detach().numpy().flatten()
    X = tri2square(mesh_initial, 128)
    mesh_X = mesh_initial.copy()
    mesh_recon = mesh_initial.copy()

    for n in tqdm(range(N), desc='Overall Iterations'):
        if n < N - K:
            sigma = (N - n - K - 1) * sigma_delta + sigma_min
        else:
            sigma = sigma_min
        gamma = np.sqrt(beta) * sigma

        X_recon = bm3d.bm3d(X, sigma_psd=alpha * sigma) # BM3D denoiser
        X_recon = squ2triangle(mesh_recon, X_recon)
        X_recon = torch.from_numpy(X_recon).to(torch.float32).reshape(-1, 1)
        X_recon = torch.clamp(X_recon, min=0.1, max=2.0)
        X = squ2triangle(mesh_X, X)
        X = torch.from_numpy(X).to(torch.float32).reshape(-1, 1)

        X = (1 - beta) * X + beta * X_recon + gamma * torch.randn_like(X)
        X = F1(Y, X, gamma, lam, max_iters=8) + gamma * torch.randn_like(X)

        mesh_X["perm"] = X.detach().numpy().flatten()
        X = tri2square(mesh_X, 128)

    X = squ2triangle(mesh_X, X)
    X = torch.from_numpy(X).to(torch.float32).reshape(-1, 1)

    # plot
    pts = mesh_obj['node']
    tri = mesh_obj['element']

    X_real = X_real.detach().numpy().flatten()
    X = X.detach().numpy().flatten()
    X_initial = X_initial.detach().numpy().flatten()

    MSE = mean_squared_error(X_real, X)
    print('MSE:', MSE)

    SSIM = ssim(X_real, X, data_range=X_real.max() - X_real.min())
    print('SSIM:', SSIM)

    perm_avg_X_real = np.zeros(pts.shape[0])
    perm_avg_X = np.zeros(pts.shape[0])
    perm_avg_X_initial = np.zeros(pts.shape[0])

    count_X_real = np.zeros(pts.shape[0])
    count_X = np.zeros(pts.shape[0])
    count_X_initial = np.zeros(pts.shape[0])

    for i in range(tri.shape[0]):
        for j in range(3):
            perm_avg_X_real[tri[i, j]] += X_real[i]
            perm_avg_X[tri[i, j]] += X[i]
            perm_avg_X_initial[tri[i, j]] += X_initial[i]
            count_X_real[tri[i, j]] += 1
            count_X[tri[i, j]] += 1
            count_X_initial[tri[i, j]] += 1

    for i in range(pts.shape[0]):
        perm_avg_X_real[i] /= count_X_real[i]
        perm_avg_X[i] /= count_X[i]
        perm_avg_X_initial[i] /= count_X_initial[i]

    vmin, vmax = np.min(np.real(perm_avg_X_real)), np.max(np.real(perm_avg_X_real))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].tripcolor(pts[:, 0], pts[:, 1], tri, np.real(perm_avg_X_initial),
                    shading="gouraud", alpha=1, cmap=plt.cm.jet,
                    vmin=vmin, vmax=vmax)
    ax[0].set_title('Initial')
    ax[1].tripcolor(pts[:, 0], pts[:, 1], tri, np.real(perm_avg_X),
                    shading="gouraud", alpha=1, cmap=plt.cm.jet,
                    vmin=vmin, vmax=vmax)
    ax[1].set_title('Reconstruction')
    ax[2].tripcolor(pts[:, 0], pts[:, 1], tri, np.real(perm_avg_X_real),
                    shading="gouraud", alpha=1, cmap=plt.cm.jet,
                    vmin=vmin, vmax=vmax, antialiased='True')
    ax[2].set_title('Ground Truth')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    im = ax[2].tripcolor(pts[:, 0], pts[:, 1], tri, np.real(perm_avg_X_real),
                         shading="gouraud", alpha=1, cmap=plt.cm.jet,
                         vmin=vmin, vmax=vmax, antialiased='True')
    colorbar = plt.colorbar(im, cax=cbar_ax)
    colorbar.set_label('Conductivity (S/m)')

    plt.show()






