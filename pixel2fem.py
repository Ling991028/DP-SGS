from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

# pyEIT 2D algorithms modules
from matplotlib.ticker import MultipleLocator

from pyeit.mesh import create, set_perm
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import thorax,circle
import pyeit.eit.jac as jac

from scipy.stats import mode
#%%
def get_most_common_value(arr):
    """
    获取数组中出现次数最多的值
    """
    mode_value, _ = mode(arr.flatten(), keepdims=False) # 找出数组中出现次数最多的值
    if mode_value == 0:
        result = 1
    else:
        result = mode_value
    return result  # 返回出现次数最多的值

def squ2triangle(mesh_obj,img):
    n = img.shape[0]
    pts = mesh_obj["node"] + 1
    tri = mesh_obj["element"]
    center = np.mean(pts[tri], axis=1)  # 计算element的中心

    # 把中心点【数值坐标】转化为矩阵（图像）的【行列坐标】
    delta = 2 / n
    # center_idx = (center // delta).astype(int)
    center_idx = np.round(center / delta).astype(int)  # 使用四舍五入来获取整数行列坐标

    # 上下翻转图像
    img_flip = img[::-1, :]  # 为了img和mesh_obj一致
    img_flip = img_flip.squeeze()

    # 每个中心点找周围的像素点值，用max()，min()  or mean()
    epsilon = 2
    # min_clip = lambda x: np.where(x >= 0, x, 0)
    # max_clip = lambda x: np.where(x <= 127, x, 127)
    min_clip = lambda x: np.clip(np.round(x), 0, n-1).astype(int)
    max_clip = lambda x: np.clip(np.round(x), 0, n-1).astype(int)
    # perm = np.array([img_flip[min_clip(c[1] - epsilon):max_clip(c[1] + epsilon),
    #                  min_clip(c[0] - epsilon):max_clip(c[0] + epsilon)].max() for c in center_idx]) # max

    # perm = np.array([img_flip[c[1], c[0]] for c in center_idx]) # exactly

    # perm = np.array([get_most_common_value(img_flip[
    #                                        min_clip(c[1] - epsilon):max_clip(c[1] + epsilon + 1),
    #                                        min_clip(c[0] - epsilon):max_clip(c[0] + epsilon + 1)])
    #                  for c in center_idx]) # most common

    perm = []
    for c in center_idx:
        location = (c[0] / (n / 2) - 1) ** 2 + (c[1] / (n / 2) - 1) ** 2
        if location > 0.95 ** 2:
            perm.append(1)
        else:
            perm.append(img_flip[min_clip(c[1] - epsilon):max_clip(c[1] + epsilon + 1),
                        min_clip(c[0] - epsilon):max_clip(c[0] + epsilon + 1)].mean())
    perm = np.array(perm) # mean

    # pts = mesh_obj["node"] - 1
    return perm

# if __name__ == "__main__":
#
#     """0. given .png data"""
#     data = np.load(r'/Users/wanghuihui/Documents/Code/paper-EIT/EIT/pyEIT-master/tool-test/26.npz')
#     img = data["xs"][0]
#     y_ori = data["ys"][0]
#     # plt.imshow(img)
#     # plt.plot(y_ori)
#     # plt.show()
#
#     """ 1. transform to triangle from png """
#     n_el = 16
#     # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
#     mesh_obj, el_pos = create(n_el, h0=0.03, fd=circle)
#     pts = mesh_obj["node"]
#     tri = mesh_obj["element"]
#     # test function for altering the permittivity in mesh
#     img_perm = squ2triangle(mesh_obj,img)
#
#     #%%
#     """ 2. calculate simulated data """
#     el_dist, step = 1, 1
#     ex_mat = eit_scan_lines(n_el, el_dist)
#     fwd = Forward(mesh_obj, el_pos)
#     f1 = fwd.solve_eit(ex_mat, step, perm=img_perm, parser="std")
#
#     # %%
#     """ 3. plot """
#     fig, ax = plt.subplots(1,2, figsize=(10, 5))
#     # tripcolor shows values on nodes (shading='flat' or 'gouraud')
#     im = ax[0].tripcolor(
#         pts[:, 0],
#         pts[:, 1],
#         tri,
#         np.real(img_perm),
#         edgecolor="w", linewidth=2,
#         shading="flat",
#         alpha=0.9,
#         cmap=plt.cm.RdBu,
#     )
#     fig.colorbar(im)
#     ax[0].axis("equal")
#     ax[0].axis([-1.2, 1.2, -1.2, 1.2])
#     # 'tricontour' interpolates values on nodes, for example
#     # ax.tricontour(pts[:, 0], pts[:, 1], tri, np.real(node_ds),
#     # shading='flat', alpha=1.0, linewidths=1,
#     # cmap=plt.cm.RdBu)
#     ax[1].plot(f1.v)
#     plt.show()
#
#     #%%
#     """ 4. compare with the original data """
#     delta = ((f1.v - y_ori) / y_ori).mean()
#     fig, axes = plt.subplots(1,2, figsize=(10, 5))
#     axes[0].imshow(img)
#     axes[0].set_title(r"img")
#
#     ax = axes[1]
#     ax.plot(y_ori,label='y_ori')
#     ax.plot(f1.v, label='y_gen')
#     ax.set_title(fr"compare:{delta}")
#     ax.set_ylim(0,0.12)
#     y_major_locator=MultipleLocator(0.02)#以每0.02显示
#     ax.yaxis.set_major_locator(y_major_locator)
#     plt.show()
#
#     # axes[2].plot(f1.v)
#     # axes[2].set_title(r"y_gen")
#     # axes[2].yaxis.set_major_locator(y_major_locator)
#     # axes[2].set_ylim(0,0.12)
#     # plt.show()

