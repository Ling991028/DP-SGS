from pyeit.eit.interp2d import meshgrid,weight_idw
from pyeit.mesh import create, set_perm
from pyeit.mesh.shape import thorax,circle
import numpy as np
import matplotlib.pyplot as plt

def tri2square(mesh_obj,n=64):
    pts = mesh_obj["node"]
    tri = mesh_obj["element"]
    xg, yg, mask = meshgrid(pts,n=n)
    im = np.ones_like(mask)
    # mapping from values on xy to values on xyi
    xy = np.mean(pts[tri], axis=1)
    xyi = np.vstack((xg.flatten(), yg.flatten())).T
    w_mat = weight_idw(xy, xyi, k=1)     #xxx 可调节参数
    # w_mat = weight_sigmod(xy, xyi,s=100)
    im = np.dot(w_mat.T, mesh_obj["perm"])
    # im = weight_linear_rbf(xy, xyi, mesh_new['perm'])
    im[mask] = 0.0
    # reshape to grid size
    im = im.reshape(xg.shape)
    return im[::-1,:]

# def squ2triangle(mesh_obj,v):
#     pts = mesh_obj["node"] + 1
#     tri = mesh_obj["element"]
#     center = np.mean(pts[tri], axis=1)  # 计算element的中心
#
#     # 把中心点【数值坐标】转化为矩阵（图像）的【行列坐标】
#     delta = 2 / 128
#     center_idx = (center // delta).astype(int)
#
#     # 上下翻转图像
#     img_flip = v[::-1, :]  # 为了img和mesh_obj一致
#     img_flip = img_flip.squeeze()
#     # 每个中心点找周围的像素点值，用max()，min()  or mean()
#     epsilon = 2
#     min_clip = lambda x: np.where(x >= 0, x, 0)
#     max_clip = lambda x: np.where(x <= 127, x, 127)
#     perm = np.array([img_flip[min_clip(c[1] - epsilon):max_clip(c[1] + epsilon),
#                      min_clip(c[0] - epsilon):max_clip(c[0] + epsilon)].max() for c in center_idx])
#     # pts = mesh_obj["node"] - 1
#     return perm

# if __name__ == "__main__":
#
#     # FEM->pixel
#     n_el, h0 = 16, 0.07071
#     seed = 2022
#     # el_dist, step = 1, 1
#     mesh_obj, el_pos = create(seed, n_el=n_el, fd=circle, h0=h0)
#
#
#     anomaly = [
#         {"x": 0.5, "y": 0.5, "d": 0.2, "perm": 10},
#         {"x": -0.2, "y": -0.2, "d": 0.4, "perm": 20},
#     ]
#     background = 1.0
#     mesh_new = set_perm(mesh_obj, anomaly=anomaly, background=background)
#
#     n = 128
#     img = tri2square(mesh_new, n=n)
#
#     plt.imshow(img)
#     plt.show()

