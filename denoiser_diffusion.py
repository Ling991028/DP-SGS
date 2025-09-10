# Diffusion denoiser

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.distributed as dist
from PIL import Image
from torchvision import datasets, transforms
from skimage.restoration import estimate_sigma



from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import numpy as np
from pyeit.mesh import create, set_perm
import pyeit.eit.jac as jac
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import thorax, circle

import cv2

from fem2pixel import tri2square
from pixel2fem import squ2triangle

import matplotlib.pyplot as plt



class Denoiser:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    m_path = os.path.join(current_dir, 'model045000.pt')

    def __init__(self, model_path=m_path, image_size=128):
        self.model_path = model_path
        self.image_size = image_size
        self.model = None
        self.diffusion = None
        self.args = None
        self._load_model()

    def _load_model(self):
        defaults = dict(
            clip_denoised=True,
            num_samples=10000,
            batch_size=1,
            use_ddim=False,
            model_path=self.model_path,
            image_size=self.image_size,
        )
        defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        self.args = parser.parse_args()

        dist_util.setup_dist()

        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(self.args, model_and_diffusion_defaults().keys())
        )
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(
            dist_util.load_state_dict(self.model_path, map_location="cpu")
        )
        self.model.to(dist_util.dev())
        self.model.eval()

    def denoiser_diffusion(self, image, t=None):
        image = torch.from_numpy(np.copy(image)).unsqueeze(0)
        noisy_image = image

        noisy_image = noisy_image.numpy().transpose(1, 2, 0)

        # estimating the noise level in images
        sigma_est = estimate_sigma(noisy_image, multichannel=True, average_sigmas=True)


        # computing t^star
        alphas_cumprod_prev = self.diffusion.alphas_cumprod_prev
        differences = np.abs(alphas_cumprod_prev - sigma_est ** 2)

        if t == None :
            t_star = np.argmin(differences).item()
            print(999 - t_star)
            if t_star == 999:
                t_star = t_star - 1
        else:
            t_star = 999-t
            print(999 - t_star)
            if t_star == 999:
                t_star = t_star - 1
        noisy_image = torch.tensor(noisy_image, dtype=torch.float32).unsqueeze(dim=0).permute(0, 3, 1, 2).to(dist_util.dev())

        all_images = []
        all_labels = []
        while len(all_images) * self.args.batch_size < 1:
            model_kwargs = {}
            if self.args.class_cond:
                classes = torch.randint(
                    low=0, high=NUM_CLASSES, size=(self.args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = (
                self.diffusion.p_sample_loop_t if not self.args.use_ddim else self.diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.model,
                (self.args.batch_size, 1, self.args.image_size, self.args.image_size), t_star, noisy_image,
                clip_denoised=self.args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if self.args.class_cond:
                gathered_labels = [
                    torch.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: self.args.num_samples]
        if self.args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: self.args.num_samples]

        dist.barrier()
        a2 = sample.squeeze(dim=0).cpu().numpy()
        return a2.squeeze(axis=-1), 999 - t_star

def get_anomaly(X_img, margin=0.05, num_inclusion=2):

    X = X_img.copy()
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
    # example
    denoiser = Denoiser()
    n_el = 16
    background = 1.0
    margin = 0.05
    el_dist, step = 1, 1
    seed = 2022
    mesh_obj, el_pos = create(seed, n_el=n_el, fd=circle, h0=0.7)

    X_img = tri2square(mesh_obj, 128)
    while True:
        X, is_intersect = get_anomaly(X_img, num_inclusion=4)
        if is_intersect == False:
            break

    kernel_size = (5, 5)
    sigma_x = 3

    circle_mask = X.copy()
    circle_mask[circle_mask != 0] = 1

    X[X == 0] = 1.0

    X_B = cv2.GaussianBlur(X, kernel_size, sigma_x)
    X_B[circle_mask == 0] = 0.0

    X[circle_mask == 0] = 0.0

    X = X_B + 0.2 * np.random.randn(*X_B.shape)

    X_deno = denoiser.denoiser_diffusion(X)[0]

    vmin, vmax = 0, 2
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].imshow(X_B, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    ax[0].set_title('X_real')
    ax[1].imshow(X, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    ax[1].set_title('X_noisy')
    ax[2].imshow(X_deno, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    ax[2].set_title('X_deno')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    im = ax[2].imshow(X_deno, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    colorbar = plt.colorbar(im, cax=cbar_ax)
    colorbar.set_label('Conductivity (S/m)')

    plt.show()

