import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.utils import save_image
from pytorch_lightning.callbacks import Callback
try:
    from pytorch_lightning.utilities.distributed import rank_zero_only
except:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        keys = list(images.keys())
        saved_img = None
        rows = 0
        for k in keys:
            img = images[k].cpu()
            if saved_img is None:
                saved_img = img
                rows = img.shape[0]
            else:
                saved_img = torch.cat((saved_img, img), 0)
        filename = "gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        saved_img = (saved_img + 1 ) / 2
        save_image(saved_img.detach(), nrow=rows, fp=path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and hasattr(pl_module, "log_images") and callable(
                pl_module.log_images) and self.max_images > 0):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
