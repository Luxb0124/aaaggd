import os
import torch
import sys
sys.path.append(os.path.dirname(__file__))
from baseline import Baseline


# FontDiffuser
class SupervisedDiff(Baseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def p_losses(self, conditions, *args, **kwargs):
        std_src_imgs = conditions['std_src_imgs']
        std_ref_imgs = conditions['std_ref_imgs']
        sty_src_imgs = conditions['sty_src_imgs']
        sty_ref_imgs = conditions['sty_ref_imgs']
        src_chars = conditions['src_chars']
        ref_chars = conditions['ref_chars']
        src_char_ids = conditions['src_char_ids']
        ref_char_ids = conditions['ref_char_ids']
        src_char_lens = conditions['src_char_lens']
        ref_char_lens = conditions['ref_char_lens']

        # diffuser: std_ref_imgs + sty_src_imgs = pred_sty_ref_imgs
        noise = torch.randn_like(sty_ref_imgs)
        bsz = sty_ref_imgs.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=sty_ref_imgs.device)
        timesteps = timesteps.long()
        # Add noise to the target_images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_sty_ref_imgs = self.noise_scheduler.add_noise(sty_ref_imgs, noise, timesteps)
        # Classifier-free training strategy
        context_mask = torch.bernoulli(torch.zeros(bsz) + self.drop_prob)
        for i, mask_value in enumerate(context_mask):
            if mask_value == 1:
                std_ref_imgs[i, :, :, :] = 1
                sty_src_imgs[i, :, :, :] = 1

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        # Predict the noise residual and compute loss
        diff_loss, offset_loss, noise_pred = self.cal_diff_offset_loss(timesteps=timesteps,
                                                                       style_images=sty_src_imgs,
                                                                       content_images=std_ref_imgs, noise=noise,
                                                                       noisy_target_images=noisy_sty_ref_imgs)

        loss_dict.update({f'{log_prefix}/diff_loss': diff_loss / self.diffuser_coefficient})
        loss = diff_loss
        # content perceptual loss
        percep_loss = self.cal_perceptron_loss(timesteps=timesteps, noise_pred=noise_pred,
                                               noisy_target_images=noisy_sty_ref_imgs,
                                               target_images=sty_ref_imgs, device=sty_ref_imgs.device)
        loss_dict.update({f'{log_prefix}/percep_loss': percep_loss / (self.perceptual_coefficient + self.eps)})
        loss = loss + percep_loss
        # offset loss
        loss_dict.update({f'{log_prefix}/offset_loss': offset_loss / (self.offset_coefficient + self.eps)})
        loss = loss + offset_loss
        return loss, loss_dict
