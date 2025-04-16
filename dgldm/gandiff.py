import os
import sys
import random
import torch
sys.path.append(os.path.dirname(__file__))
from baseline import Baseline, instantiate_from_config, get_scheduler, save_image
from auxiliary.aux_utils.ops import compute_grad_gp, calc_adv_loss, calc_recon_loss


class GANDiff(Baseline):
    def __init__(self, input_sty_class_id='sty_class_id', w_aux_gp=10.0, w_aux_adv=1.0, w_aux_rec=0.1,
                 w_aux_vec=0.01, w_aux_off=0.5, prob_self_supervised=10, aux_D_config=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_network_D = instantiate_from_config(aux_D_config)

        # for aux test
        self.input_key_source = self.input_std_src_img
        self.input_key_reference = self.input_sty_ref_img
        self.input_key_target = self.input_sty_src_img

        # for aux train
        self.input_key_imgs = self.input_sty_ref_img
        self.input_key_y_org = input_sty_class_id

        self.w_aux_gp = w_aux_gp
        self.w_aux_adv = w_aux_adv
        self.w_aux_rec = w_aux_rec
        self.w_aux_vec = w_aux_vec
        self.w_aux_off = w_aux_off
        self.prob_self_supervised = prob_self_supervised

        self.automatic_optimization = False

    def configure_optimizers(self):
        # diffuser model
        params = list(self.model.parameters())
        opt_diff = torch.optim.AdamW(params,
            lr=self.optimizer_config.learning_rate,
            betas=(self.optimizer_config.adam_beta1, self.optimizer_config.adam_beta2),
            weight_decay=self.optimizer_config.adam_weight_decay,
            eps=self.optimizer_config.adam_epsilon)

        # auxiliary model
        c_lr, g_lr, d_lr = self.optimizer_config.c_lr, self.optimizer_config.g_lr, self.optimizer_config.d_lr
        c_w_decay, g_w_decay, d_w_decay = (self.optimizer_config.c_w_decay, self.optimizer_config.g_w_decay,
                                           self.optimizer_config.d_w_decay)
        opt_c = torch.optim.Adam(self.aux_network_C.parameters(), lr=c_lr, weight_decay=c_w_decay)
        opt_g = torch.optim.RMSprop(self.aux_network_G.parameters(), lr=g_lr, weight_decay=g_w_decay)
        opt_d = torch.optim.RMSprop(self.aux_network_D.parameters(), lr=d_lr, weight_decay=d_w_decay)
        if self.use_scheduler:
            lr_scheduler = get_scheduler(
                self.optimizer_config.lr_scheduler,
                optimizer=opt_diff,
                num_warmup_steps=self.optimizer_config.lr_warmup_steps * self.optimizer_config.gradient_accumulation_steps,
                num_training_steps=self.optimizer_config.max_train_steps * self.optimizer_config.gradient_accumulation_steps, )
            sch_diff = {"scheduler": lr_scheduler, "interval": "step", 'frequency': 1}
            return [opt_diff, opt_d, opt_g, opt_c], [sch_diff]
        else:
            return [opt_diff, opt_d, opt_g, opt_c], []

    def get_input(self, batch, N=None, *args, **kwargs):
        # x: images, y: class, z: standard
        # org: original, ref: reference
        x_org = self.get_base_input(batch, self.input_key_imgs, N)
        z_ref = self.get_base_input(batch, self.input_std_ref_img, N)
        y_org = batch[self.input_key_y_org][:N]
        x_ref_idx = torch.randperm(x_org.size(0)).to(x_org.device)
        x_ref = x_org.clone()
        x_ref = x_ref[x_ref_idx]
        y_ref = y_org.clone()
        y_ref = y_ref[x_ref_idx]
        z_ref = z_ref[x_ref_idx]
        src_char_ids = self.get_base_input(batch, self.input_src_char_id, N)
        src_char_lens = batch[self.input_src_char_len][:N]
        debug = False
        if debug:
            save_image(x_org.cpu().detach(), nrow=x_org.size(0), fp='x_org.png')
            save_image(x_ref.cpu().detach(), nrow=x_org.size(0), fp='x_ref.png')
            save_image(z_ref.cpu().detach(), nrow=x_org.size(0), fp='z_ref.png')
            assert 1 == 2
        conditions = dict(x_org=x_org, y_org=y_org, x_ref=x_ref, y_ref=y_ref, z_ref=z_ref,
                          src_char_ids=src_char_ids, src_char_lens=src_char_lens)
        return conditions

    def training_step_for_auxD(self, conditions, loss_dict, log_prefix, d_opt):
        x_org, x_ref, y_ref = conditions['x_org'], conditions['x_ref'], conditions['y_ref']
        with torch.no_grad():
            # src_imgs, ref_imgs
            x_fake, _ = self.generate_aux_images(src_imgs=x_org, ref_imgs=x_ref)
        x_ref.requires_grad_()
        d_real_logit, _ = self.aux_network_D(x_ref, y_ref)
        d_fake_logit, _ = self.aux_network_D(x_fake.detach(), y_ref)
        d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
        d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')
        d_adv = d_adv_real + d_adv_fake
        d_gp = self.w_aux_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)
        d_loss = d_adv + d_gp

        d_opt.zero_grad()
        d_adv_real.backward(retain_graph=True)
        d_gp.backward()
        d_adv_fake.backward()
        d_opt.step()

        loss_dict.update({f'{log_prefix}/d_adv': d_adv})
        loss_dict.update({f'{log_prefix}/d_gp': d_gp / self.w_aux_gp})
        loss_dict.update({f'{log_prefix}/d_loss': d_loss})
        return loss_dict

    def training_step_for_pure_Diff(self, source, reference, target, loss_dict, log_prefix):
        # diffuser: std_ref_imgs + pred_sty_src_imgs = pred_sty_ref_imgs
        noise = torch.randn_like(target)
        bsz = target.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=target.device)
        timesteps = timesteps.long()

        # Add noise to the target_images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_target = self.noise_scheduler.add_noise(target, noise, timesteps)

        # Classifier-free training strategy
        context_mask = torch.bernoulli(torch.zeros(bsz) + self.drop_prob)
        context_mask = context_mask.view(-1, 1, 1, 1).expand_as(source).to(source.device)

        source = source * (1 - context_mask) + context_mask
        reference = reference * (1 - context_mask) + context_mask

        # Predict the noise residual and compute loss
        diff_loss, offset_loss, noise_pred = self.cal_diff_offset_loss(timesteps=timesteps,
                                                                       style_images=reference,
                                                                       content_images=source, noise=noise,
                                                                       noisy_target_images=noisy_target)

        loss_dict.update({f'{log_prefix}/diff_loss': diff_loss / self.diffuser_coefficient})
        loss = diff_loss

        # content perceptual loss
        percep_loss = self.cal_perceptron_loss(timesteps=timesteps, noise_pred=noise_pred,
                                               noisy_target_images=noisy_target,
                                               target_images=target, device=target.device)
        loss_dict.update({f'{log_prefix}/percep_loss': percep_loss / (self.perceptual_coefficient + self.eps)})
        loss = loss + percep_loss

        # offset loss
        loss_dict.update({f'{log_prefix}/offset_loss': offset_loss / (self.offset_coefficient + self.eps)})
        loss = loss + offset_loss
        return loss, loss_dict

    def training_step_for_auxCG_Diff(self, conditions, loss_dict, log_prefix, g_opt, c_opt, diff_opt, diff_sch=None):
        # auxiliary loss begin
        x_org, x_ref, y_org, y_ref = conditions['x_org'], conditions['x_ref'], conditions['y_org'], conditions['y_ref']
        src_char_ids, src_char_lens = conditions['src_char_ids'], conditions['src_char_lens']
        s_src = self.aux_network_C.moco(x_org)
        s_ref = self.aux_network_C.moco(x_ref)
        c_src, skip1, skip2 = self.aux_network_G.cnt_encoder(x_org)
        # x_fake: source + reference
        x_fake, g_offset_loss = self.aux_network_G.decode(c_src, s_ref, skip1, skip2)
        # x_rec: source + reference
        x_rec, _ = self.aux_network_G.decode(c_src, s_src, skip1, skip2)
        g_fake_logit, _ = self.aux_network_D(x_fake, y_ref)
        g_rec_logit, _ = self.aux_network_D(x_rec, y_org)
        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        g_adv_rec = calc_adv_loss(g_rec_logit, 'g')
        g_adv = g_adv_fake + g_adv_rec
        g_img_rec = calc_recon_loss(x_rec, x_org)
        c_x_fake, _, _ = self.aux_network_G.cnt_encoder(x_fake)
        g_con_rec = calc_recon_loss(c_x_fake, c_src)
        g_loss = self.w_aux_adv * g_adv + self.w_aux_rec * g_img_rec + self.w_aux_rec * g_con_rec + self.w_aux_off * g_offset_loss
        # auxiliary loss end
        aux_ocr_loss = self.cal_ocr_loss(pred_sty_images=x_fake, std_images=x_org, char_ids=src_char_ids,
                                         char_lens=src_char_lens)
        # todo
        source = conditions['z_ref']
        bsz = source.size(0)
        context_mask = torch.bernoulli(torch.zeros(bsz) + self.prob_self_supervised)
        context_mask = context_mask.view(-1, 1, 1, 1).expand_as(source).to(source.device)
        # context_mask = 1, self-supervised train, reference = x_ref
        # context_mask = 0, auxiliary-supervised train, reference = x_fake
        reference = x_fake * (1 - context_mask) + x_ref * context_mask
        diff_loss, loss_dict = self.training_step_for_pure_Diff(source=source, reference=reference, target=x_ref,
                                                                loss_dict=loss_dict, log_prefix=log_prefix)
        g_loss = g_loss + aux_ocr_loss + diff_loss

        g_opt.zero_grad()
        c_opt.zero_grad()
        diff_opt.zero_grad()
        g_loss.backward()
        c_opt.step()
        g_opt.step()
        diff_opt.step()
        if diff_sch:
            diff_sch.step()

        loss_dict.update({f'{log_prefix}/g_adv': g_adv})
        loss_dict.update({f'{log_prefix}/g_img_rec': g_img_rec})
        loss_dict.update({f'{log_prefix}/g_con_rec': g_con_rec})
        loss_dict.update({f'{log_prefix}/g_offset_loss': g_offset_loss})
        loss_dict.update({f'{log_prefix}/g_loss': g_loss})
        loss_dict.update({f'{log_prefix}/aux_ocr_loss': aux_ocr_loss / (
                self.aux_offset_coefficient + self.eps) / (self.aux_coefficient + self.eps)})
        return loss_dict

    def training_step(self, batch, batch_idx):
        conditions = self.get_input(batch)
        opt_diff, opt_d, opt_g, opt_c = self.optimizers()
        if self.use_scheduler:
            sch_diff = self.lr_schedulers()
        else:
            sch_diff = None

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'

        loss_dict = self.training_step_for_auxD(conditions=conditions, loss_dict=loss_dict, log_prefix=log_prefix,
                                             d_opt=opt_d)
        loss_dict = self.training_step_for_auxCG_Diff(conditions=conditions, loss_dict=loss_dict, log_prefix=log_prefix,
                                              c_opt=opt_c, g_opt=opt_g, diff_opt=opt_diff, diff_sch=sch_diff)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False,
                 batch_size=self.batch_size)
        if self.use_scheduler:
            lr = self.optimizers()[0].param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=self.batch_size)

