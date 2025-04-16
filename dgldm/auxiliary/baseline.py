import os
import torch
import pytorch_lightning as pl
from einops import rearrange
from torchvision.utils import save_image
from aux_utils.util import instantiate_from_config
from aux_utils.ops import compute_grad_gp, calc_adv_loss, calc_recon_loss


class Baseline(pl.LightningModule):
    def __init__(self, C_config, G_config, D_config, optimizer_config, dataset_config, dataloader_config,
                 envs_config, logger_config, test_dataset_config, test_config,
                 input_std_src_img='std_src_img', input_sty_src_img='sty_src_img',
                 input_sty_ref_img='sty_ref_img', input_sty_class_id='sty_class_id',
                 w_gp=10.0, w_adv=1.0, w_rec=0.1, w_vec=0.01, w_off=0.5, seed=0, ):
        super().__init__()
        # modules
        self.network_C = instantiate_from_config(C_config)
        self.network_G = instantiate_from_config(G_config)
        self.network_D = instantiate_from_config(D_config)
        # optimizer configures
        self.optimizer_config = optimizer_config

        self.envs_config = envs_config
        self.logger_config = logger_config
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        self.optimizer_config = optimizer_config
        self.test_dataset_config = test_dataset_config
        self.test_config = test_config

        # for test
        self.input_key_source = input_std_src_img
        self.input_key_reference = input_sty_ref_img
        self.input_key_target = input_sty_src_img

        # for train
        self.input_key_imgs = input_sty_ref_img
        self.input_key_y_org = input_sty_class_id

        self.w_gp = w_gp
        self.w_adv = w_adv
        self.w_rec = w_rec
        self.w_vec = w_vec
        self.w_off = w_off

        self.generator = torch.manual_seed(seed)
        self.batch_size = self.dataloader_config.train_batch_size
        self.automatic_optimization = False

    def configure_optimizers(self):
        # 1e-4, 1e-4, 1e-4
        c_lr, g_lr, d_lr = self.optimizer_config.c_lr, self.optimizer_config.g_lr, self.optimizer_config.d_lr
        # 0.001, 0.0001, 0.0001
        c_w_decay, g_w_decay, d_w_decay = (self.optimizer_config.c_w_decay, self.optimizer_config.g_w_decay,
                                           self.optimizer_config.d_w_decay)
        opt_c = torch.optim.Adam(self.network_C.parameters(), lr=c_lr, weight_decay=c_w_decay)
        opt_g = torch.optim.RMSprop(self.network_G.parameters(), lr=g_lr, weight_decay=g_w_decay)
        opt_d = torch.optim.RMSprop(self.network_D.parameters(), lr=d_lr, weight_decay=d_w_decay)
        return [opt_d, opt_g, opt_c], []

    def get_base_input(self, batch, k, N):
        x = batch[k][:N]
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 3:
                x = x[..., None]
                x = rearrange(x, 'b h w c -> b c h w').contiguous()
            x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_input(self, batch, N=None, *args, **kwargs):
        # x: images, y: class
        # org: original, ref: reference
        x_org = self.get_base_input(batch, self.input_key_imgs, N)
        y_org = batch[self.input_key_y_org][:N]
        x_ref_idx = torch.randperm(x_org.size(0)).to(x_org.device)
        x_ref = x_org.clone()
        x_ref = x_ref[x_ref_idx]
        y_ref = y_org.clone()
        y_ref = y_ref[x_ref_idx]
        conditions = dict(x_org=x_org, y_org=y_org, x_ref=x_ref, y_ref=y_ref)
        return conditions

    def generate(self, source, reference):
        style_feature = self.network_C.moco(reference)
        content_feature, content_skip1, content_skip2 = self.network_G.cnt_encoder(source)
        fake, offset = self.network_G.decode(content_feature, style_feature, content_skip1, content_skip2)
        return fake, offset

    def trainint_step_for_D(self, conditions, loss_dict, log_prefix, d_opt):
        x_org, x_ref, y_ref = conditions['x_org'], conditions['x_ref'], conditions['y_ref']
        with torch.no_grad():
            x_fake, _ = self.generate(source=x_org, reference=x_ref)
        x_ref.requires_grad_()
        d_real_logit, _ = self.network_D(x_ref, y_ref)
        d_fake_logit, _ = self.network_D(x_fake.detach(), y_ref)
        d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
        d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')
        d_adv = d_adv_real + d_adv_fake
        d_gp = self.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)
        d_loss = d_adv + d_gp

        d_opt.zero_grad()
        d_adv_real.backward(retain_graph=True)
        d_gp.backward()
        d_adv_fake.backward()
        d_opt.step()

        loss_dict.update({f'{log_prefix}/d_adv': d_adv})
        loss_dict.update({f'{log_prefix}/d_gp': d_gp / self.w_gp})
        loss_dict.update({f'{log_prefix}/d_loss': d_loss})
        return loss_dict

    def trainint_step_for_CG(self, conditions, loss_dict, log_prefix, g_opt, c_opt):
        x_org, x_ref, y_org, y_ref = conditions['x_org'], conditions['x_ref'], conditions['y_org'], conditions['y_ref']
        s_src = self.network_C.moco(x_org)
        s_ref = self.network_C.moco(x_ref)
        c_src, skip1, skip2 = self.network_G.cnt_encoder(x_org)
        x_fake, offset_loss = self.network_G.decode(c_src, s_ref, skip1, skip2)
        x_rec, _ = self.network_G.decode(c_src, s_src, skip1, skip2)
        g_fake_logit, _ = self.network_D(x_fake, y_ref)
        g_rec_logit, _ = self.network_D(x_rec, y_org)
        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        g_adv_rec = calc_adv_loss(g_rec_logit, 'g')
        g_adv = g_adv_fake + g_adv_rec
        g_img_rec = calc_recon_loss(x_rec, x_org)
        c_x_fake, _, _ = self.network_G.cnt_encoder(x_fake)
        g_con_rec = calc_recon_loss(c_x_fake, c_src)
        g_loss = self.w_adv * g_adv + self.w_rec * g_img_rec + self.w_rec * g_con_rec + self.w_off * offset_loss

        g_opt.zero_grad()
        c_opt.zero_grad()
        g_loss.backward()
        c_opt.step()
        g_opt.step()

        loss_dict.update({f'{log_prefix}/g_adv': g_adv})
        loss_dict.update({f'{log_prefix}/g_img_rec': g_img_rec})
        loss_dict.update({f'{log_prefix}/g_con_rec': g_con_rec})
        loss_dict.update({f'{log_prefix}/offset_loss': offset_loss})
        loss_dict.update({f'{log_prefix}/g_loss': g_loss})
        return loss_dict

    def training_step(self, batch, batch_idx):
        conditions = self.get_input(batch)
        opt_d, opt_g, opt_c = self.optimizers()

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'

        loss_dict = self.trainint_step_for_D(conditions=conditions, loss_dict=loss_dict, log_prefix=log_prefix,
                                             d_opt=opt_d)
        loss_dict = self.trainint_step_for_CG(conditions=conditions, loss_dict=loss_dict, log_prefix=log_prefix,
                                              c_opt=opt_c, g_opt=opt_g)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False,
                 batch_size=self.batch_size)

    def get_test_input(self, batch, N=None, *args, **kwargs):
        source_imgs = self.get_base_input(batch, self.input_key_source, N)
        reference_imgs = self.get_base_input(batch, self.input_key_reference, N)
        target_imgs = self.get_base_input(batch, self.input_key_target, N)
        conditions = dict(source_imgs=source_imgs, reference_imgs=reference_imgs, target_imgs=target_imgs)
        return conditions

    @torch.no_grad()
    def log_images(self, batch, N=4, **kwargs):
        log = dict()
        conditions = self.get_test_input(batch, N)

        source_imgs = conditions['source_imgs']
        reference_imgs = conditions['reference_imgs']
        target_imgs = conditions['target_imgs']
        fake_imgs, _ = self.generate(source=source_imgs, reference=reference_imgs)
        log['00_source_imgs'] = source_imgs
        log['01_reference_imgs'] = reference_imgs
        log['02_fake_imgs'] = fake_imgs
        log['03_target_imgs'] = target_imgs
        return log

    def on_test_epoch_start(self) -> None:
        print('luxb debug...', 'on_test_epoch_start')
        os.makedirs(self.test_config.result_dir, exist_ok=True)
        self.eval()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        style_names = set(batch[self.test_config.style_name_key])
        assert len(style_names) == 1, 'please reset test_batch_size'
        style_name = list(style_names)[0]
        log = self.log_images(batch=batch, N=self.dataloader_config.test_batch_size)
        keys = list(log.keys())
        keys.sort()
        concat_sample = None
        for key in keys:
            crt_img = log[key]
            if concat_sample is None:
                concat_sample = crt_img
            else:
                concat_sample = torch.cat((concat_sample, crt_img), 0)
        save_image(concat_sample.cpu().detach(), nrow=self.dataloader_config.test_batch_size,
                   fp=os.path.join(self.test_config.result_dir, '%s.png' % (style_name)))

