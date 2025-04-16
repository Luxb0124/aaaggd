import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange
from torchvision.utils import save_image
from diffusers.optimization import get_scheduler
import sys
sys.path.append(os.path.dirname(__file__))
from utils.util import instantiate_from_config, x0_from_epsilon, reNormalize_img, normalize_mean_std
from df.df_models.font_diffuser import FontDiffuserModel
from df.df_modules.criterion import ContentPerceptualLoss
from df.df_dpm_solver.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from auxiliary.aux_ocrs.ocr_loss import create_predictor, TextRecognizer


# DGFont + FontDiffuser
class Baseline(pl.LightningModule):
    def __init__(self, aux_C_config, aux_G_config,  unet_config, style_config, content_config,
                 noise_config, envs_config=None, logger_config=None, sampler_config=None,
                 dataset_config=None, dataloader_config=None, optimizer_config=None,
                 test_dataset_config=None, test_config=None, ocr_config=None,
                 input_std_src_img='std_src_img', input_std_ref_img='std_ref_img', input_sty_src_img='sty_src_img',
                 input_sty_ref_img='sty_ref_img', input_src_char='src_char',
                 input_ref_char='ref_char', input_src_char_id='src_char_id',
                 input_ref_char_id='ref_char_id', input_src_char_len='src_char_len',
                 input_ref_char_len='ref_char_len',
                 drop_prob=0.1, perceptual_coefficient=0.01, offset_coefficient=0.5,
                 diffuser_coefficient=1.0, aux_coefficient=1.0, aux_offset_coefficient=0.5,
                 aux_ctc_neck_coefficient=0.01, aux_ctc_coefficient=0.001, char_weight=1.0,
                 show_debug=False, eps=1e-6,
                 ):
        super().__init__()
        assert diffuser_coefficient > 0
        unet = instantiate_from_config(unet_config)
        self.aux_network_C = instantiate_from_config(aux_C_config)
        self.aux_network_G = instantiate_from_config(aux_G_config)
        assert ocr_config is not None
        self.text_predictor = create_predictor(model_lang=ocr_config.model_lang).eval()
        self.ocr = TextRecognizer(rec_image_shape=ocr_config.rec_image_shape, predictor=self.text_predictor)
        style_encoder = instantiate_from_config(style_config)
        content_encoder = instantiate_from_config(content_config)
        self.model = FontDiffuserModel(unet=unet, style_encoder=style_encoder, content_encoder=content_encoder)

        self.envs_config = envs_config
        self.logger_config = logger_config
        self.sampler_config = sampler_config
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        self.optimizer_config = optimizer_config
        self.test_dataset_config = test_dataset_config
        self.test_config = test_config

        self.input_std_src_img = input_std_src_img
        self.input_std_ref_img = input_std_ref_img
        self.input_sty_src_img = input_sty_src_img
        self.input_sty_ref_img = input_sty_ref_img
        self.input_src_char = input_src_char
        self.input_ref_char = input_ref_char
        self.input_src_char_id = input_src_char_id
        self.input_ref_char_id = input_ref_char_id
        self.input_src_char_len = input_src_char_len
        self.input_ref_char_len = input_ref_char_len

        self.perceptual_coefficient = perceptual_coefficient
        self.offset_coefficient = offset_coefficient
        self.diffuser_coefficient = diffuser_coefficient
        self.aux_coefficient = aux_coefficient
        self.aux_offset_coefficient = aux_offset_coefficient
        self.aux_ctc_neck_coefficient = aux_ctc_neck_coefficient
        self.aux_ctc_coefficient = aux_ctc_coefficient

        self.char_weight = char_weight
        self.drop_prob = drop_prob
        self.content_encoder_downsample_size = unet_config.params.content_encoder_downsample_size
        self.show_debug = show_debug
        self.eps = eps

        self.noise_scheduler = instantiate_from_config(noise_config)
        self.sample_noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.noise_scheduler.betas)
        self.perceptual_loss = ContentPerceptualLoss()
        if self.optimizer_config.lr_scheduler is None:
            self.use_scheduler = False
        else:
            self.use_scheduler = True
        self.generator = torch.manual_seed(self.sampler_config.seed)
        self.batch_size = self.dataloader_config.train_batch_size

    def configure_optimizers(self):
        # for name, _ in self.model.named_parameters():
        #     print('parameter name', name)
        params = list(self.model.parameters())
        params += list(self.aux_network_C.parameters())
        params += list(self.aux_network_G.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.optimizer_config.learning_rate,
            betas=(self.optimizer_config.adam_beta1, self.optimizer_config.adam_beta2),
            weight_decay=self.optimizer_config.adam_weight_decay,
            eps=self.optimizer_config.adam_epsilon)
        if self.use_scheduler:
            lr_scheduler = get_scheduler(
                self.optimizer_config.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.optimizer_config.lr_warmup_steps * self.optimizer_config.gradient_accumulation_steps,
                num_training_steps=self.optimizer_config.max_train_steps * self.optimizer_config.gradient_accumulation_steps, )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", 'frequency': 1}]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False,
                 batch_size=self.batch_size)
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=self.batch_size)
        return loss

    def cal_perceptron_loss(self, timesteps, noise_pred, noisy_target_images, target_images, device):
        if self.perceptual_coefficient > 0:
            # target_images and pred_original_sample_norm range: -1~1
            pred_original_sample_norm = x0_from_epsilon(scheduler=self.noise_scheduler,
                                                        noise_pred=noise_pred, x_t=noisy_target_images,
                                                        timesteps=timesteps)
            # nonorm_target_images and pred_original_sample range: 0~1
            pred_original_sample = reNormalize_img(pred_original_sample_norm)
            nonorm_target_images = reNormalize_img(target_images)
            norm_pred_ori = normalize_mean_std(pred_original_sample)
            norm_target_ori = normalize_mean_std(nonorm_target_images)
            percep_loss = self.perceptual_loss.calculate_loss(generated_images=norm_pred_ori,
                                                              target_images=norm_target_ori,
                                                              device=device)
            percep_loss = self.perceptual_coefficient * percep_loss
        else:
            percep_loss = torch.tensor(0).float().to(device)
        return percep_loss

    def cal_diff_offset_loss(self, timesteps, style_images, content_images, noise, noisy_target_images):
        noise_pred, offset_out_sum = self.model(x_t=noisy_target_images, timesteps=timesteps,
                                                style_images=style_images, content_images=content_images,
                                                content_encoder_downsample_size=self.content_encoder_downsample_size)
        diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        diff_loss = self.diffuser_coefficient * diff_loss

        if self.offset_coefficient > 0:
            offset_loss = offset_out_sum / 2
            offset_loss = self.offset_coefficient * offset_loss
        else:
            offset_loss = torch.tensor(0).float().to(style_images.device)
        return diff_loss, offset_loss, noise_pred

    def cal_ocr_loss(self, pred_sty_images, std_images, char_ids, char_lens, char_weight=None):
        if char_weight is None:
            char_weight = self.char_weight
        if self.aux_coefficient > 0:
            sty_ctc, sty_ctc_neck = self.ocr.pred_tensor(pred_sty_images, show_debug=self.show_debug)
            if self.aux_ctc_coefficient > 0:
                ctc_loss = self.ocr.get_ctcloss(preds=sty_ctc, targets=char_ids, target_lengths=char_lens, weight=char_weight)
                ctc_loss = self.aux_ctc_coefficient * ctc_loss.mean()
            else:
                ctc_loss = torch.tensor(0).float().to(pred_sty_images.device)
            if self.aux_ctc_neck_coefficient:
                std_ctc, std_ctc_neck = self.ocr.pred_tensor(std_images, show_debug=self.show_debug)
                ctc_neck_loss = self.ocr.get_ctc_neck_loss(preds=sty_ctc_neck, targets=std_ctc_neck, weight=char_weight)
                ctc_neck_loss = self.aux_ctc_neck_coefficient * ctc_neck_loss.mean()
            else:
                ctc_neck_loss = torch.tensor(0).float().to(pred_sty_images.device)
            aux_loss = ctc_loss + ctc_neck_loss
        else:
            aux_loss = torch.tensor(0).float().to(pred_sty_images.device)
        return aux_loss

    def generate_aux_images(self, src_imgs, ref_imgs):
        feature_sty_ref = self.aux_network_C.moco(ref_imgs)
        feature_cnt_src, feature_cnt_skip1, feature_cnt_skip2 = self.aux_network_G.cnt_encoder(src_imgs)
        pred_sty_src, offset_loss = self.aux_network_G.decode(feature_cnt_src, feature_sty_ref, feature_cnt_skip1, feature_cnt_skip2)
        aux_offset_loss = self.aux_coefficient * self.aux_offset_coefficient * offset_loss
        return pred_sty_src, aux_offset_loss

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

        # auxiliary: generate the fake style images [pred_sty_src_imgs]
        pred_sty_src_imgs, aux_offset_loss = self.generate_aux_images(src_imgs=std_src_imgs, ref_imgs=sty_ref_imgs)

        # diffuser: std_ref_imgs + pred_sty_src_imgs = pred_sty_ref_imgs
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
                pred_sty_src_imgs[i, :, :, :] = 1

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        # Predict the noise residual and compute loss
        diff_loss, offset_loss, noise_pred = self.cal_diff_offset_loss(timesteps=timesteps,
                                                                       style_images=pred_sty_src_imgs,
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
        # auxiliary offset loss
        loss_dict.update({f'{log_prefix}/aux_offset_loss': aux_offset_loss / (
                    self.aux_offset_coefficient + self.eps) / (self.aux_coefficient + self.eps)})
        loss = loss + aux_offset_loss
        # auxiliary ocr loss
        aux_ocr_loss = self.cal_ocr_loss(pred_sty_images=pred_sty_src_imgs, std_images=std_src_imgs, char_ids=src_char_ids,
                                     char_lens=src_char_lens)
        loss_dict.update({f'{log_prefix}/aux_ocr_loss': aux_ocr_loss / (
                self.aux_offset_coefficient + self.eps) / (self.aux_coefficient + self.eps)})
        loss = loss + aux_ocr_loss
        return loss, loss_dict

    def get_base_input(self, batch, k, N):
        x = batch[k][:N]
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 3:
                x = x[..., None]
                x = rearrange(x, 'b h w c -> b c h w').contiguous()
            x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_input(self, batch, N=None, *args, **kwargs):
        std_src_imgs = self.get_base_input(batch, self.input_std_src_img, N)
        std_ref_imgs = self.get_base_input(batch, self.input_std_ref_img, N)
        sty_src_imgs = self.get_base_input(batch, self.input_sty_src_img, N)
        sty_ref_imgs = self.get_base_input(batch, self.input_sty_ref_img, N)
        src_chars = self.get_base_input(batch, self.input_src_char, N)
        ref_chars = self.get_base_input(batch, self.input_ref_char, N)
        src_char_ids = self.get_base_input(batch, self.input_src_char_id, N)
        ref_char_ids = self.get_base_input(batch, self.input_ref_char_id, N)
        src_char_lens = batch[self.input_src_char_len][:N]
        ref_char_lens = batch[self.input_ref_char_len][:N]

        conditions = dict(std_src_imgs=std_src_imgs, std_ref_imgs=std_ref_imgs,
                          sty_src_imgs=sty_src_imgs, sty_ref_imgs=sty_ref_imgs,
                          src_chars=src_chars, ref_chars=ref_chars,
                          src_char_ids=src_char_ids, ref_char_ids=ref_char_ids,
                          src_char_lens=src_char_lens, ref_char_lens=ref_char_lens)
        return conditions

    def get_test_input(self, batch, N=None, *args, **kwargs):
        std_src_imgs = self.get_base_input(batch, self.input_std_src_img, N)
        std_ref_imgs = self.get_base_input(batch, self.input_std_ref_img, N)
        sty_src_imgs = self.get_base_input(batch, self.input_sty_src_img, N)
        sty_ref_imgs = self.get_base_input(batch, self.input_sty_ref_img, N)
        conditions = dict(std_src_imgs=std_src_imgs, std_ref_imgs=std_ref_imgs,
                          sty_src_imgs=sty_src_imgs,sty_ref_imgs=sty_ref_imgs)
        return conditions

    def forward(self, conditions, *args, **kwargs):
        return self.p_losses(conditions, *args, **kwargs)

    def shared_step(self, batch, *args, **kwargs):
        conditions= self.get_input(batch)
        loss, loss_dict = self(conditions, *args, **kwargs)
        return loss, loss_dict

    @torch.no_grad()
    def sample(self, content_images, style_images, N):
        model_kwargs = {}
        model_kwargs["version"] = self.sampler_config.version
        model_kwargs["content_encoder_downsample_size"] = self.content_encoder_downsample_size
        n = content_images.shape[0]
        if N is None:
            N = n
        else:
            N = min(n, N)

        cond = []
        cond.append(content_images)
        cond.append(style_images)

        uncond = []
        uncond_content_images = torch.ones_like(content_images).to(self.model.device)
        uncond_style_images = torch.ones_like(style_images).to(self.model.device)
        uncond.append(uncond_content_images)
        uncond.append(uncond_style_images)

        # 2.Convert the discrete-time model to the continuous-time
        model_fn = model_wrapper(
            model=self.model,
            noise_schedule=self.sample_noise_schedule,
            model_type=self.sampler_config.model_type,
            model_kwargs=model_kwargs,
            guidance_type=self.sampler_config.guidance_type,
            condition=cond,
            unconditional_condition=uncond,
            guidance_scale=self.sampler_config.guidance_scale
        )

        # 3. Define dpm-solver and sample by multistep DPM-Solver.
        # (We recommend multistep DPM-Solver for conditional sampling)
        # You can adjust the `steps` to balance the computation costs and the sample quality.
        dpm_solver = DPM_Solver(
            model_fn=model_fn,
            noise_schedule=self.sample_noise_schedule,
            algorithm_type=self.sampler_config.algorithm_type,
            correcting_x0_fn=self.sampler_config.correcting_x0_fn
        )
        # If the DPM is defined on pixel-space images, you can further set `correcting_x0_fn="dynamic_thresholding"

        # 4. Generate
        # Sample gaussian noise to begin loop => [batch, 3, height, width]
        x_T = torch.randn((N, 3, self.sampler_config.resolution, self.sampler_config.resolution),
                          generator=self.generator, )
        x_T = x_T.to(self.model.device)

        x_sample = dpm_solver.sample(
            x=x_T,
            steps=self.sampler_config.num_inference_steps,
            order=self.sampler_config.order,
            skip_type=self.sampler_config.skip_type,
            method=self.sampler_config.method,
        )
        return x_sample

    @torch.no_grad()
    def log_images(self, batch, N=4, **kwargs):
        log = dict()
        conditions = self.get_test_input(batch, N)
        # std_src_imgs=std_src_imgs, std_ref_imgs=std_ref_imgs,
        #                           sty_src_imgs=sty_src_imgs,sty_ref_imgs=sty_ref_imgs
        std_src_imgs = conditions['std_src_imgs']
        std_ref_imgs = conditions['std_ref_imgs']
        sty_src_imgs = conditions['sty_src_imgs']
        sty_ref_imgs = conditions['sty_ref_imgs']

        pred_sty_src_imgs, _ = self.generate_aux_images(src_imgs=std_src_imgs, ref_imgs=sty_ref_imgs)
        pred_sty_ref_imgs, _ = self.generate_aux_images(src_imgs=std_ref_imgs, ref_imgs=sty_src_imgs)

        # to speed up...
        # generated_src_all_imgs = self.sample(content_images=std_src_imgs, style_images=pred_sty_ref_imgs, N=N)
        # generated_src_diff_imgs = self.sample(content_images=std_src_imgs, style_images=sty_ref_imgs, N=N)
        # generated_ref_all_imgs = self.sample(content_images=std_ref_imgs, style_images=pred_sty_src_imgs, N=N)
        # generated_ref_diff_imgs = self.sample(content_images=std_ref_imgs, style_images=sty_src_imgs, N=N)
        content_images_cat = torch.cat((std_src_imgs[:N], std_src_imgs[:N], std_ref_imgs[:N], std_ref_imgs[:N]), 0)
        style_images_cat = torch.cat((pred_sty_ref_imgs[:N], sty_ref_imgs[:N], pred_sty_src_imgs[:N], sty_src_imgs[:N]),
                                     0)
        generated_imgs = self.sample(content_images=content_images_cat, style_images=style_images_cat, N=N * 4)
        generated_src_all_imgs, generated_src_diff_imgs, generated_ref_all_imgs, generated_ref_diff_imgs = torch.chunk(
            generated_imgs, chunks=4, dim=0)

        log['00_std_src_imgs'] = std_src_imgs
        log['01_std_ref_imgs'] = std_ref_imgs
        log['02_sty_src_imgs'] = sty_src_imgs
        log['03_sty_ref_imgs'] = sty_ref_imgs

        log['04_fake_sty_src_all_imgs'] = generated_src_all_imgs
        log['05_fake_sty_src_diff_imgs'] = generated_src_diff_imgs
        log['06_fake_sty_ref_all_imgs'] = generated_ref_all_imgs
        log['07_fake_sty_ref_diff_imgs'] = generated_ref_diff_imgs

        log['08_aux_sty_src_imgs'] = pred_sty_src_imgs
        log['09_aux_sty_ref_imgs'] = pred_sty_ref_imgs
        return log

    def on_test_epoch_start(self) -> None:
        print('luxb debug...', 'on_test_epoch_start')
        os.makedirs(self.test_config.result_dir, exist_ok=True)
        self.eval()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # len(set(lst)) <= 1
        # todo
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
