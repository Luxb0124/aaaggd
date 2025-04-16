import torch
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin


class FontDiffuserModel(ModelMixin, ConfigMixin):
    def __init__(self, unet, style_encoder, content_encoder):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    def forward(self, x_t, timesteps, style_images, content_images, content_encoder_downsample_size,):
        style_img_feature, _, _ = self.style_encoder(style_images)

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height * width, channel).contiguous()

        # Get the content feature
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)

        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, style_hidden_states, style_content_res_features]

        out = self.unet(x_t, timesteps, encoder_hidden_states=input_hidden_states,
                        content_encoder_downsample_size=content_encoder_downsample_size,)
        noise_pred = out[0]
        offset_out_sum = out[1]
        return noise_pred, offset_out_sum

    @torch.no_grad()
    def generate(self, x_t, timesteps, cond, content_encoder_downsample_size, version=None):
        content_images = cond[0]
        style_images = cond[1]

        style_img_feature, _, style_residual_features = self.style_encoder(style_images)

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height * width, channel).contiguous()

        # Get content feature
        content_img_feture, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feture)

        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, style_hidden_states,
                               style_content_res_features]
        out = self.unet(x_t, timesteps, encoder_hidden_states=input_hidden_states,
                        content_encoder_downsample_size=content_encoder_downsample_size)
        noise_pred = out[0]
        return noise_pred
