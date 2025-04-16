import os
import torch
import importlib
import torchvision.transforms as transforms
from omegaconf import OmegaConf


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def x0_from_epsilon(scheduler, noise_pred, x_t, timesteps):
    """Return the x_0 from epsilon
    """
    batch_size = noise_pred.shape[0]
    for i in range(batch_size):
        noise_pred_i = noise_pred[i]
        noise_pred_i = noise_pred_i[None, :]
        t = timesteps[i]
        x_t_i = x_t[i]
        x_t_i = x_t_i[None, :]

        pred_original_sample_i = scheduler.step(
            model_output=noise_pred_i,
            timestep=t,
            sample=x_t_i,
            # predict_epsilon=True,
            generator=None,
            return_dict=True,
        ).pred_original_sample
        if i == 0:
            pred_original_sample = pred_original_sample_i
        else:
            pred_original_sample = torch.cat((pred_original_sample, pred_original_sample_i), dim=0)
    return pred_original_sample


def reNormalize_img(pred_original_sample):
    pred_original_sample = (pred_original_sample / 2 + 0.5).clamp(0, 1)
    return pred_original_sample


def normalize_mean_std(image):
    transforms_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transforms_norm(image)
    return image


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def get_state_dict(d):
    return d.get('state_dict', d)
