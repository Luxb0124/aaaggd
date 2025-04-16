import torch
from PIL import Image
from torchvision.utils import make_grid
from aux_models.guidingNet import GuidingNet
from aux_models.generator import Generator
from aux_datasets.base_datasets import get_train_dataloder


def show_imgs(imgs, nrow=8):
    imgs = torch.cat(imgs, 0)
    imgs = make_grid(imgs, nrow=nrow)
    imgs = (imgs + 1) / 2
    imgs = imgs.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    imgs = Image.fromarray(imgs)
    imgs.show()


def get_models(sty_dim=128, output_k=400, n_res=2, device='cuda:0'):
    network_C = GuidingNet(sty_dim, output_k, load_path='default', load_key='C_EMA_state_dict').to(device)
    network_G = Generator(sty_dim, n_res, use_sn=False, load_path='default', load_key='G_EMA_state_dict').to(device)
    return network_C, network_G


def get_data(batch_size=8, device='cuda:0'):
    dataloader, all_rec_chars = get_train_dataloder(batch_size=batch_size)
    for data in dataloader:
        content_img = data['std_src_img']
        reference_img = data['sty_ref_img']
        ground_truth_img = data['sty_src_img']
        break
    return content_img.to(device), reference_img.to(device), ground_truth_img.to(device)


def generate_imgs(network_C, network_G, content_img, reference_img):
    feature_sty_ref = network_C.moco(reference_img)
    # [8, 128]
    print('feature_sty_ref', feature_sty_ref.shape)
    feature_cnt_src, feature_cnt_skip1, feature_cnt_skip2 = network_G.cnt_encoder(content_img)
    # [8, 256, 24, 24]
    print('feature_cnt_src', feature_cnt_src.shape)
    # [8, 64, 96, 96]
    print('feature_cnt_skip1', feature_cnt_skip1.shape)
    # [8, 128, 48, 48]
    print('feature_cnt_skip2', feature_cnt_skip2.shape)
    generated_img, _ = network_G.decode(feature_cnt_src, feature_sty_ref, feature_cnt_skip1, feature_cnt_skip2)
    return generated_img


if __name__ == '__main__':
    batch_size = 8
    network_C, network_G = get_models()
    content_img, reference_img, ground_truth_img = get_data(batch_size)
    generated_img = generate_imgs(network_C, network_G, content_img, reference_img)
    imgs_lst = [content_img, reference_img, generated_img, ground_truth_img]
    show_imgs(imgs_lst, nrow=batch_size)
