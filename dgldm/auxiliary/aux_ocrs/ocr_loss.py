import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
import numpy as np
from easydict import EasyDict as edict
import glob
from ..aux_ocrs.ocr_recog.RecModel import RecModel
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])


def create_predictor(model_dir=None, model_lang='ch', is_onnx=False):
    model_file_path = model_dir
    if model_file_path is None:
        ocr_weight_dir = os.path.join(os.path.dirname(__file__), 'ocr_weights')
        model_file_path = os.path.join(ocr_weight_dir, 'ppv3_rec.pth')

    if model_file_path is not None and not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(model_file_path))

    if is_onnx:
        import onnxruntime as ort
        # 'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'
        sess = ort.InferenceSession(model_file_path, providers=['CPUExecutionProvider'])
        return sess
    else:
        if model_lang == 'ch':
            n_class = 6625
        elif model_lang == 'en':
            n_class = 97
        else:
            raise ValueError(f"Unsupported OCR recog model_lang: {model_lang}")
        rec_config = edict(
            in_channels=3,
            backbone=edict(type='MobileNetV1Enhance', scale=0.5, last_conv_stride=[1, 2], last_pool_type='avg'),
            neck=edict(type='SequenceEncoder', encoder_type="svtr", dims=64, depth=2, hidden_dims=120, use_guide=True),
            head=edict(type='CTCHead', fc_decay=0.00001, out_channels=n_class, return_feats=True)
        )

        rec_model = RecModel(rec_config)
        if model_file_path is not None:
            rec_model.load_state_dict(torch.load(model_file_path, map_location="cpu", weights_only=True))
            rec_model.eval()
        return rec_model.eval()


class TextRecognizer(object):
    def __init__(self, rec_image_shape, predictor=None, model_lang='ch'):
        if predictor is None:
            self.predictor = create_predictor(model_lang=model_lang)
        else:
            self.predictor = predictor
        self.rec_image_shape = [int(v) for v in rec_image_shape.split(",")]
        self.ctc_loss = torch.nn.CTCLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def pred_tensor(self, img_inputs, show_debug=False):
        if show_debug:
            resized_img = self.resize_img(img_inputs)
            preds = self.predictor(resized_img)
            saved_dir = os.path.join(os.path.dirname(__file__), 'debug')
            os.makedirs(saved_dir, exist_ok=True)
            origin_paths = glob.glob('%s/origin*.png' %(saved_dir))
            save_image(img_inputs, os.path.join(saved_dir, 'origin_%d.png' %(len(origin_paths))))
            save_image(resized_img, os.path.join(saved_dir, 'resized_%d.png' % (len(origin_paths))))
        else:
            preds = self.predictor(self.resize_img(img_inputs))
        return preds['ctc'], preds['ctc_neck']

    # preds: preds['ctc']
    def get_ctcloss(self, preds, targets, target_lengths, weight=1.0):
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight).to(preds.device)
        # NTC-->TNC
        log_probs = preds.log_softmax(dim=2).permute(1, 0, 2)
        input_lengths = torch.tensor([log_probs.shape[0]]*(log_probs.shape[1])).to(preds.device)
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss = loss / input_lengths * weight
        return loss

    def get_ctc_neck_loss(self, preds, targets, weight=1.0):
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight).to(preds.device)
        loss = self.mse_loss(input=preds, target=targets)
        loss = torch.mean(loss, dim=(1, 2)) * weight
        return loss


    def resize_img(self, img):
        B, _, _, _ = img.shape
        C, H, W = self.rec_image_shape
        normal_H_W = min(H, W)
        resized_image = torch.nn.functional.interpolate(img, size=(normal_H_W, normal_H_W), mode='bilinear',
                                                        align_corners=True, )
        padding_im = torch.zeros((B, C, H, W), dtype=torch.float32).to(img.device)
        padding_im[:, :, :, 0:normal_H_W] = resized_image
        return padding_im


def get_inputs(src_dir, get_gray=True):
    img_paths = glob.glob('%s/*.png' %os.path.join(src_dir))
    img_inputs = []
    input_chars = []
    for img_path in img_paths:
        basename = os.path.basename(img_path)
        char = basename[0]
        input_chars.append(char)
        if get_gray:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')
        img = img.resize((128, 128))
        img_tensor = img_transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_inputs.append(img_tensor)
    img_inputs = torch.cat((img_inputs), 0)
    return img_paths, img_inputs, input_chars
