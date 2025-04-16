import os
import torch
import numpy as np
from easydict import EasyDict as edict
from aux_ocrs.ocr_loss import get_inputs, create_predictor, TextRecognizer
from aux_datasets.base_datasets import get_train_dataloder


def decode(mat):
    text_index = mat.detach().cpu().numpy().argmax(axis=1)
    ignored_tokens = [0]
    selection = np.ones(len(text_index), dtype=bool)
    selection[1:] = text_index[1:] != text_index[:-1]
    for ignored_token in ignored_tokens:
        selection &= text_index != ignored_token
    return text_index[selection], np.where(selection)[0]


def get_text(order, chars):
    char_list = [chars[text_id] for text_id in order]
    return ''.join(char_list)


def main():
    model_dir = None
    show_debug = True
    batch_size = 16
    rec_image_shape = "3, 48, 320"
    ctc_loss_weight = 1.0
    ctc_neck_loss_weight = 1.0
    predictor = create_predictor(model_dir=model_dir)
    ocr = TextRecognizer(rec_image_shape=rec_image_shape, predictor=predictor)
    dataloader, all_rec_chars = get_train_dataloder(batch_size=batch_size)
    for data in dataloader:
        src_char = data['src_char']
        ref_char = data['ref_char']
        std_src_img = data['std_src_img']
        std_ref_img = data['std_ref_img']
        sty_ref_img = data['sty_ref_img']
        src_char_id = data['src_char_id']
        ref_char_id = data['ref_char_id']
        src_char_len = data['src_char_len']
        ref_char_len = data['ref_char_len']
        pred_std_src_ctc, pred_std_src_ctc_neck = ocr.pred_tensor(std_src_img, show_debug=show_debug)
        pred_std_ref_ctc, pred_std_ref_ctc_neck = ocr.pred_tensor(std_ref_img, show_debug=show_debug)
        pred_sty_ref_ctc, pred_sty_ref_ctc_neck = ocr.pred_tensor(sty_ref_img, show_debug=show_debug)
        std_src_ctc_loss = ocr.get_ctcloss(preds=pred_std_src_ctc, targets=src_char_id, target_lengths=src_char_len,
                                           weight=ctc_loss_weight)
        std_ref_ctc_loss = ocr.get_ctcloss(preds=pred_std_ref_ctc, targets=ref_char_id, target_lengths=ref_char_len,
                                           weight=ctc_loss_weight)
        sty_ref_ctc_loss = ocr.get_ctcloss(preds=pred_sty_ref_ctc, targets=ref_char_id, target_lengths=ref_char_len,
                                           weight=ctc_loss_weight)
        std_ref_sty_ref_ctc_neck_loss = ocr.get_ctc_neck_loss(preds=pred_sty_ref_ctc_neck, targets=pred_std_ref_ctc_neck,
                                                              weight=ctc_neck_loss_weight)
        print(
            'std_src_ctc_loss=%05f, std_ref_ctc_loss=%05f, sty_ref_ctc_loss=%05f, std_ref_sty_ref_ctc_neck_loss=%05f' % (
            std_src_ctc_loss.mean(), std_ref_ctc_loss.mean(), sty_ref_ctc_loss.mean(),
            std_ref_sty_ref_ctc_neck_loss.mean()))
        pred_std_src_ctc_softmax = pred_std_src_ctc.softmax(dim=2)
        pred_std_ref_ctc_softmax = pred_std_ref_ctc.softmax(dim=2)
        pred_sty_ref_ctc_softmax = pred_sty_ref_ctc.softmax(dim=2)

        for i in range(len(src_char)):
            pred_std = pred_std_ref_ctc_softmax[i]
            pred_sty = pred_sty_ref_ctc_softmax[i]
            order_std, _ = decode(pred_std)
            order_sty, _ = decode(pred_sty)
            pred_std_char = get_text(order=order_std, chars=all_rec_chars)
            pred_sty_char = get_text(order=order_sty, chars=all_rec_chars)
            print('char:%s, pred_std:%s, pred_sty:%s, ctc_neck_loss:%04f, std_ctc_loss:%04f, sty_ctc_loss:%04f,' % (
                                    ref_char[i], pred_std_char, pred_sty_char, std_ref_sty_ref_ctc_neck_loss[i],
                                    std_ref_ctc_loss[i], sty_ref_ctc_loss[i]))

            # print('char:%s, pred_std:%s, pred_sty:%s, ctc_neck_loss:%04f' %(ref_char[i], pred_std_char, pred_sty_char, std_ref_sty_ref_ctc_neck_loss[i]))
        # for i in range(len(src_char)):
        #     pred = pred_std_src_ctc_softmax[i]
        #     order, idx = decode(pred)
        #     text = get_text(order=order, chars=all_rec_chars)
        #     print('pred_std char: %s, pred:%s, gt:%s, loss:%04f' % (src_char[i], text, src_char[i], std_src_ctc_loss[i]))
        #
        #
        # for i in range(len(ref_char)):
        #     pred = pred_sty_ref_ctc_softmax[i]
        #     order, idx = decode(pred)
        #     text = get_text(order=order, chars=all_rec_chars)
        #     print(
        #         'pred_sty char: %s, pred:%s, gt:%s, loss:%04f' % (ref_char[i], text, ref_char[i], std_src_ctc_loss[i]))
        break


if __name__ == "__main__":
    main()

