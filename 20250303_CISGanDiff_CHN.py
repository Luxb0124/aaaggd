import os
import math
import glob
import torch
import datetime
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dgldm.utils.util import create_model, load_state_dict, instantiate_from_config
from dgldm.datasets.collate_fn import CollateFN
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


gpus = '0,1,2,3,4,5'
os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_dataloader(dataset_config, dataloader_config, is_train=True):
    dataset = instantiate_from_config(dataset_config)
    if is_train:
        dataloader = DataLoader(dataset, shuffle=True, batch_size=dataloader_config.train_batch_size,
                                collate_fn=CollateFN(), num_workers=8)
    else:
        dataloader = DataLoader(dataset, shuffle=False, batch_size=dataloader_config.test_batch_size,
                                collate_fn=CollateFN())
    return dataloader


def get_logger(logger_config, model_yaml_path):
    model_config = os.path.basename(model_yaml_path).replace('.yaml', '')
    crt_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.basename(crt_dir)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    base_logger_root = logger_config.base_logger_root
    base_logger_dir = os.path.join(crt_dir, base_logger_root, model_name, model_config, current_time)
    os.makedirs(base_logger_dir, exist_ok=True)
    command = 'cp -rf %s %s' %(model_yaml_path, base_logger_dir)
    os.system(command)
    ckpt_dir = os.path.join(base_logger_dir, 'ckpts')
    base_logger = pl_loggers.TensorBoardLogger(save_dir=base_logger_dir)
    ckpt_logger = ModelCheckpoint(dirpath=ckpt_dir, every_n_train_steps=logger_config.save_ckpt_frequency)
    image_logger = instantiate_from_config(logger_config.image_logger_config)
    return base_logger, ckpt_logger, image_logger


def get_ckpt_path(test_config):
    ckpt_paths = glob.glob('%s/*.ckpt' %(test_config.ckpt_dir))
    ckpt_paths.sort()
    return ckpt_paths[-1]


def train():
    model_yaml_path = 'configs/20250303_CISGanDiff_CHN.yaml'
    # model
    model = create_model(model_yaml_path).cpu()
    # some configs
    dataset_config = model.dataset_config
    dataloader_config = model.dataloader_config
    optimizer_config = model.optimizer_config
    logger_config = model.logger_config
    envs_config = model.envs_config
    assert gpus == envs_config.gpus, f'GPU:{gpus} != {envs_config.gpus}'
    gpus_nums = len(gpus.split(','))
    # dataloader
    dataloader = get_dataloader(dataset_config=dataset_config, dataloader_config=dataloader_config)
    epoches = math.ceil(optimizer_config.max_train_steps / len(dataloader))
    print(optimizer_config.max_train_steps, len(dataloader), gpus_nums, epoches, dataloader_config.train_batch_size)
    # logger
    base_logger, ckpt_logger, image_logger = get_logger(logger_config, model_yaml_path)
    # trainer
    if gpus_nums > 1:
        trainer = pl.Trainer(strategy="ddp", max_steps=optimizer_config.max_train_steps,
                             logger=base_logger,
                             callbacks=[image_logger, ckpt_logger],
                             devices=torch.cuda.device_count(), )

    else:
        trainer = pl.Trainer(max_steps=optimizer_config.max_train_steps, logger=base_logger,
                             callbacks=[image_logger, ckpt_logger])

    # fit
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    train()
