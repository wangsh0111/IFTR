# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim
from mmcv import Config
from mmcv.runner.checkpoint import load_checkpoint
from mmdet3d.models import build_model


def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    if file_list:
        assert len(file_list) == 1
        print("resuming best validation model at epoch %d" %
              eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
        model.load_state_dict(torch.load(file_list[0], map_location='cpu'), strict=False)
        return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(
            os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch), map_location='cpu'), strict=False)

    return initial_epoch, model


def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """

    config = hypes['model']['config']
    cfg = Config.fromfile(config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    instance = build_model(cfg.model)
    instance.init_weights()

    return instance


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or not hasattr(inputs, 'to'):
            return inputs
        return inputs.to(device, non_blocking=True)


def logging(lr, loss_dict, total_loss, epoch, batch_id, batch_len, writer=None, suffix=""):
    """
    Print out  the loss function for current iteration.

    Parameters
    ----------
    epoch : int
        Current epoch for training.
    batch_id : int
        The current batch.
    batch_len : int
        Total batch length in one iteration of training,
    writer : SummaryWriter
        Used to visualize on tensorboard
    """
    loss_dict['reg_loss'] = 0
    loss_dict['cls_loss'] = 0
    loss_dict['depth_loss'] = 0

    for key, value in loss_dict.items():
        if 'loss_cls' in key:
            loss_dict['cls_loss'] += value
        elif 'loss_bbox' in key:
            loss_dict['reg_loss'] += value

    reg_loss = loss_dict.get('reg_loss', 0)
    cls_loss = loss_dict.get('cls_loss', 0)
    depth_loss = loss_dict.get('loss_dense_depth', 0)

    print("[epoch %d][%d/%d]%s || || Lr: %.8f || Loss: %.4f || Conf Loss: %.4f "
          "|| Loc Loss: %.4f || Depth Loss: %.4f" % (
              epoch, batch_id + 1, batch_len, suffix, lr,
              total_loss, cls_loss, reg_loss, depth_loss))

    if not writer is None:
        writer.add_scalar('Regression_loss' + suffix, reg_loss,
                          epoch*batch_len + batch_id)
        writer.add_scalar('Confidence_loss' + suffix, cls_loss,
                          epoch * batch_len + batch_id)
        writer.add_scalar('Depth_loss' + suffix, depth_loss,
                          epoch * batch_len + batch_id)
