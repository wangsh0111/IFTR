import argparse
import os
import sys
import statistics
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../mmdet3d'))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils

# 出现任意Nan都会引起模型报错, 且会使训练速度大大减慢
# torch.autograd.set_detect_anomaly(True)


def warm_up_cosine_lr_scheduler(optimizer, max_iters=100000, warmup_iters=500, eta_min=1e-9):
    if warmup_iters <= 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=eta_min)

    else:
        lr_func = lambda step: eta_min + (step / warmup_iters) if step <= warmup_iters \
            else eta_min + 0.5 * (np.cos((step - warmup_iters) / (max_iters - warmup_iters) * np.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)

    print('--------------------Dataset Building--------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset, batch_sampler=batch_sampler_train, num_workers=4,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset, sampler=sampler_val, num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_train, drop_last=True)
    else:
        train_loader = DataLoader(opencood_train_dataset, batch_size=hypes['train_params']['batch_size'],
                                  num_workers=4, collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True, pin_memory=True, drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset, batch_size=hypes['train_params']['batch_size'],
                                num_workers=4, collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True, pin_memory=True, drop_last=True)

    print('--------------------Creating Model--------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        model.to(device)

    # record lowest validation loss checkpoint.
    epoches = hypes['train_params']['epoches']
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # ddp setting
    model_without_ddp = model
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    img_backbone_params_list = list(map(id, model_without_ddp.img_backbone.parameters()))
    base_params = filter(lambda p: id(p) not in img_backbone_params_list, model_without_ddp.parameters())
    optimizer = torch.optim.AdamW([
        {'params': base_params},
        {'params': model_without_ddp.img_backbone.parameters(), 'lr': 1e-4 * 0.5}
    ], lr=1e-4, weight_decay=0.001)
    # grad clip config
    grad_clip_config = {'max_norm': 25, 'norm_type': 2}

    scheduler = warm_up_cosine_lr_scheduler(
        optimizer=optimizer, max_iters=epoches * len(train_loader), warmup_iters=500, eta_min=1e-3)
    if init_epoch > 0:
        for _ in range(init_epoch * len(train_loader)):
            scheduler.step()
        print(f"model is resume from {init_epoch} epoch.")

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('--------------------Training start--------------------')
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if opt.distributed:
            sampler_train.set_epoch(epoch)

        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum() == 0:
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch

            lr = optimizer.state_dict()['param_groups'][0]['lr']
            if not opt.half:
                losses_dict, final_loss = model(ego=batch_data['ego'])
                train_utils.logging(lr, losses_dict, final_loss, epoch, i, len(train_loader), writer)
            else:
                with torch.cuda.amp.autocast():
                    losses_dict, final_loss = model(ego=batch_data['ego'])
                    train_utils.logging(lr, losses_dict, final_loss, epoch, i, len(train_loader), writer)

            if not opt.half:
                # back-propagation & grad clip
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_without_ddp.parameters(), **grad_clip_config)

                # update parameters
                optimizer.step()
                scheduler.step()

            else:
                # back-propagation & grad clip
                scaler.scale(final_loss).backward()
                torch.nn.utils.clip_grad_norm_(model_without_ddp.parameters(), **grad_clip_config)

                # update parameters
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

        if epoch % hypes['train_params']['eval_freq'] == 0 or epoch == max(epoches, init_epoch) - 1:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    print(f"[epoch {epoch}][{i}/{len(val_loader)}][val]")
                    if batch_data is None or batch_data['ego']['object_bbx_mask'].sum() == 0:
                        continue
                    model.module.zero_grad()
                    optimizer.zero_grad()
                    model.module.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch

                    losses_dict, final_loss = model.module(ego=batch_data['ego'])
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch, valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model_without_ddp.state_dict(), os.path.join(
                    saved_path, 'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(
                        os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    if opt.rank == 0:
                        os.remove(os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0 or epoch == max(epoches, init_epoch) - 1:
            torch.save(model_without_ddp.state_dict(), os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
