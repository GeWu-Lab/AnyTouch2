import os
import torch
from transformers import AutoConfig
from model.linear_probe import TactileProbeVideo
from config_probe import parse_args
import random
import numpy as np

from data.downstream_dataset import *

from probe_touchd_engine import train_one_epoch, evaluate


import datetime
import json
import time
import copy

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def load_model_from_multi_clip(ckpt, model):

    new_ckpt = {}
    for key,item in ckpt.items():
        # print(key)
        if "touch_mae_model" in key and 'decoder' not in key and 'mask_token' not in key:
            new_ckpt[key.replace('touch_mae_model.','tactile_model.')] = copy.deepcopy(item)
    
    for k,v in model.named_parameters():
        if k not in new_ckpt.keys():
            new_ckpt[k] = v
    
    model.load_state_dict(new_ckpt, strict=True)

    return model

def random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.dataset == 'ours_force':
        dataset_train = MyForceDataset_video(args, mode = 'train')
        dataset_val = MyForceDataset_video(args, mode = 'test')

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )


    if global_rank == 0 and args.log_dir is not None:
        # os.makedirs(args.log_dir + '-' + args.data_sensor, exist_ok=True)
        added_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.log_dir = args.log_dir + '-' + args.data_sensor + '-' + added_time
        args.output_dir = args.log_dir
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = None
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    if args.model_size == 'base':
        config = AutoConfig.from_pretrained('/home/yaoguocai/code-frx/CLIP-B-16/config.json')
    else:
        raise NotImplementedError
    print(args)

    if args.model == 'anytouch':
        model = TactileProbeVideo(args, config, args.num_frames, 1)

        load_dir = args.load_path
        ckpt = torch.load(load_dir, map_location='cpu')
        model = load_model_from_multi_clip(ckpt, model)
        
        print(load_dir)
    
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(model_without_ddp.head.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args, 0)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['rmse']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = 10000.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        test_stats = evaluate(data_loader_val, model, device, args, epoch)
        # if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
        if test_stats["rmse"] <= min_loss:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=0)
        
        print(f"RMSE of the network on the {len(dataset_val)} test images: {test_stats['rmse']:.2f}")
        min_loss = min(min_loss, test_stats["rmse"])
        print(f'Min RMSE: {min_loss:.2f}')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_rmse', test_stats['rmse'], epoch)
            # log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    

if __name__ == "__main__":
    args = parse_args()
    args = args.parse_args()
    main(args)