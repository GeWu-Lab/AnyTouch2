import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', default='output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--model', default='anytouch',
                        help='')
    parser.add_argument('--model_size', default='base',
                        help='modelsize')
    parser.add_argument("--no_schedule", action='store_true', help="")
    parser.add_argument('--data_sensor', default='digit',
                        help='')

    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='beta_1')
    parser.add_argument('--beta_2', type=float, default=0.99,
                        help='beta_2')
    parser.add_argument("--use_diff", action='store_true', help="")
    parser.add_argument("--alpha_vl", type=float, default=0.2, help="")
    parser.add_argument("--alpha_vt", type=float, default=0.2, help="")
    parser.add_argument("--alpha_lt", type=float, default=1.0, help="")

    parser.add_argument("--norm_pix_loss", action='store_true', help="")
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument("--num_frames", type=int, default=4, help="")
    parser.add_argument("--stride", type=int, default=2, help="")

    parser.add_argument('--mask_ratio', type=float, default=0, help='')

    parser.add_argument("--cross_attention", action='store_true', help="")

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--num_workers', type=int, default=32, metavar='N',
                        help='')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--use_universal", action='store_true', help="")
    
    parser.add_argument('--dataset', default='material', type=str,
                        help='rough, material')     
    parser.add_argument('--split', default=0, type=int, metavar='N',
                        help='split')           
    parser.add_argument('--pooling', default='global',
                        help='cls, global')
    parser.add_argument("--eval", action='store_true', help="")
    parser.add_argument("--load_from_clip", action='store_true', help="")
    parser.add_argument("--load_from_align", action='store_true', help="")
    parser.add_argument("--load_path", type=str, help="")
    parser.add_argument("--norm", type=str, default='joint', help="")
    parser.add_argument("--policy", type=str, default='dp', help="")


    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )

    # distributed training parameters
    parser.add_argument("--distributed", action='store_true', help="")
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    return parser