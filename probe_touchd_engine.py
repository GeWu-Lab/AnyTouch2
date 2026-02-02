import math
import sys

import torch
import torch.nn.functional as F

import util.misc as misc
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation(forces_gt, forces_pred, log_path, epoch):
    colors = ["#7998e8", "#52a375", "#803b6b"]

    correlation_fig = plt.figure(figsize=(20, 5))
    axs: np.ndarray = correlation_fig.subplots(1, 3)
    for i, (force_gt, force_pred) in enumerate(zip(forces_gt.T, forces_pred.T)):
        axs[i].scatter(
            force_gt,
            force_pred,
            s=2,
            color=colors[i]
        )
        axs[i].set_xlabel("Ground Truth (N)")
        axs[i].set_ylabel("Prediction (N)")
        axs[i].set_title(f"Force {['X', 'Y', 'Z'][i]}")
        axs[i].grid(True)
        # plot 1:1 line
        axs[i].plot(
            [force_gt.min(), force_gt.max()],
            [force_gt.min(), force_gt.max()],
            "--",
            color="gray",
        )
        axs[i].legend()
    # return correlation_fig, axs

    plt.savefig(log_path+'/correlation_epoch'+str(epoch)+'.png')
    plt.close("all")

def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):


        samples = batch[0].to(device, non_blocking=True)
        sensors = batch[1].to(device, non_blocking=True).int()
        labels = batch[2].to(device, non_blocking=True)
        force_scale = batch[3].to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            out = model(samples, sensor_type = sensors)
            loss = F.smooth_l1_loss(out, labels, beta=0.02)
            out = out.detach() 
            out_scaled = out * force_scale
            label_scaled = labels * force_scale
            mse_xyz = F.mse_loss(out_scaled, label_scaled, reduction='none').mean(dim=0)
            rmse_xyz = torch.sqrt(mse_xyz)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(rmse_x=rmse_xyz[0].item())
        metric_logger.update(rmse_y=rmse_xyz[1].item())
        metric_logger.update(rmse_z=rmse_xyz[2].item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_rmse_x', rmse_xyz[0].item(), epoch_1000x)
            log_writer.add_scalar('train_rmse_y', rmse_xyz[1].item(), epoch_1000x)
            log_writer.add_scalar('train_rmse_z', rmse_xyz[2].item(), epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    all_predictions = []
    all_targets = []
    all_force_scales = []

    for batch in metric_logger.log_every(data_loader, 40, header):

        images = batch[0].to(device, non_blocking=True)
        sensors = batch[1].to(device, non_blocking=True).int()
        target = batch[2].to(device, non_blocking=True)
        force_scale = batch[3].to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda'):
            output = model(images, sensor_type = sensors)

            all_predictions.append(output.detach())
            all_targets.append(target.detach())
            all_force_scales.append(force_scale.detach())


    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_force_scales = torch.cat(all_force_scales, dim=0)

    all_predictions = all_predictions * all_force_scales
    all_targets = all_targets * all_force_scales

    forces_rmse_xyz = torch.sqrt(((all_predictions - all_targets) ** 2).mean(dim=0)) * 1000  # in mN
    total_rmse = forces_rmse_xyz.sum()

    # print(total_rmse.shape, forces_rmse_xyz.shape)
    metric_logger.update(rmse_x=forces_rmse_xyz[0].item())
    metric_logger.update(rmse_y=forces_rmse_xyz[1].item())
    metric_logger.update(rmse_z=forces_rmse_xyz[2].item())
    metric_logger.update(rmse=total_rmse.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    print("Averaged stats:", metric_logger)
    plot_correlation(all_targets.cpu().numpy(), all_predictions.cpu().numpy(), args.log_dir, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}