import math
import sys

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from timm.utils import accuracy


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

    if args.dataset in ['material', 'cloth']:
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        loss_func = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, sensors, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        sensors = sensors.to(device, non_blocking=True).int()
        labels = labels.to(device, non_blocking=True)


        if args.use_universal:
            sensors = torch.ones_like(sensors) * -1
            sensors = sensors.int()

        with torch.amp.autocast('cuda'):
            out = model(samples, sensor_type = sensors)
            if args.dataset in ['rough', 'hard']:
                out = out.squeeze(1)
                labels = labels.float()
            # print(out.shape, labels.shape)
            loss = loss_func(out, labels)

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

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args):
    if args.dataset in ['material', 'cloth']:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        sigmoid = torch.nn.Sigmoid()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 40, header):
        images = batch[0]
        sensors = batch[1]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        sensors = sensors.to(device, non_blocking=True).int()
        target = target.to(device, non_blocking=True)

        if args.use_universal:
            sensors = torch.ones_like(sensors) * -1
            sensors = sensors.int()
        # compute output
        with torch.amp.autocast('cuda'):
            output = model(images, sensor_type = sensors)
            if args.dataset in ['rough', 'hard']:
                output = output.squeeze(1)
                target = target.float()
            loss = criterion(output, target)

        if args.dataset in ['material', 'cloth']:
            acc1, acc5 = accuracy(output, target, topk=(1,5))
        else:
            output = sigmoid(output)
            predictions = (output > 0.5).float()
            correct_predictions = (predictions == target).sum().item()
            acc1 = correct_predictions / target.size(0) * 100.0

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        if args.dataset in ['material']:
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        else:
            metric_logger.meters['acc1'].update(acc1, n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}