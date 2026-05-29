import math
import monai
import numpy as np
import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch import autograd

import utils.misc as misc
from engine.utils import sliding_window_inference


def run_validation(model, data_loader, criterion, device, epoch, cfg, log_writer=None):

    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('mDice', misc.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    for c in range(cfg.output_dim):
        name = 'class' + str(c) + 'Dice'
        metric_logger.add_meter(name, misc.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    header = 'Validation for epoch: [{}]'.format(epoch)
    print_freq = 1

    post_label = AsDiscrete(to_onehot=cfg.output_dim)
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.output_dim)
    dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=True)
    sliding_window_device = torch.device('cpu')

    # Post-processing transforms
    # !!NOT USED!!
    if cfg.knlc:
        ap = list(range(1, cfg.output_dim))
        knlc = monai.transforms.KeepLargestConnectedComponent(applied_labels=ap)
    if cfg.rso:
        rso = monai.transforms.RemoveSmallObjects(min_size=cfg.rso_size)
    if cfg.fillholes:
        fillholes = monai.transforms.FillHoles(applied_labels=ap)

    if cfg.t_normalize:
        air_cval = (0.0 - cfg.t_norm_mean)/cfg.t_norm_std
    else:
        air_cval = 0.0

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))
    skip_update = False

    with torch.no_grad():
        for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            autograd.set_detect_anomaly(cfg.anomaly_detection)

            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                inputs, labels = (batch["image"], batch["label"])
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(sliding_window_device, non_blocking=True)

                outputs = sliding_window_inference(inputs=inputs,
                                                   predictor=model,
                                                   roi_size=cfg.vol_size,
                                                   sw_batch_size=cfg.batch_size_val,
                                                   overlap=cfg.val_infer_overlap,
                                                   mode='gaussian',
                                                   device=sliding_window_device,
                                                   sw_device=device,
                                                   cval=air_cval
                                                   )

            labels_list = decollate_batch(labels)
            labels_convert = [post_label(label_tensor) for label_tensor in labels_list]
            outputs_list = decollate_batch(outputs)
            output_convert = [post_pred(pred_tensor) for pred_tensor in outputs_list]
            # Apply post-processing
            for i in range(len(output_convert)):
                if cfg.knlc:
                    output_convert[i] = knlc(output_convert[i])
                if cfg.rso:
                    output_convert[i] = rso(output_convert[i])
                if cfg.fillholes:
                    output_convert[i] = fillholes(output_convert[i])
            dice_metric.reset()
            dice_metric(y_pred=output_convert, y=labels_convert)
            dice_scores, dice_not_nans = dice_metric.aggregate()

            class_means = torch.zeros(cfg.output_dim)
            for c in range(cfg.output_dim):
                if dice_not_nans[:,c].sum() > 0:
                    class_dice = dice_scores[:,c].nanmean()
                else:
                    class_dice = np.nan
                class_means[c] = class_dice
                keyword_args = {'class' + str(c) + 'Dice': class_dice}
                metric_logger.update(**keyword_args)

            mDice = class_means.nanmean()
            metric_logger.update(mDice=mDice.item())

    # gather the stats from all processes
    torch.cuda.synchronize()
    metric_logger.synchronize_between_processes()
    print("Validation averaged stats:", metric_logger.log_all_average())
    val_dict = {'val/' + k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if log_writer is not None:
        for k, v in val_dict.items():
            log_writer.add_scalar(k, v, epoch)
    val_dict['skip_update'] = skip_update
    return val_dict


