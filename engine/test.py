import os

import numpy as np
import torch
import monai
import SimpleITK as sitk
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data.nifti_writer import write_nifti
from torch import autograd

import utils.misc as misc
from engine.utils import sliding_window_inference


def eval_model(inferer, model, data_loader, criterion, device, cfg, log_writer=None):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mDice', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    for c in range(cfg.output_dim):
        name = 'class' + str(c) + 'Dice'
        metric_logger.add_meter(name, misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Evaluation starting'
    print_freq = 1

    post_label = AsDiscrete(to_onehot=cfg.output_dim)
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.output_dim)
    dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=True)

    saver_rs = monai.data.NiftiSaver(
        output_dir=os.path.join(cfg.output_dir, 'monai_eval_output'), output_postfix="seg_rs", resample=True, output_dtype=np.uint8
    )

    saver = monai.data.NiftiSaver(
        output_dir=os.path.join(cfg.output_dir, 'monai_eval_output'), output_postfix="seg", resample=False, output_dtype=np.uint8
    )

    # Post-processing transforms
    # !! NOT USED !!
    if cfg.knlc:
        ap = list(range(1,cfg.output_dim))
        knlc = monai.transforms.KeepLargestConnectedComponent(applied_labels=ap)
    if cfg.rso:
        rso = monai.transforms.RemoveSmallObjects(min_size=cfg.rso_size)
    if cfg.fillholes:
        fillholes = monai.transforms.FillHoles(applied_labels=ap)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))

    autograd.set_detect_anomaly(cfg.anomaly_detection)
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        torch.cuda.empty_cache()

        inputs, labels = (batch["image"], batch["label"])
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        original_affine = batch['label_meta_dict']['original_affine']
        affine = batch['image_meta_dict']['affine']
        img_name = os.path.split(batch['image_meta_dict']['filename_or_obj'][0])[-1]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                outputs = inferer(inputs=inputs, network=model)
                loss = criterion(outputs.cpu(), labels.cpu())

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
        loss_value = loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(mDice=mDice.item())

        dice_metric.reset()

        if cfg.save_eval_output:
            if cfg.t_voxel_spacings:
                for t in batch['image_transforms']:
                    if t['class'][0] == 'Spacingd':
                        target_size = t['orig_size']
            img_name_out = 'mdice_' + str(mDice) + '_' + img_name
            # Save pred
            outputs_sm = torch.softmax(outputs.squeeze(), 0)
            outputs_hard = torch.argmax(outputs_sm, dim=0)
            #saver.save_batch(outputs_hard.cpu().unsqueeze(0), batch["image_meta_dict"])
            #saver_rs.save_batch(outputs_hard.unsqueeze(0), batch["image_meta_dict"])
            misc.save_eval_output(outputs_hard.cpu().numpy().astype(np.uint8), affine, cfg.output_dir, img_name_out, original_affine, target_size,
                                  output_folder='eval_output')
            inputImage = sitk.ReadImage(os.path.join(cfg.output_dir, 'eval_output', 'rs', img_name_out))
            npa_res = sitk.GetArrayFromImage(inputImage)
            result_image = sitk.GetImageFromArray(npa_res)
            result_image.CopyInformation(inputImage)
            os.makedirs(os.path.join(cfg.output_dir, 'sitk'), exist_ok=True)
            sitk.WriteImage(result_image, os.path.join(cfg.output_dir, 'sitk', img_name_out))

    # gather the stats from all processes
    print("Evaluation averaged stats:", metric_logger.log_all_average())
    eval_dict = {'eval/' + k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if log_writer is not None:
        for k, v in eval_dict.items():
            log_writer.add_scalar(k, v)
    return eval_dict

def test_model(model, data_loader, device, cfg, save_output=False, log_writer=None):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing starting'
    print_freq = 1

    if cfg.t_normalize:
        air_cval = (0.0 - cfg.t_norm_mean)/cfg.t_norm_std
    else:
        air_cval = 0.0

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))

    sm_outputs = []
    affines = []
    og_affines = []
    img_names = []
    target_sizes = []

    sliding_window_device = torch.device('cpu')

    autograd.set_detect_anomaly(cfg.anomaly_detection)
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        torch.cuda.empty_cache()

        inputs = batch["image"]
        inputs = inputs.to(device, non_blocking=True)
        original_affine = batch['image_meta_dict']['original_affine']
        affine = batch['image_meta_dict']['affine']
        img_name = os.path.split(batch['image_meta_dict']['filename_or_obj'][0])[-1]
        img_name = img_name.split('img')[-1]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
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

        test_outputs = torch.softmax(outputs, dim=1)


        if cfg.t_voxel_spacings:
            for t in batch['image_transforms']:
                if t['class'][0] == 'Spacingd':
                   target_size = t['orig_size']

        if save_output:
            output_am = torch.argmax(test_outputs.squeeze(), dim=0).cpu().numpy()
            os.makedirs(os.path.join(cfg.output_dir, "test_output"), exist_ok=True)
            filename = os.path.join(cfg.output_dir, "test_output", img_name)
            print("Saving file {} with original affine {} and pred affine {}".format(filename, original_affine[0], affine[0]))
            write_nifti(output_am, filename, affine[0], original_affine[0], resample=True, output_spatial_shape=target_size,
                        mode='nearest', align_corners=True, dtype=np.float32, output_dtype=np.uint8)
        else:
            sm_outputs.append(test_outputs.half().cpu())
            affines.append(affine)
            og_affines.append(original_affine)
            img_names.append(img_name)
            target_sizes.append(target_size)

    return sm_outputs, affines, og_affines, img_names, target_sizes