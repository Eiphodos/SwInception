import datetime
import json
import os
import time
from pathlib import Path

import monai
import neptune
import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
from monai.data import DataLoader
from monai.losses import DiceCELoss, DiceFocalLoss, TverskyLoss, MaskedDiceLoss
from tensorboardX import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record

import utils.misc as misc
from data.dataset_builder import build_train_and_val_datasets, build_train_datasets
from engine.train import train_one_epoch
from engine.val import run_validation
from models.model_builder import build_model
from models.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.arguments import get_args


@record
def main(cfg):
    # -- Initialize distributed mode and hardware --
    misc.init_distributed_mode(cfg)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Fix the seed for reproducibility --
    if cfg.determinism:
        seed = cfg.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        monai.utils.set_determinism(seed=seed, additional_settings=None)

    # -- Setup config --
    cfg_dict = vars(cfg)

    # -- Enable logging to file and online logging to Neptune --
    if misc.get_rank() == 0 and cfg.log_dir is not None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(logdir=cfg.log_dir)
    else:
        log_writer = None
    if misc.get_rank() == 0 and cfg.neptune_logging:
        tags = misc.tag_builder(cfg)
        neptune_logger = neptune.init_run(description=cfg.description, tags=tags)
        neptune_logger['parameters'] = cfg_dict

    # -- Setup data --
    if cfg.validation:
        dataset_train, dataset_val = build_train_and_val_datasets(cfg)

        data_loader_val = DataLoader(
            dataset_val,
            batch_size=1,
            num_workers=cfg.n_workers_val,
            pin_memory=cfg.pin_mem,
            drop_last=False,
            persistent_workers=False
        )
    else:
        dataset_train = build_train_datasets(cfg)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.n_images_per_batch,
        num_workers=cfg.n_workers_train,
        pin_memory=cfg.pin_mem,
        drop_last=True,
        persistent_workers=False
    )

    # Setup model
    model = build_model(cfg)
    model.to(device)
    model_without_ddp = model
    if misc.get_rank() == 0:
        print("Model = %s" % str(model_without_ddp))
    if cfg.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = misc.count_parameters(model_without_ddp)
    if misc.get_rank() == 0 and cfg.neptune_logging:
        neptune_logger['parameters/n_parameters'] = n_parameters

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-6)
    b_opt = optimizer
    if misc.get_rank() == 0:
        print(optimizer)
    loss_scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision, init_scale=4096)
    if cfg.lr_scheduler == 'CosineAnnealing':
        scheduler = LinearWarmupCosineAnnealingLR(b_opt,
                                                  warmup_epochs=cfg.warmup_epochs,
                                                  max_epochs=cfg.epochs)
        cfg.scheduler_step_every_batch = False
    elif cfg.lr_scheduler == 'OneCycle':
        spe = len(data_loader_train) // cfg.n_images_per_batch
        spe = misc.ceil_div(spe, cfg.grad_accum_iter)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(b_opt, max_lr=cfg.lr, epochs=cfg.epochs,
                                                        steps_per_epoch=spe)
        cfg.scheduler_step_every_batch = True

    misc.load_model(cfg=cfg, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler,
                    scheduler=scheduler)

    if cfg.loss_fn == 'DiceCE':
        criterion = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=False)
    elif cfg.loss_fn == 'Tversky':
        criterion = TverskyLoss(to_onehot_y=True, softmax=True, alpha=cfg.tversky_alpha, beta=cfg.tversky_beta)
    elif cfg.loss_fn == 'DiceFocal':
        criterion = DiceFocalLoss(to_onehot_y=True, softmax=True)
    elif cfg.loss_fn == 'DiceMask':
        criterion = MaskedDiceLoss(to_onehot_y=True, softmax=True)
    else:
        raise RuntimeError('Could not parse loss function argument.')

    # Init best validation result
    best_val_metric = 0.0
    best_epoch = 0
    checkpoint_files = []

    # Run training
    start_time = time.time()
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            torch.distributed.barrier()
        train_stats = train_one_epoch(
            model, data_loader_train, scheduler,
            optimizer, criterion, device, epoch,
            loss_scaler, cfg, log_writer=log_writer)
        log_stats = {**{f'{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if cfg.validation and not((epoch + 1) % cfg.val_interval):
            if cfg.clear_val_cuda_cache:
                torch.cuda.empty_cache()
            if cfg.distributed:
                torch.distributed.barrier()
            val_stats = run_validation(model, data_loader_val, criterion, device, epoch,
                log_writer=log_writer, cfg=cfg)
            if not val_stats['skip_update']:
                log_stats_val = {**{f'{k}': v for k, v in val_stats.items()},
                         'epoch': epoch, }
                log_stats = {**log_stats, **log_stats_val}

                if log_stats_val['val/mDice'] > best_val_metric:
                    print("New record at epoch {}! Previous best metric: {}, new best metric: {}".format(epoch,
                                                                                                         best_val_metric,
                                                                                                         log_stats_val[
                                                                                                             'val/mDice']))
                    best_val_metric = log_stats_val['val/mDice']
                    best_epoch = epoch
                    misc.save_model(
                        cfg=cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, scheduler=scheduler, file_name='best_model')
                if cfg.cache_dataset and cfg.cache_rate_val < 1.0:
                    dataset_val.set_data(dataset_val.data)
            if cfg.clear_val_cuda_cache:
                torch.cuda.empty_cache()
            if cfg.distributed:
                torch.distributed.barrier()

        if cfg.output_dir and (epoch + 1) % cfg.save_ckpt_freq == 0:
            fn = 'checkpoint-{}'.format(epoch)
            misc.save_model(
                cfg=cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, scheduler=scheduler, file_name=fn)
            checkpoint_files.append(os.path.join(cfg.output_dir, fn + '.pth'))
        if cfg.output_dir and epoch + 1 == cfg.epochs:
            fn = 'final-checkpoint-{}'.format(epoch)
            misc.save_model(
                cfg=cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, scheduler=scheduler, file_name=fn)

        if misc.is_main_process() and cfg.neptune_logging:
            misc.log_to_neptune(neptune_logger, log_stats)

        if cfg.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if not cfg.scheduler_step_every_batch:
            scheduler.step()
        if cfg.cache_dataset and cfg.cache_rate_train < 1.0:
            dataset_train.set_data(dataset_train.data)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training complete! Total training time {}. Best validation metric {} at epoch {} '.format(total_time_str,
                                                                                                     best_val_metric,
                                                                                                     best_epoch))
    print("Cleaning up old checkpoints...")
    if misc.is_main_process():
        misc.cleanup_checkpoints(checkpoint_files)
    print("Finished cleaning up checkpoints!")
    if misc.is_main_process() and cfg.neptune_logging:
        neptune_logger.stop()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
