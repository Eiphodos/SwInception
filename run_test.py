import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import ThreadDataLoader
from tensorboardX import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record

import utils.misc as misc
from data.dataset_builder import build_test_dataset
from engine.test import test_model
from models.model_builder import build_model
from utils.arguments import get_args


@record
def main(cfg):
    # -- Initialize hardware --
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Enable logging to file --
    if misc.get_rank() == 0 and cfg.log_dir is not None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(logdir=cfg.log_dir)
    else:
        log_writer = None

    # -- Setup data --
    dataset_test = build_test_dataset(cfg)

    data_loader_eval = ThreadDataLoader(
        dataset_test,
        batch_size=1,
        num_workers=0,
        pin_memory=cfg.pin_mem,
        drop_last=False,
    )

    if len(cfg.eval_folds) == 1:
        # Setup model
        model = build_model(cfg)
        model.to(device)
        print("Model = %s" % str(model))

        if isinstance(cfg.checkpoint_names, tuple):
            ckpt_name = cfg.checkpoint_names[0]
        else:
            ckpt_name = cfg.checkpoint_names
        ckpt = os.path.join(cfg.output_dir, 'Fold' + str(cfg.eval_folds[0]), ckpt_name)
        model_state_dict = torch.load(ckpt)['model']
        model.load_state_dict(model_state_dict)

        # Run evaluation
        start_time = time.time()

        test_model(model, data_loader_eval, device, cfg, save_output=True, log_writer=log_writer)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Testing time {} fold {}'.format(total_time_str, cfg.eval_folds[0]))
    else:
        sum_outputs_all_folds = []
        for j, fold in enumerate(cfg.eval_folds):
            # Setup model
            model = build_model(cfg)
            model.to(device)
            print("Model = %s" % str(model))

            if isinstance(cfg.checkpoint_names, tuple):
                ckpt_name = cfg.checkpoint_names[j]
            else:
                ckpt_name = cfg.checkpoint_names
            ckpt = os.path.join(cfg.output_dir, 'Fold' + str(fold), ckpt_name)
            model_state_dict = torch.load(ckpt)['model']
            model.load_state_dict(model_state_dict)

            # Run evaluation
            start_time = time.time()

            sm_outputs, affines, og_affines, img_names, target_sizes = test_model(model, data_loader_eval, device, cfg,
                                                                                  save_output=False, log_writer=log_writer)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Testing time {} fold {}'.format(total_time_str, fold))

            for i in range(len(img_names)):
                if j == 0:
                    sum_outputs_all_folds.append(sm_outputs[i].squeeze())
                else:
                    sum_outputs_all_folds[i] += sm_outputs[i].squeeze()


        for i in range(len(img_names)):
            output_mean = torch.div(sum_outputs_all_folds[i], len(cfg.eval_folds))
            output_am = torch.argmax(output_mean, dim=0).cpu().numpy().astype(np.uint8)
            if cfg.save_eval_output:
                misc.save_eval_output(output_am, affines[i], cfg.output_dir, img_names[i], og_affines[i], target_sizes[i])


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
