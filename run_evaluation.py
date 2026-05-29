import datetime
import json
import os
import time
from pathlib import Path

import monai
import numpy as np
import torch
from monai.data import ThreadDataLoader
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from tensorboardX import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record

import utils.misc as misc
from data.dataset_builder import build_eval_dataset
from engine.test import eval_model
from models.model_builder import build_model
from utils.arguments import get_args


@record
def main(cfg):
    # -- Initialize hardware --
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Fix the seed for reproducibility --
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed, additional_settings=None)

    # -- Enable logging to file --
    if misc.get_rank() == 0 and cfg.log_dir is not None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(logdir=cfg.log_dir)
    else:
        log_writer = None

    # -- Setup data --
    dataset_eval = build_eval_dataset(cfg)

    data_loader_eval = ThreadDataLoader(
        dataset_eval,
        batch_size=1,
        num_workers=0,
        pin_memory=cfg.pin_mem,
        drop_last=False,
    )

    criterion = DiceCELoss(to_onehot_y=True, softmax=True)

    # Setup model
    model = build_model(cfg)
    model.to(device)
    print("Model = %s" % str(model))

    model_state_dict = torch.load(cfg.resume)['model']
    model.load_state_dict(model_state_dict)

    if cfg.t_normalize:
        air_cval = (0.0 - cfg.t_norm_mean)/cfg.t_norm_std
    else:
        air_cval = 0.0

    inferer = SlidingWindowInferer(
        roi_size=cfg.vol_size,
        sw_batch_size=cfg.batch_size_val,
        overlap=cfg.val_infer_overlap,
        mode='gaussian',
        cval=air_cval
    )

    # Run evaluation
    start_time = time.time()

    eval_metrics = eval_model(inferer, model, data_loader_eval, criterion, device, cfg, log_writer=log_writer)
    log_stats = {**{f'{k}': v for k, v in eval_metrics.items()}}
    if cfg.output_dir:
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
