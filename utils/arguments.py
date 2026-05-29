import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_data_config_args(parser)
    parser = add_transform_config_args(parser)
    parser = add_post_proc_config_args(parser)
    parser = add_optimizer_config_args(parser)
    parser = add_training_config_args(parser)
    parser = add_misc_config_args(parser)

    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    args_dict = vars(args)

    # Change all arguments that are lists into either single values or tuples.
    for k, v in args_dict.items():
        if isinstance(v, list):
            if len(v) == 1:
                setattr(args, k, v[0])
            else:
                setattr(args, k, tuple(v))

    return args


def add_model_config_args(parser):
    group = parser.add_argument_group('model', 'Model type and settings')

    group.add_argument('--model', default='SwInception', type=str,
                        choices=['SwInception'],
                        help='Model name')

    group.add_argument('--vol_size', nargs='*', default=[96, 96, 96], type=int,
                        help='volume input size, can be a single number or for example --input_size 128 64 32 as H*W*D')
    group.add_argument('--patch_size', nargs='*', default=[2, 2, 2], type=int,
                        help='patch input size, can be a single number or for example --patch_size 128 64 32 as H*W*D')
    group.add_argument('--window_size', nargs='*', default=[7, 7, 7, 7], type=int,
                       help='Attention window size, can be a single number or for example --window_size 3 3 3 3 as H*W*D')

    group.add_argument('--input_dim', default=3, type=int,
                        help='Dimension of the input, allowed values are 2 and 3')
    group.add_argument('--output_dim', default=3, type=int,
                        help='Dimension of the output, must be equal to the number of classes being segmented.')
    group.add_argument('--in_chans', default=1, type=int,
                        help='Number of channels for the volumes')
    group.add_argument('--hidden_dim', default=48, type=int,
                        help='Dimension of hidden/embedding dimension')
    group.add_argument('--depths', nargs='*', default=[2, 2, 2, 2], type=int,
                       help='Depth each internal layer')
    group.add_argument('--num_heads', nargs='*', default=[3, 6, 12, 24], type=int,
                       help='Number of transformer heads in each layer.')
    group.add_argument('--mlp_ratio', default=4.0, type=float,
                        help='Hidden dimension scale for feedforward layers')

    group.add_argument('--no_rel_pos_bias', action='store_false', dest='rel_pos_bias',
                        help='Disable relative position bias in the encoder')
    group.set_defaults(rel_pos_bias=True)

    group.add_argument('--abs_pos_emb', action='store_true',
                        help='Use absolute position emb in the encoder')
    group.set_defaults(abs_pos_emb=False)

    group.add_argument('--no_qkv_bias', action='store_true',
                        help='Dont use bias for attention qkv in the encoder')
    group.set_defaults(qkv_bias=True)

    group.add_argument('--gradient_clipping', type=float,
                       help='Sets the gradient clipping to the specified number. Gradient clipping disabled when None')
    group.set_defaults(gradient_clipping=None)

    group.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision for model, operations and input')
    group.set_defaults(mixed_precision=False)

    group.add_argument('--use_abs_pos_emb', action='store_true',
                       help='Use global absolute position embeddings')
    group.set_defaults(use_abs_pos_emb=False)

    group.add_argument('--segformer_sr', nargs='*', default=[8, 4, 2, 1], type=int,
                       help='Segformer SR ratios')
    group.add_argument('--decode_hs_ratio', default=2.0, type=float,
                       help='The ratio for the hidden size of the decoder')
    group.add_argument('--drop_path_rate', default=0.2, type=float, help='Path drop rate for each Transformer block')
    group.add_argument('--win_drop_rate', nargs='*', default=[0.0, 0.0, 0.0, 0.0], type=float,
                       help='Window drop rate for each Transformer layer')
    group.add_argument('--top_k_ratio', nargs='*', default=[0.0, 0.0, 0.0, 0.0], type=float,
                       help='Number of tokens selected by top K for heuristic attention')
    group.add_argument('--top_k_dim_ratio', nargs='*', default=[0.0, 0.0, 0.0, 0.0], type=float,
                       help='Number of dims selected by top K for heuristic attention')



    return parser


def add_transform_config_args(parser):
    group = parser.add_argument_group('transform', 'Settings for transform, both training and validation')

    group.add_argument('--t_voxel_spacings', action='store_true',
                       help='Transform all data into the same voxel spacings')
    group.set_defaults(t_voxel_spacings=False)
    group.add_argument('--t_voxel_dims', nargs='*', default=[1.0], type=float,
                        help='The spacings to transform the voxels into. Can be a single value or a tuple of 3.')

    group.add_argument('--t_cubed_ct_intensity', action='store_true',
                       help='Take the cube root of all ct intensity values to make the input space smaller')
    group.set_defaults(t_cubed_ct_intensity=False)

    group.add_argument('--t_fixed_ct_intensity', action='store_true',
                       help='Clip and normalize ct intensity range to the fixed range between ct min and ct max')
    group.set_defaults(t_fixed_ct_intensity=False)

    group.add_argument('--t_percentile_ct_intensity', action='store_true',
                       help='Clip and normalize ct intensity range to the percentile based ct min and ct max. Default is 5 to 95 percent.')
    group.set_defaults(t_percentile_ct_intensity=False)

    group.add_argument('--t_ct_min', default=-1000, type=int,
                        help='The minimum CT intensity value to clip to')
    group.add_argument('--t_ct_max', default=1000, type=int,
                       help='The maximum CT intensity value to clip to')

    group.add_argument('--t_crop_foreground_img', action='store_true',
                       help='Crop volumes of space that consists of air')
    group.set_defaults(t_crop_foreground_img=False)
    group.add_argument('--t_crop_foreground_kdiv', action='store_true',
                       help='Crop volumes of space that consists of air with a restriction that the resulting volume must be divisible by the target crop size')
    group.set_defaults(t_crop_foreground_kdiv=False)
    group.add_argument('--t_rand_crop_fgbg', action='store_true',
                       help='Crop subvolumes based on foreground/background')
    group.set_defaults(t_rand_crop_fgbg=False)
    group.add_argument('--t_rand_crop_pos_weight', type=float, default=1.0,
                       help='Weight used for positive examples in random cropping')
    group.add_argument('--t_rand_crop_neg_weight', type=float, default=1.0,
                       help='Weight used for negative examples in random cropping')
    group.add_argument('--t_rand_crop_classes', action='store_true',
                       help='Crop subvolumes based on classes')
    group.set_defaults(t_rand_crop_classes=False)
    group.add_argument('--t_rand_crop_dilated_center', action='store_true',
                       help='Crop subvolumes by selecting voxels not only of a specific class fg/bg as the center but with a dilation from those voxels')
    group.set_defaults(t_rand_crop_dilated_center=False)
    group.add_argument('--t_rand_spatial_crop', action='store_true',
                       help='Basic random spatial crop')
    group.set_defaults(t_rand_spatial_crop=False)


    group.add_argument('--t_spatial_pad', action='store_true',
                       help='Pad volumes to the input volume size')
    group.set_defaults(t_spatial_pad=False)

    group.add_argument('--t_convert_labels_to_brats', action='store_true',
                       help='Convert labels to multi-channel based on BRATS classes')
    group.set_defaults(t_convert_labels_to_brats=False)


    group.add_argument('--t_normalize', action='store_true',
                       help='Normalize intensity values')
    group.set_defaults(t_normalize=False)
    group.add_argument('--t_normalize_channel_wise', action='store_true',
                       help='Normalize intensity values channel wise, used for multi-channel input such as MRI')
    group.set_defaults(t_normalize_channel_wise=False)
    group.add_argument('--t_norm_mean', default=0.1943, type=float,
                       help='The probability for a random flip in a direction')
    group.add_argument('--t_norm_std', default=0.2786, type=float,
                       help='The probability for a random flip in a direction')


    group.add_argument('--t_n_patches_per_image', default=4, type=int,
                       help='The number of samples for the random cropping')

    group.add_argument('--t_flip_prob', default=0.0, type=float,
                       help='The probability for a random flip in a direction')

    group.add_argument('--t_rot_prob', default=0.0, type=float,
                       help='The probability for a random rotate in a direction')

    group.add_argument('--t_intensity_shift_os', default=0.1, type=float,
                       help='The offset for random intensity shift')
    group.add_argument('--t_intensity_shift_prob', default=0.0, type=float,
                       help='The probability for a random intensity shift')
    group.add_argument('--t_intensity_scale_factors', default=0.1, type=float,
                       help='The offset for random intensity shift')
    group.add_argument('--t_intensity_scale_prob', default=0.0, type=float,
                       help='The probability for a random intensity shift')
    group.add_argument('--t_axcodes', default='RAS', type=str, help='Orientation of image')
    group.add_argument('--t_gibbs_noise_prob', default=0.0, type=float,
                       help='The probability for gibbs noise (MRI)')
    group.add_argument('--t_rician_noise_prob', default=0.0, type=float,
                       help='The probability for rician noise (MRI)')
    group.add_argument('--t_gaussian_noise_prob', default=0.0, type=float,
                       help='The probability for gaussian noise')
    group.add_argument('--t_rand_affine_prob', default=0.0, type=float,
                       help='The probability to add affine transform')
    group.add_argument('--t_adjust_contrast_prob', default=0.0, type=float,
                       help='The probability to adjust contrast')
    group.add_argument('--t_adjust_contrast_gamma', default=4.5, type=float,
                       help='The gamma to adjust contrast with')
    group.add_argument('--t_zoom_prob', default=0.0, type=float, help='Probability to apply random zoom')
    group.add_argument('--t_zoom_min', default=1.2, type=float, help='Minimum zoom factor')
    group.add_argument('--t_zoom_max', default=1.5, type=float, help='Maximum zoom factor')

    return parser

def add_post_proc_config_args(parser):
    group = parser.add_argument_group('post_proc', 'Settings for post processing transforms and algorithms')

    group.add_argument('--knlc', action='store_true', help='If enabled, applies post processing on labels'
                                                                          'and keeps n largest components')
    group.set_defaults(knlc=False)
    group.add_argument('--knlc_n_components', type=int, default=1, help='Number of largest components to keep')
    group.add_argument('--rso', action='store_true', help='If enabled, applies post processing on labels'
                                                                          'and removes small objects')
    group.set_defaults(rso=False)
    group.add_argument('--rso_size', type=int, default=5, help='Objects smaller than this size are removed')
    group.add_argument('--fillholes', action='store_true', help='If enabled, applies post processing on labels'
                                                                          'and fill holes in objects')
    group.set_defaults(fillholes=False)

    return parser

def add_data_config_args(parser):
    group = parser.add_argument_group('data', 'Settings for data, both reading from disk and loading into model')

    group.add_argument('--data_path', default='/datasets/', type=str,
                        help='Dataset path')
    group.add_argument('--json_list', default='dataset.json', type=str,
                       help='Json file containing the list of dataset files in Decathlon format')
    group.add_argument('--task', default='Task03_Liver', type=str, help='The segmentation task to finetune on.')
    group.add_argument('--batch_size_val', type=int, default=4, help='Batch size for validation data loader')
    group.add_argument('--n_images_per_batch', type=int, default=1, help='Number of unique images per batch to pull patches from. Total Batch size is n_images_per_batch * t_n_samples_per_image.')
    group.add_argument('--n_workers_train', type=int, default=16, help='Number of workers used in Train DataLoader')
    group.add_argument('--n_workers_val', type=int, default=8, help='Number of workers used in Val DataLoader')
    group.add_argument('--no_pin_memory', action='store_false', dest='pin_mem',
                       help='When enabled Dataloaders wont pin GPU memory')
    group.set_defaults(pin_mem=True)
    group.add_argument('--no_cache_dataset', action='store_false', dest='cache_dataset',
                       help='When enabled only use the default type of Dataset instead of CacheDataset, mostly used for debugging.')
    group.set_defaults(cache_dataset=True)
    group.add_argument('--cache_rate_train', type=float, default=1.0,
                       help='The percentage of cached training data in total.')
    group.add_argument('--cache_rate_val', type=float, default=1.0,
                       help='The percentage of cached validation data in total.')
    group.add_argument('--dataset_ratio', type=float, default=1.0,
                       help='The percentage of the entire dataset to use during training, if below 1.0'
                            ' the remaining data is used for validation')

    return parser


def add_optimizer_config_args(parser):
    group = parser.add_argument_group('optimizer', 'Optimzer settings')

    group.add_argument('--loss_fn', type=str, default='DiceCE',
                       help='The name of the loss function to use. Available options: DiceCE, DiceFocal and Tversky')
    group.add_argument('--tversky_alpha', type=float, default=0.5,
                       help='Weight for false positives. When both alpha and beta is 0.5, Tversky loss is identical to Dice loss.')
    group.add_argument('--tversky_beta', type=float, default=0.5,
                       help='Weight for false negatives. When both alpha and beta is 0.5, Tversky loss is identical to Dice loss.')
    group.add_argument('--smooth_nr', type=float, default=1e-5,
                       help='Smoothing factor for numerator in Dice loss to avoid 0')
    group.add_argument('--smooth_dr', type=float, default=1e-5,
                       help='Smoothing factor for denominator in Dice loss to avoid nan')
    group.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay (default: 0.05)')
    group.add_argument('--lr', type=float, default=4e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    group.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for optimizer')
    group.add_argument('--warmup_epochs', type=int, default=50, metavar='N',
                        help='epochs to warmup LR')
    group.add_argument('--lr_scheduler', type=str, default='CosineAnnealing',
                       help='The name of the learning rate scheduler')
    group.add_argument('--grad_accum_iter', type=int, default=1,
                       help='The number of iterations to accumulate gradients before doing a step')


    return parser


def add_training_config_args(parser):
    group = parser.add_argument_group('training', 'Training settings')

    group.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    group.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    group.add_argument('--save_ckpt_freq', default=50, type=int)
    group.add_argument('--val_interval', default=50, type=int)
    group.add_argument('--no_validation', action='store_false', dest='validation',
                       help='Disables validation and enables training with the entire dataset')
    group.set_defaults(validation=True)
    group.add_argument('--cv_fold', default=0, type=int, help='Current fold for cross validation')
    group.add_argument('--cv_max_folds', default=5, type=int, help='Max folds for cross validation')
    group.add_argument('--eval_folds', type=str, nargs='*', default=[0, 1, 2, 3, 4],
                       help='Folds to use for evaluation/testing')
    group.add_argument('--val_infer_overlap', default=0.5, type=float,
                       help='Overlap between each sliding window in validation')


    # distributed training parameters
    group.add_argument('--world_size', default=1, type=int,  help='number of distributed processes')
    group.add_argument('--local_rank', default=-1, type=int)
    group.add_argument('--dist_on_itp', action='store_true')
    group.add_argument('--dist_url', default='env://',  help='url used to set up distributed training')
    group.add_argument('--backend', default='nccl',  help='Backend to use for distributed training')

    # Checkpoint loading
    group.add_argument('--resume', default='', help='Path to checkpoint to resume from')
    group.add_argument('--resume_training', action='store_true', help='If enabled, resuming also loads optimizer, '
                                                                      'scheduler and scaler')
    group.add_argument('--pretrained', type=str, help='Pretrained checkpoint for backbone')
    group.add_argument('--checkpoint_names', type=str, nargs='*', default=['best_model.pth'],
                       help='One or sevaral checkpoint names for each fold used in testing')

    group.add_argument('--freeze_loaded_weights', action='store_true', help='If all loaded weights should be frozen')
    group.add_argument('--freeze_n_epochs', type=int, default=25, help='Number of epochs to keep weights frozen for')

    return parser


def add_misc_config_args(parser):
    group = parser.add_argument_group('misc', 'Misc settings')

    group.add_argument('--seed', type=int, default=13, help='Random seed for deterministic training')
    group.add_argument('--no_determinism', action='store_false', dest='determinism',
                       help='Disables deterministic training')
    group.set_defaults(determinism=True)
    group.add_argument('--no_cuddn_auto_tuner', action='store_true', help='Disables the CUDDN benchmark in PyTorch')
    group.add_argument('--anomaly_detection', action='store_true', help='Enables PyTorch anomaly detection')

    group.add_argument('--log_dir', type=str, help='Folder where the logs should be saved')
    group.add_argument('--no_neptune_logging', action='store_false', dest='neptune_logging',
                       help='If online logging to neptune should be disabled')
    group.set_defaults(neptune_logging=True)
    group.add_argument('--save_eval_output', action='store_true', help='If evaluated volumes should be saved on disk')
    group.add_argument('--output_dir', type=str, help='path where to save, empty for no saving')
    group.add_argument('--description', type=str, help='The description of the experiment, used for Neptune logging.')
    group.add_argument('--debug', action='store_true', help='General flag to enable debugging (often prints)')
    group.add_argument('--clear_val_cuda_cache', action='store_true', help='If enabled, clears CUDA cache before and after validation')

    return parser