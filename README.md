# SwInception

Official implementation of **SwInception — Local Attention Meets Convolutions**, published in *Pattern Recognition and Artificial Intelligence*, 2025.

> David Hagerman, Roman Naeem, Jakob Lindqvist, Carl Lindström, Fredrik Kahl, Lennart Svensson

---

## Abstract

Sparse vision transformers have gained popularity as efficient encoders for medical volumetric segmentation, with Swin emerging as a prominent choice. Swin uses local attention to reduce complexity and yields excellent performance for many tasks but still tends to overfit on small datasets. To mitigate this weakness, we propose a novel architecture that further enhances Swin's inductive bias by introducing Inception blocks in the feed-forward layers. The introduction of these multi-branch convolutions enables more direct reasoning over local, multi-scale features within the transformer block. We have also modified the decoder layers in order to capture finer details using fewer parameters. We demonstrate a performance improvement on eleven different medical datasets through extensive experimentation. We specifically showcase advancements over the previous state-of-the-art backbones on benchmark challenges like the Medical Segmentation Decathlon and Beyond the Cranial Vault. By showing that the existing inductive bias in Swin can be further improved, our work presents a promising avenue for enhancing the capabilities of sparse vision transformers for both medical and natural image segmentation tasks.

---

## Results

All scores are Mean Dice (%). **Bold** = best. \* = pre-trained weights.

### Medical Segmentation Decathlon

| Method | Brain Tumour | Heart | Liver | Hippocampus | Prostate |
|--------|:---:|:---:|:---:|:---:|:---:|
| DiNTS | 72.63 | 92.20 | 72.21 | 88.13 | 70.60 |
| nnUNet | 74.03 | **93.30** | 76.84 | 89.04 | 71.72 |
| SwinUNETR | 74.26 | 90.78 | 78.69 | 87.08 | 71.59 |
| SwInception | 74.49 | 92.57 | 79.22 | 87.34 | 73.01 |
| SwInception\* | **74.57** | 92.60 | **82.19** | **89.06** | **74.77** |

| Method | Lung | Pancreas | Hepatic Vessel | Spleen | Colon | All |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| DiNTS | 60.35 | 57.98 | 59.94 | 94.68 | 37.54 | 70.63 |
| nnUNet | 64.09 | 66.58 | **66.58** | 95.35 | 41.53 | 73.91 |
| SwinUNETR | 64.68 | 62.97 | 62.72 | 95.66 | 42.74 | 73.12 |
| SwInception | 66.73 | 64.57 | 64.10 | 96.24 | 43.73 | 74.20 |
| SwInception\* | **68.03** | **67.03** | 66.33 | **96.39** | **48.19** | **75.92** |

### BTCV Multi-Organ Segmentation

| Method | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | All |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| DiNTS | 77.11 | 72.49 | 76.57 | 75.78 | 71.04 | 74.60 |
| nnUNet | 82.91 | **79.33** | 81.32 | 82.15 | 73.57 | 79.86 |
| SwinUNETR | 80.64 | 71.78 | 79.19 | 78.01 | 77.75 | 77.14 |
| SwInception | 82.53 | 71.61 | 80.49 | 80.06 | **78.67** | 78.67 |
| SwInception\* | **84.15** | 73.00 | **82.45** | **83.14** | 77.82 | **80.11** |

---

## Pre-trained Weights

| Model | Dataset | Checkpoint |
|-------|---------|:----------:|
| SwInception\* | Self-supervised pre-training following the [SwinUNETR recipe](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain) | [Download](https://drive.google.com/file/d/1P5iviDocinewt7vJNR4XmsgX_C4jpKfa/view?usp=sharing) |

---

## Installation

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

---

## Dataset Preparation

Download the datasets and place them as follows:

```
/your/dataset/dir/
├── Task03_Liver/
│   ├── imagesTr/
│   ├── labelsTr/
│   └── dataset.json
├── Task07_Pancreas/
│   └── ...
└── Abdomen/          # BTCV
    ├── imagesTr/
    └── labelsTr/
```

- [Medical Segmentation Decathlon](https://decathlon-10.grand-challenge.org/)
- [BTCV (Synapse)](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)

Cross-validation splits are provided in `data/splits/`.

---

## Training

Only distributed training is supported (runs on single GPU with `--nproc_per_node=1`).

**Example: SwInception on Decathlon Liver, fold 0, 4 GPUs**

```bash
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    run_training.py \
        --data_path /your/dataset/dir \
        --output_dir /your/output/dir \
        --log_dir /your/output/dir \
        --model SwInception \
        --pretrained /path/to/pretrained/weights \
        --task Task03_Liver \
        --cv_fold 0 \
        --seed 123 \
        --epochs 2500 \
        --output_dim 2 \
        --t_ct_min -57 \
        --t_ct_max 175 \
        --t_fixed_ct_intensity \
        --t_rand_crop_fgbg \
        --t_crop_foreground_img \
        --t_voxel_spacings \
        --t_voxel_dims 1.0 1.0 1.0 \
        --t_intensity_scale_prob 0.5 \
        --t_flip_prob 0.5 \
        --t_rot_prob 0.25 \
        --t_zoom_prob 0.2 \
        --mixed_precision \
        --no_neptune_logging
```

---

## Evaluation

```bash
python run_evaluation.py \
    --data_path /your/dataset/dir \
    --output_dir /your/output/dir \
    --log_dir /your/output/dir \
    --model SwInception \
    --resume /path/to/model/checkpoint \
    --task Task07_Pancreas \
    --cv_fold 2 \
    --seed 123 \
    --output_dim 3 \
    --t_ct_min -87 \
    --t_ct_max 199 \
    --t_fixed_ct_intensity \
    --t_voxel_spacings \
    --t_voxel_dims 1.0 1.0 1.0 \
    --mixed_precision \
    --save_eval_output
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@incollection{hagerman2025swinception,
    title = {SwInception - Local Attention Meets Convolutions},
    volume = {14892},
    booktitle = {Pattern Recognition and Artificial Intelligence},
    author = {Hagerman, David and Naeem, Roman and Lindqvist, Jakob and Lindström, Carl and Kahl, Fredrik and Svensson, Lennart},
    year = {2025},
    pages = {3--17},
}
```

---

## License

This project is released under the [MIT License](LICENSE).
