from models.backbones.swinception import SwInception
from models.segmentors.swin_unetr_experimental import SwInceptionDecoder


def build_model(cfg):
    if cfg.model == 'SwInception':
        encoder = SwInception(
            pretrain_img_size=cfg.vol_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.hidden_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            qkv_bias=cfg.qkv_bias,
            drop_path_rate=cfg.drop_path_rate,
            use_rel_pos_bias=cfg.rel_pos_bias
        )
        model = SwInceptionDecoder(
            encoder,
            in_channels=cfg.in_chans,
            out_channels=cfg.output_dim,
            img_size=cfg.vol_size,
            hidden_size=cfg.hidden_dim,
            patch_size=cfg.patch_size,
            decode_hs_ratio=cfg.decode_hs_ratio
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model}")
    return model
