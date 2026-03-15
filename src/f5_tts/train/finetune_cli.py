
import argparse
import os
import shutil

from f5_tts.config import (
    build_train_app_config_from_args,
    validate_train_config,
    resolve_model_cls,
    resolve_checkpoint_dir,
    trainer_kwargs_from_config,
    print_train_summary,
)
from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer

# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"


def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM Model")
    parser.add_argument("--dataset_name", type=str, default="KSS")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="F5TTS_Base",
        choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base", "PEFT-TTS_base", "PEFTTTS_Base"],
        help="Experiment name",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size_per_gpu", type=int, default=3200)
    parser.add_argument("--batch_size_type", type=str, default="frame", choices=["frame", "sample"])
    parser.add_argument("--bucket_batching", action="store_true")
    parser.add_argument("--speaker_aware_batching", action="store_true")
    parser.add_argument("--bucket_size", type=int, default=512)
    parser.add_argument("--max_speakers_per_batch", type=int, default=8)
    parser.add_argument("--max_samples_per_speaker", type=int, default=8)
    parser.add_argument("--speaker_balanced_batching", action="store_true")
    parser.add_argument("--speakers_per_batch", type=int, default=8)
    parser.add_argument("--samples_per_speaker", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num_warmup_updates", type=int, default=14200)
    parser.add_argument("--save_per_updates", type=int, default=14200)
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=-1)
    parser.add_argument("--last_per_updates", type=int, default=71000)
    parser.add_argument("--finetune", dest="finetune", action="store_true", default=True)
    parser.add_argument("--no_finetune", dest="finetune", action="store_false")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="pinyin", choices=["pinyin", "char", "jamo", "custom"])
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--logger", type=str, default=None, choices=["wandb", "tensorboard"])
    parser.add_argument("--bnb_optimizer", action="store_true")
    parser.add_argument("-vv", "--view_training_procedure2", action="store_true")
    parser.add_argument("--prosody_loss_weight", type=float, default=0.3,
                        help="Weight for prosody-aware loss (0=disabled, 0.3=recommended)")
    return parser.parse_args()



def _resolve_tokenizer_source(train_cfg):
    if train_cfg.tokenizer == "custom":
        return train_cfg.tokenizer_path
    return train_cfg.dataset_name


def main():
    args = parse_args()
    cfg = build_train_app_config_from_args(args)
    validate_train_config(cfg)

    train_cfg = cfg.train
    batching_cfg = cfg.batching
    model_cfg = cfg.model

    print_train_summary(cfg)

    checkpoint_path = resolve_checkpoint_dir(
        train_cfg.dataset_name,
        model_cfg.checkpoint_subdir,
        model_cfg.name,
    )

    if train_cfg.finetune and train_cfg.pretrained_ckpt:
        os.makedirs(checkpoint_path, exist_ok=True)
        file_checkpoint = os.path.basename(train_cfg.pretrained_ckpt)
        if not file_checkpoint.startswith("pretrained_"):
            file_checkpoint = "pretrained_" + file_checkpoint
        dst = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(dst):
            shutil.copy2(train_cfg.pretrained_ckpt, dst)
            print("Copied checkpoint for finetune")

    tokenizer_source = _resolve_tokenizer_source(train_cfg)
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_source, train_cfg.tokenizer)

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model_cls = resolve_model_cls(model_cfg.backbone)
    print(f"model_cfg : {model_cfg.transformer_kwargs}")

    model = CFM(
        transformer=model_cls(**model_cfg.to_transformer_kwargs(), text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    trainer = Trainer(
        model,
        train_cfg.epochs,
        train_cfg.learning_rate,
        **trainer_kwargs_from_config(cfg, checkpoint_path=checkpoint_path, mel_spec_type=mel_spec_type),
    )

    train_dataset = load_dataset(train_cfg.dataset_name, train_cfg.tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    trainer.train(train_dataset, resumable_with_seed=666)


if __name__ == "__main__":
    main()
