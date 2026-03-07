import os
import sys

sys.path.append(os.getcwd())

import argparse
import time
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import tqdm


import sys
sys.path.insert(0, "/workspace/PEFT-TTS/src")

from f5_tts.eval.utils_eval import (
    get_inference_prompt,
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
    get_zeroshot_testset_metainfo,
    get_kss_testset_metainfo,
)
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM, DiT, UNetT  # noqa: F401. used for config
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.modules import LoraConfig
from f5_tts.train.hparam import get_model_cfg
accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"


use_ema = True
target_rms = 0.1


rel_path = str(files("f5_tts").joinpath("../../"))


def main():
    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("-s", "--seed", default=None, type=int)
    parser.add_argument("-d", "--dataset_name", default="KSS", type=str)
    parser.add_argument("-bn", "--base_name", default="F5TTS_Base", type=str, choices=["F5TTS_Base", "F5TTS_v1_Base"])
    parser.add_argument("-c", "--ckptstep", default=355000, type=int)

    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-o", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)

    parser.add_argument(
        "--tokenizer", type=str, default="pinyin", choices=["pinyin", "char", "custom"], help="Tokenizer type"
    )

    parser.add_argument("-mc", "--model_config", default="PEFT-TTS", type=str)

    parser.add_argument("-sp", "--speaker", default="single", choices=["single", "multi"], type=str)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    seed = args.seed
    base_name = args.base_name
    ckpt_step = args.ckptstep
    tokenizer = args.tokenizer
    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling
    model_config = args.model_config
    speaker = args.speaker

    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    cfg_strength = 2.0
    speed = 1.0
    use_truth_duration = True # False
    no_ref_audio = False

    model_cls = DiT
    if base_name == "F5TTS_v1_Base" :
        base_config = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )
        model_cfg, adapter_config, adpt_dict = get_model_cfg(base_config)
        model_base = "PEFT-TTS_v1"
    
    elif base_name == "F5TTS_Base" :
        base_config = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
        )
        model_cfg, adapter_config, adpt_dict = get_model_cfg(base_config)
        model_base = "PEFT-TTS_base"

    target_sample_rate = 24000
    n_mel_channels = 100
    hop_length = 256
    win_length = 1024
    n_fft = 1024
    target_rms = 0.1
    mel_spec_type = "vocos"



    if speaker == "single":
        metalst = rel_path + '/data/metadata/KSS_test_metadata.lst'
    elif speaker == "multi":
        metalst = rel_path + '/data/metadata/zeroshot_metadata.lst'

    metainfo = get_kss_testset_metainfo(metalst)
    # path to save genereted wavs
    """
    output_dir = (
        f"{rel_path}/"
        f"results/{exp_name}_{ckpt_step}/{testset}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"_cfg{cfg_strength}_speed{speed}"
        f"{'_gt-dur' if use_truth_duration else ''}"
        f"{'_no-ref-audio' if no_ref_audio else ''}"
    )
    """
    # path to save genereted wavs
    output_dir = (
        f"{rel_path}/"
        f"src/f5_tts/eval"
    )

    output_dir = output_dir + f"/{dataset_name}/{model_base}/{speaker}/model_{ckpt_step}"



    # -------------------------------------------------#

    prompts_all = get_inference_prompt(
        metainfo,
        speed=speed,
        tokenizer=tokenizer,
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
        mel_spec_type=mel_spec_type,
        target_rms=target_rms,
        use_truth_duration=use_truth_duration,
        infer_batch_size=infer_batch_size,
    )

    # Vocoder model
    local = False
    if mel_spec_type == "vocos":
        vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

    # Tokenizer

    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # Model
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    #print(model)

    ckpt_path = rel_path + f"/ckpts/KSS/{model_base}/model_{ckpt_step}.pt"
    #print(ckpt_path)

    if not os.path.exists(ckpt_path):
        print(f"No such ckpt_path : {ckpt_path}")
        exit()
        #print("Loading from self-organized training checkpoints rather than released pretrained.")
        #ckpt_path = rel_path + f"/{model_cfg.ckpts.save_dir}/model_{ckpt_step}.pt"
    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)

    # start batch inference
    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = prompt
            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
            total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)

            # Inference
            with torch.inference_mode():
                generated, _ = model.sample(
                    cond=ref_mels,
                    text=final_text_list,
                    duration=total_mel_lens,
                    lens=ref_mel_lens,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    no_ref_audio=no_ref_audio,
                    seed=seed,
                    #cache=False,
                )
                # Final result
                for i, gen in enumerate(generated):
                    gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
                    gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
################################################### GET-MEL-SPECTROGRAM ####################################################

                    """
                    import numpy as np
                    # torch tensor를 numpy 배열로 변환 (채널 차원 제거)
                    mel_np = gen_mel_spec.squeeze(0).cpu().numpy()

                    #save_folder = f"eval_mel/{output_dir}"
                    save_folder = output_dir.replace("/wav/", "/mel/", 1)
                    #print(f"save_folder : {save_folder}")
                    #exit()
                    os.makedirs(save_folder, exist_ok=True)
                    np.save(os.path.join(save_folder, f"{utts[i]}_mel_spec.npy"), mel_np)
                    """
############################################################################################################################
                    if mel_spec_type == "vocos":
                        generated_wave = vocoder.decode(gen_mel_spec).cpu()
                    elif mel_spec_type == "bigvgan":
                        generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

                    if ref_rms_list[i] < target_rms:
                        generated_wave = generated_wave * ref_rms_list[i] / target_rms
                    torchaudio.save(f"{output_dir}/{utts[i]}.wav", generated_wave, target_sample_rate)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60 :.2f} minutes.")


if __name__ == "__main__":
    main()