import json
import sys
import os
import torch
sys.path.append(os.getcwd())
import argparse
from pathlib import Path
import librosa
from tqdm import tqdm
import multiprocessing as mp
from importlib.resources import files
import csv

import numpy as np
import sys
sys.path.insert(0, "/workspace/PEFT-TTS/src")
from f5_tts.eval.utils_eval import (
    get_kss_test,
    #get_librispeech_test,
    get_zeroshot_test,
    #get_emilia_test,
    run_asr_wer,
    run_sim,
    run_mos,
    #run_asr_cer,
)

rel_path = str(files("f5_tts").joinpath("../../"))


parser = argparse.ArgumentParser(description="WER(CER), SIM and UTMOS test")

parser.add_argument("-m", "--metrices", default="sim", type=str, choices=["cer","wer", "sim", "mos", "all"])
parser.add_argument("-l", "--language", default="ko", type=str, choices=["zh","en","ko"])
parser.add_argument("-gt", "--ground_truth", default=False, type=bool)
parser.add_argument("-g", "--gpus", default=1, type=int, choices=[0, 1])

parser.add_argument("-mc", "--model_config", default="PEFT-TTS_base", type=str, choices=["PEFT-TTS_base", "PEFT-TTS_v1"])
parser.add_argument("-d", "--dataset", default="KSS", type=str)
parser.add_argument("-c", "--ckpt_step", default=100000, type=int)
parser.add_argument("-sp", "--speaker", default="single", type=str, choices=["single", "multi"])

args = parser.parse_args()


eval_task = args.metrices
lang = args.language
eval_gt = True if args.ckpt_step == 0 else False
#eval_gt = args.ground_truth

model_config = args.model_config
train_dataset = args.dataset 
ckpt_step = args.ckpt_step
speaker = args.speaker

#gpus = [0,1]
gpus = [1]#[args.gpus]


if train_dataset == "KSS":
    if speaker == "single":
        metalst = rel_path + '/data/metadata/KSS_test_metadata.lst'
    elif speaker == "multi":
        metalst = rel_path + "/data/metadata/zeroshot_metadata.lst"


gen_wav_dir = f"{train_dataset}/{args.model_config}/{args.speaker}/model_{ckpt_step}"

if eval_gt:
    test_set = get_zeroshot_test(metalst, gen_wav_dir, gpus, eval_ground_truth=True)
else:
    test_set = get_zeroshot_test(metalst, gen_wav_dir, gpus,  ref_is_gen=False)

print(f"gen_wav_dir : {gen_wav_dir}")


    

local = False
if local:  # use local custom checkpoint dir
    asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3"
else:
    asr_ckpt_dir = ""  # auto download to cache dir

wavlm_ckpt_dir = "../checkpoints/UniSpeech/wavlm_large_finetune.pth"




def save_results():
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([epochs, cer, wer, sim, utmos])



if args.metrices is not None:
    if eval_gt :
        save_metric_dir = f"result/{train_dataset}/Ground_Truth_{args.speaker}.txt"
    else:
        save_metric_dir = f"result/{train_dataset}/{model_config}_{train_dataset}_{speaker}.txt"

    
    dir_path = os.path.dirname(save_metric_dir)

    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
# --------------------------- SIM ---------------------------
# In KSS, no need
if eval_task in ["sim"]:
    sim_list = []

    with mp.Pool(processes=len(gpus)) as pool:
        args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
        results = pool.map(run_sim, args)
        for sim_ in results:
            sim_list.extend(sim_)

    sim = round(sum(sim_list) / len(sim_list), 3)

    if eval_gt:
        evaluation_setting = f"Ground Truth\n"
    else:
        evaluation_setting = f"Steps : {ckpt_step}\n"

    result_text = f"Total {len(sim_list)} samples\nSIM      : {sim}%\n\n"
    print(f"\nTotal {len(sim_list)} samples")
    print(f"SIM      : {sim}")

    with open(save_metric_dir,"a") as f:
        f.write(evaluation_setting)
        f.write(result_text)

# --------------------------- MOS ---------------------------

if eval_task in ["mos"]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utmos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    utmos_predictor = utmos_predictor.to(device)
    
    audio_paths = [gen_wav for gen_wav, _, _ in test_set[0][1]]
    
    utmos_results = {}
    utmos_score = 0

    for audio_path in tqdm(audio_paths, desc="Processing"):
        wav_name = Path(audio_path).stem
        #wav_name = audio_path.stem
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).to(device).unsqueeze(0)

        min_length = 16000  # 최소 1초 길이 보장 (16kHz 기준)
        if wav_tensor.shape[-1] < min_length:
            wav_tensor = torch.nn.functional.pad(wav_tensor, (0, min_length - wav_tensor.shape[-1]))




        score = utmos_predictor(wav_tensor, sr)
        utmos_results[str(wav_name)] = score.item()
        utmos_score += score.item()

    avg_score = utmos_score / len(audio_paths) if len(audio_paths) > 0 else 0
    print(f"UTMOS: {avg_score}")


    if eval_gt:
        evaluation_setting = f"Ground Truth\n"
    else:
        evaluation_setting = f"Steps : {ckpt_step}\n"

    result_text = f"UTMOS      : {avg_score}\n\n"
    print(result_text)

    with open(save_metric_dir, "a") as f:
        f.write(evaluation_setting)
        f.write(result_text)
# --------------------------- WER ---------------------------

if eval_task in ["wer"]:
    wers = []
    cers = []

    with mp.Pool(processes=min(len(gpus), mp.cpu_count())) as pool:
        args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
        results = pool.map(run_asr_wer, args)

        for wers_ in results[0][0]:
            wers.extend([wers_])
        for cers_ in results[0][1]:
            cers.extend([cers_])

    wer = round(np.mean(wers) * 100, 3)
    cer = round(np.mean(cers) * 100, 3)

    if eval_gt :
        evaluation_setting = f"Ground Truth\n"
    else :
        evaluation_setting = f"Steps : {ckpt_step}\n"
    result_text = f"Total {len(wers)} samples\nWER      : {wer}%\n\n"
    result_text = result_text +f"Total {len(cers)} samples\nCER      : {cer}%\n\n"
    print(result_text)

    with open(save_metric_dir,"a") as f:
        f.write(evaluation_setting)
        f.write(result_text)
