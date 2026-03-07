import math
import os
import random
import string
import re

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from importlib.resources import files

import sys
sys.path.insert(0, "/workspace/PEFT-TTS/src")

from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import convert_char_to_pinyin


# seedtts testset metainfo: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_seedtts_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            gt_wav = os.path.join(os.path.dirname(metalst), "wavs", utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))
    return metainfo


def get_zeroshot_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []

    #rel_path = "workspace/F5-TTS/Emilia_Dataset/KO/"
    rel_path = str(files("f5_tts").joinpath("../../Emilia_Dataset/KO"))
    for line in lines:
        print(line)
        ref_wav, ref_dur, ref_txt, gen_wav, gen_dur, gen_txt = line.strip().split("|")
        gen_utt = ""+ gen_wav.split("/")[-1][:-4]

        ref_wav = os.path.join(rel_path, ref_wav)
        gen_wav = os.path.join(rel_path, gen_wav)


        metainfo.append((gen_utt, ref_txt, ref_wav, " "+gen_txt, gen_wav))

    return metainfo

def get_kss_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        print(line)
        ref_wav, ref_dur, ref_txt, gen_wav, gen_dur, gen_txt = line.strip().split("|")
        gen_utt = ""+ gen_wav.split("/")[-1][:-4]
        metainfo.append((gen_utt, ref_txt, ref_wav, " "+gen_txt, gen_wav))

    return metainfo
        
def get_emilia_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        print(line)
        ref_wav, ref_dur, ref_txt, gen_wav, gen_dur, gen_txt = line.strip().split("|")
        gen_utt = ""+ gen_wav.split("/")[-1][:-4]
        metainfo.append((gen_utt, ref_txt, ref_wav, " "+gen_txt, gen_wav))

    return metainfo
# librispeech test-clean metainfo: gen_utt, ref_txt, ref_wav, gen_txt, gen_wav
def get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        # ref_txt = ref_txt[0] + ref_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        # gen_txt = gen_txt[0] + gen_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
        gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")

        metainfo.append((gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav))

    return metainfo


# padded to max length mel batch
def padded_mel_batch(ref_mels):
    max_mel_length = torch.LongTensor([mel.shape[-1] for mel in ref_mels]).amax()
    padded_ref_mels = []
    for mel in ref_mels:
        padded_ref_mel = F.pad(mel, (0, max_mel_length - mel.shape[-1]), value=0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    padded_ref_mels = padded_ref_mels.permute(0, 2, 1)
    return padded_ref_mels



# get prompts from metainfo containing: utt, prompt_text, prompt_wav, gt_text, gt_wav


def get_inference_prompt(
    metainfo,
    speed=1.0,
    tokenizer="pinyin",
    polyphone=True,
    target_sample_rate=24000,
    n_fft=1024,
    win_length=1024,
    n_mel_channels=100,
    hop_length=256,
    mel_spec_type="vocos",
    target_rms=0.1,
    use_truth_duration=False,
    infer_batch_size=1,
    num_buckets=200,
    min_secs=3,
    max_secs=40,
):
    prompts_all = []

    min_tokens = min_secs * target_sample_rate // hop_length
    max_tokens = max_secs * target_sample_rate // hop_length

    batch_accum = [0] * num_buckets
    utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = (
        [[] for _ in range(num_buckets)] for _ in range(6)
    )

    mel_spectrogram = MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    for utt, prompt_text, prompt_wav, gt_text, gt_wav in tqdm(metainfo, desc="Processing prompts..."):
        # Audio
        print(f"utt : {utt}\nprompt_text : {prompt_text}\nprmopt_wav : {prompt_wav}\ngt_text : {gt_text}\ngt_wav : {gt_wav}")
        assert os.path.exists(prompt_wav), f"Prompt WAV file not found: {prompt_wav}"
        ref_audio, ref_sr = torchaudio.load(prompt_wav)
        # ref_audio: [C, samples]
        if ref_audio.size(0) > 1:
            # 채널 평균이나 첫 채널만 사용
            print("it's not mono")
            ref_audio = ref_audio.mean(dim=0, keepdim=True)  # -> [1, samples]

        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
        if ref_rms < target_rms:
            ref_audio = ref_audio * target_rms / ref_rms
        assert ref_audio.shape[-1] > 5000, f"Empty prompt wav: {prompt_wav}, or torchaudio backend issue."
        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio)

        # Text
        if len(prompt_text[-1].encode("utf-8")) == 1:
            prompt_text = prompt_text + " "
        text = [prompt_text + gt_text]
        print(f"before tokenizer after concatenation : {text}")
        if tokenizer == "pinyin":
            text_list = convert_char_to_pinyin(text, polyphone=polyphone)
            print(f"after tokenizer : {text_list}")
        else:
            text_list = text

        # Duration, mel frame length
        ref_mel_len = ref_audio.shape[-1] // hop_length
        if use_truth_duration:
            gt_audio, gt_sr = torchaudio.load(gt_wav)
            if gt_sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(gt_sr, target_sample_rate)
                gt_audio = resampler(gt_audio)
            total_mel_len = ref_mel_len + int(gt_audio.shape[-1] / hop_length / speed)

            # # test vocoder resynthesis
            # ref_audio = gt_audio
        else:
            ref_text_len = len(prompt_text.encode("utf-8"))
            gen_text_len = len(gt_text.encode("utf-8"))
            #ref_text_len = len(prompt_text)
            #gen_text_len = len(gt_text)
            total_mel_len = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len / speed)

################################################# PRINT DURATION #######################################################
            print(f"ref_audio.shape[-1] : {ref_audio.shape[-1]}")
            print(f"hop_length : {hop_length}")
            print(f"ref_mel_len : {ref_mel_len}")

            print(f"promt_text : {prompt_text.encode('utf-8')}")
            print(f"gen_text   : {gt_text.encode('utf-8')}")
            print(f"ref_text_len : {ref_text_len}")
            print(f"gen_text_len : {gen_text_len}")
            print(f"total_mel_len : {total_mel_len}")
            #exit()

########################################################################################################################
        # to mel spectrogram
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.squeeze(0)

        # deal with batch
        assert infer_batch_size > 0, "infer_batch_size should be greater than 0."
        assert (
            min_tokens <= total_mel_len <= max_tokens
        ), f"Audio {utt} has duration {total_mel_len*hop_length//target_sample_rate}s out of range [{min_secs}, {max_secs}]."
        bucket_i = math.floor((total_mel_len - min_tokens) / (max_tokens - min_tokens + 1) * num_buckets)

        utts[bucket_i].append(utt)
        ref_rms_list[bucket_i].append(ref_rms)
        ref_mels[bucket_i].append(ref_mel)
        ref_mel_lens[bucket_i].append(ref_mel_len)
        total_mel_lens[bucket_i].append(total_mel_len)
        final_text_list[bucket_i].extend(text_list)

        batch_accum[bucket_i] += total_mel_len

        if batch_accum[bucket_i] >= infer_batch_size:
            # print(f"\n{len(ref_mels[bucket_i][0][0])}\n{ref_mel_lens[bucket_i]}\n{total_mel_lens[bucket_i]}")
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
            batch_accum[bucket_i] = 0
            (
                utts[bucket_i],
                ref_rms_list[bucket_i],
                ref_mels[bucket_i],
                ref_mel_lens[bucket_i],
                total_mel_lens[bucket_i],
                final_text_list[bucket_i],
            ) = [], [], [], [], [], []

    # add residual
    for bucket_i, bucket_frames in enumerate(batch_accum):
        if bucket_frames > 0:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
    # not only leave easy work for last workers
    random.seed(666)
    random.shuffle(prompts_all)

    return prompts_all


# get wav_res_ref_text of seed-tts test metalst
# https://github.com/BytedanceSpeech/seed-tts-eval


def get_seed_tts_test(metalst, gen_wav_dir, gpus):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")

        if not os.path.exists(os.path.join(gen_wav_dir, utt + ".wav")):
            continue
        gen_wav = os.path.join(gen_wav_dir, utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

        test_set_.append((gen_wav, prompt_wav, gt_text))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set

def get_zeroshot_test(metalst, gen_wav_dir, gpus, eval_ground_truth=False, ref_is_gen=False):
    f= open(metalst)
    lines = f.readlines()
    f.close()
    rel_path = str(files("f5_tts").joinpath("../../Emilia_Dataset/KO"))
    test_set_ = []

    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("|")
        if ref_is_gen:
            ref_wav = gen_utt
        else:
            ref_wav = os.path.join(rel_path, ref_utt)

        if eval_ground_truth:
            gen_wav = os.path.join(rel_path, gen_utt)
            if not os.path.exists(gen_wav):
                raise FileNotFoundError(f"Generated wav not found: {gen_wav}")

        else:
            gen_utt = ""+ gen_utt.split("/")[-1][:-4]
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + f".wav")):
                raise FileNotFoundError(f"Generated wav not found: {gen_wav_dir}/{gen_utt}.wav")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + f".wav")



        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set

def get_kss_test(metalst, gen_wav_dir, gpus, eval_ground_truth=False, ref_is_gen=False):
    f= open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")
        if ref_is_gen:
            ref_wav = gen_utt
        else:
            ref_wav = ref_utt
        if eval_ground_truth:
            gen_wav = gen_utt
        else:
            gen_utt = ""+ gen_utt.split("/")[-1][:-4]
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".wav")):
                raise FileNotFoundError(f"Generated wav not found: {gen_utt}")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")

        
        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set

# get librispeech test-clean cross sentence test

def get_emilia_test(metalst, gen_wav_dir, gpus, eval_ground_truth=False, ref_is_gen=False):
    f= open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("|")

        if ref_is_gen:
            ref_wav = gen_utt
        else:
            ref_wav = ref_utt

        if eval_ground_truth:
            #gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
            #gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".wav")
            
            gen_wav = gen_utt
            #print(gen_wav)
        else:
            gen_utt = ""+ gen_utt.split("/")[-1][:-4]
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".wav")):
                raise FileNotFoundError(f"Generated wav not found: {gen_wav_dir}/{gen_utt}.wav")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")

        #ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


def get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth=False):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        if eval_ground_truth:
            gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
            gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")
        else:
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".wav")):
                raise FileNotFoundError(f"Generated wav not found: {gen_utt}")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")

        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


# load asr model


def load_asr_model(lang, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel

        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            # vad_model = os.path.join(ckpt_dir, "fsmn-vad"),
            # punc_model = os.path.join(ckpt_dir, "ct-punc"),
            # spk_model = os.path.join(ckpt_dir, "cam++"),
            disable_update=True,
        )  # following seed-tts setting
    elif lang == "en":
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")

    elif lang == "ko":
        from transformers import pipeline

        model = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",  
            device="cuda",  
            #language= "ko",
        )

        """
        # whisper-large
        import whisper
        model = whisper.load_model("large", device="cuda")


        # whisper-large-v3
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        """
    return model

# WER Evaluation, the way Seed-TTS does

def caculate_cer(truth, hypo):
    from jiwer import compute_measures
    measures = compute_measures(truth, hypo)
    total_chars = len(truth.replace(" ", "")) # exclude space
    if total_chars == 0:
        return 0.0

    cer = (measures["substitutions"] + measures["deletions"] + measures["insertions"]) / total_chars
    return cer

def create_asr_result(truth: str, hypo: str, dataset_name: str) -> str:
    os.makedirs("ASR_RESULT", exist_ok=True)

    # Define the results file path
    result_file = os.path.join("ASR_RESULT", f"{dataset_name}.txt")

    # If this is the first time writing, add a header
    write_header = not os.path.exists(result_file)

    # Open in append mode and write the entry
    with open(result_file, "a", encoding="utf-8") as fw:
        clean_truth = truth.replace("\n", " ").strip()
        clean_hypo  = hypo.replace("\n", " ").strip()
        fw.write(f"truth  : {clean_truth}\n")
        fw.write(f"hypo. : {clean_hypo}\n\n")

    #return result_file

def run_asr_wer(args):
    rank, lang, test_set, ckpt_dir = args
    
    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en" or lang == "ko":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' & 'ko' (faster-whisper-large-v3), for now."
        )

    asr_model = load_asr_model(lang, ckpt_dir=ckpt_dir)
    print("fininsh loading asr_model")
    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation
    wers = []
    cers = []

    from jiwer import compute_measures

    for gen_wav, prompt_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        elif lang == "en":
            segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="ko")
            hypo = ""
            for segment in segments:
                hypo = hypo + " " + segment.text
        elif lang == "ko":
            """
            segments = asr_model.transcribe(gen_wav, beam_size =5, language="ko")
            hypo = segments["text"]
            """
            hypo = asr_model(gen_wav, return_timestamps=False)["text"]
        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()
        elif lang == "ko":  
            truth = " ".join(truth.split())  
            hypo = " ".join(hypo.split())





        measures = compute_measures(truth, hypo)
        #dataset_name = "CosyVoice2"
        cer = caculate_cer(truth, hypo)
        wer = measures["wer"]

        print(f"truth : {truth}")
        print(f"hypo  : {hypo}")
        #create_asr_result(truth, hypo, dataset_name)
        wers.append(wer)
        cers.append(cer)

    return wers, cers


# SIM Evaluation


def run_sim(args):
    rank, test_set, ckpt_dir = args
    device = f"cuda:{rank}"
    print(f"device : {device}")
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sim_list = []
    for wav1, wav2, truth in tqdm(test_set):
        print(truth)
        print(f"wav1 : {wav1}")
        print(f"wav2 : {wav2}")

        wav1, sr1 = torchaudio.load(wav1)
        wav2, sr2 = torchaudio.load(wav2)


        resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
        resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
        wav1 = resample1(wav1)
        wav2 = resample2(wav2)

        # 여기에서 길이 체크 및 패딩 필요!
        min_length = 16000  # 최소 1초

        if wav1.shape[-1] < min_length:
            pad_size = min_length - wav1.shape[-1]
            wav1 = F.pad(wav1, (0, pad_size))

        if wav2.shape[-1] < min_length:
            pad_size = min_length - wav2.shape[-1]
            wav2 = F.pad(wav2, (0, pad_size))



        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        sim_list.append(sim)

    return sim_list


def run_mos(args):
    rank, sub_test_set = args
    mos_scores = []
    
    for sample in sub_test_set:
        audio_path = sample['gen_wav']  # Assuming 'gen_wav' holds the path to the generated audio
        audio, sr = torchaudio.load(audio_path)
        audio = audio.cuda()

        # Resampling to 16kHz
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).cuda()
        audio_16k = resampler(audio)

        # 여기에서 길이 체크 및 패딩 필요!
        min_length = 16000  # 최소 1초

        # 오디오 최소 길이 확인 후 패딩 추가 (최소 1초 보장)
        min_length = 16000  # 최소 1초 (16kHz 기준)
        if audio_16k.shape[-1] < min_length:
            audio_16k = torch.nn.functional.pad(audio_16k, (0, min_length - audio_16k.shape[-1]))



        # Predict MOS
        mos = utmos_predictor(audio_16k, 16000).item()
        mos_scores.append(mos)
    return mos_scores


