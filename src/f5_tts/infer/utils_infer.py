# Unified inference helpers for CLI / app usage.
import hashlib
import inspect
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
third_party_bigvgan = Path(__file__).resolve().parents[3] / "third_party" / "BigVGAN"
if third_party_bigvgan.exists():
    sys.path.append(str(third_party_bigvgan))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download, snapshot_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import convert_char_to_pinyin_orig, get_tokenizer

_ref_audio_cache = {}
_resampler_cache = {}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = get_device()

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

asr_pipe = None


def chunk_text(text, max_chars=135):
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        candidate = sentence + (" " if sentence and len(sentence[-1].encode("utf-8")) == 1 else "")
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += candidate
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")

        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        return vocoder.eval().to(device)

    if vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError as exc:
            raise ImportError("BigVGAN is not available. Initialize the submodule first.") from exc

        if is_local:
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        return vocoder.eval().to(device)

    raise ValueError(f"Unsupported vocoder_name: {vocoder_name}")


def initialize_asr_pipeline(device: str = device, dtype=None):
    global asr_pipe
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 6
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    kwargs = {"task": "transcribe"}
    if language:
        kwargs["language"] = language
    return asr_pipe(ref_audio, chunk_length_s=30, batch_size=128, generate_kwargs=kwargs, return_timestamps=False)[
        "text"
    ].strip()


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 6
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = Path(ckpt_path).suffix.lstrip(".")
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}

    for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
        checkpoint["model_state_dict"].pop(key, None)

    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if load_result.missing_keys:
        print("Missing checkpoint keys:", load_result.missing_keys)
    if load_result.unexpected_keys:
        print("Unexpected checkpoint keys:", load_result.unexpected_keys)

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch detected. "
            "Pass the correct --model_cfg / --model / --vocab_file, or use a matching checkpoint."
        )

    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model.to(device)


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if not vocab_file:
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab_ko.txt"))
    tokenizer = "custom"

    print("\nUsing vocab:", vocab_file)
    print("Tokenizer  :", tokenizer)
    print("Checkpoint :", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)

    model_cfg = dict(model_cfg)
    valid_keys = set(inspect.signature(model_cls.__init__).parameters.keys())
    valid_keys.discard("self")
    filtered_model_cfg = {k: v for k, v in model_cfg.items() if k in valid_keys}
    ignored_keys = sorted(set(model_cfg.keys()) - set(filtered_model_cfg.keys()))
    if ignored_keys:
        print("Ignoring model config keys not accepted by", model_cls.__name__, ":", ignored_keys)

    model = CFM(
        transformer=model_cls(**filtered_model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(method=ode_method),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    return load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

def remove_silence_edges(audio, silence_threshold=-42):
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    return audio[: int(non_silent_end_duration * 1000)]


def preprocess_ref_audio_text(ref_audio_orig, ref_text, clip_short=True, show_info=print, device=device):
    show_info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                    show_info("Audio is over 15s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            if len(non_silent_wave) > 12000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                        show_info("Audio is over 15s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave
            if len(aseg) > 12000:
                aseg = aseg[:12000]
                show_info("Audio is over 15s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    with open(ref_audio, "rb") as audio_file:
        audio_hash = hashlib.md5(audio_file.read()).hexdigest()

    global _ref_audio_cache
    if not ref_text.strip():
        if audio_hash in _ref_audio_cache:
            show_info("Using cached reference text...")
            ref_text = _ref_audio_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            _ref_audio_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        ref_text = ref_text + (" " if ref_text.endswith(".") else ". ")

    print("\nref_text ", ref_text)
    return ref_audio, ref_text


def _get_resampler(source_sr: int):
    if source_sr not in _resampler_cache:
        _resampler_cache[source_sr] = torchaudio.transforms.Resample(source_sr, target_sample_rate)
    return _resampler_cache[source_sr]


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    audio, sr = torchaudio.load(ref_audio)
    audio_seconds = max(audio.shape[-1] / sr, 1e-6)
    approx_chars = len(ref_text.encode("utf-8")) / audio_seconds
    max_chars = max(30, int(approx_chars * max(1.0, 22 - audio_seconds)))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)

    for i, chunk in enumerate(gen_text_batches):
        print(f"gen_text {i}: {chunk}")
    print()

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
        )
    )


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / max(rms, torch.tensor(1e-8))
    if sr != target_sample_rate:
        audio = _get_resampler(sr)(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = 0.3 if len(gen_text.encode("utf-8")) < 10 else speed
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin_orig(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            ref_text_len = max(len(ref_text.encode("utf-8")), 1)
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :].permute(0, 2, 1)

            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            else:
                raise ValueError(f"Unsupported mel_spec_type: {mel_spec_type}")

            if rms < target_rms:
                generated_wave = generated_wave * rms / max(target_rms, 1e-8)

            generated_wave = generated_wave.squeeze().cpu().numpy()
            generated_cpu = generated[0].cpu().numpy()

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j : j + chunk_size], target_sample_rate
            else:
                yield generated_wave, generated_cpu

    if streaming:
        iterator = progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches
        for gen_text in iterator:
            yield from process_batch(gen_text)
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
        iterator = progress.tqdm(futures) if progress is not None else futures
        for future in iterator:
            result = future.result()
            if result:
                generated_wave, generated_mel_spec = next(result)
                generated_waves.append(generated_wave)
                spectrograms.append(generated_mel_spec)

    if not generated_waves:
        yield None, target_sample_rate, None
        return

    if cross_fade_duration <= 0:
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
            final_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )

    combined_spectrogram = np.concatenate(spectrograms, axis=1) if spectrograms else None
    yield final_wave, target_sample_rate, combined_spectrogram


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    non_silent_wave.export(filename, format="wav")


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
