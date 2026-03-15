
import argparse
import re
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf

from f5_tts.config import (
    build_infer_app_config_from_args,
    validate_infer_config,
    resolve_model_cls,
    default_vocoder_local_path,
    print_infer_summary,
)
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)

DEFAULT_MODEL = "PEFT-TTS_base"
VOICE_TAG_SPLIT = r"(?=\[\w+\])"
VOICE_TAG_MATCH = r"\[(\w+)\]"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m f5_tts.infer.infer_cli",
        description="Command-line inference for PEFT-TTS / F5-TTS.",
        epilog="CLI values override config values.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(files("f5_tts").joinpath("infer/examples/single/single.toml")),
        help="Path to TOML config. Optional when using --ckpt_file with saved checkpoint metadata.",
    )
    parser.add_argument("-m", "--model", type=str, default=None,
        help=f"Model name override. Defaults to checkpoint metadata, config, then {DEFAULT_MODEL}.")
    parser.add_argument("-mc", "--model_cfg", type=str, help="Path to YAML model config.")
    parser.add_argument("-p", "--ckpt_file", type=str, help="Path to checkpoint file.")
    parser.add_argument("-v", "--vocab_file", type=str, help="Path to vocab file.")
    parser.add_argument("-r", "--ref_audio", type=str, help="Reference audio file.")
    parser.add_argument("-s", "--ref_text", type=str, help="Reference transcript.")
    parser.add_argument("-t", "--gen_text", type=str, help="Text to synthesize.")
    parser.add_argument("-f", "--gen_file", type=str, help="UTF-8 text file with generation text.")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory.")
    parser.add_argument("-w", "--output_file", type=str, help="Output wav file name.")
    parser.add_argument("--save_chunk", action="store_true", help="Save generated chunks separately.")
    parser.add_argument("--remove_silence", action="store_true", help="Trim long silences from output.")
    parser.add_argument("--load_vocoder_from_local", action="store_true")
    parser.add_argument("--vocoder_name", type=str, choices=["vocos", "bigvgan"])
    parser.add_argument("--target_rms", type=float)
    parser.add_argument("--cross_fade_duration", type=float)
    parser.add_argument("--nfe_step", type=int)
    parser.add_argument("--cfg_strength", type=float)
    parser.add_argument("--sway_sampling_coef", type=float)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--fix_duration", type=float)
    parser.add_argument("-cu", "--ckpt_update", type=int, default=355000)
    parser.add_argument("-nv", "--normalizing_version", type=int)
    return parser




def parse_voice_chunks(gen_text: str):
    for chunk in re.split(VOICE_TAG_SPLIT, gen_text):
        if not chunk.strip():
            continue
        match = re.match(VOICE_TAG_MATCH, chunk)
        if match:
            voice = match.group(1)
            text = re.sub(VOICE_TAG_MATCH, "", chunk).strip()
        else:
            voice = "main"
            text = chunk.strip()
        if text:
            yield voice, text


def build_voices(app_cfg) -> dict:
    raw_config = getattr(app_cfg, "_raw_infer_config", {})
    voices = dict(raw_config.get("voices", {}))
    voices["main"] = {"ref_audio": app_cfg.infer.ref_audio, "ref_text": app_cfg.infer.ref_text}

    for voice_name in list(voices.keys()):
        print(f"Voice: {voice_name}")
        ref_audio, ref_text = preprocess_ref_audio_text(
            voices[voice_name]["ref_audio"],
            voices[voice_name].get("ref_text", ""),
        )
        voices[voice_name]["ref_audio"] = ref_audio
        voices[voice_name]["ref_text"] = ref_text

    return voices


def infer_segments(app_cfg, voices: dict, model, vocoder):
    infer_cfg = app_cfg.infer
    segments = []
    mel_specs = []
    output_chunk_dir = None

    if infer_cfg.save_chunk:
        output_chunk_dir = Path(infer_cfg.output_dir) / f"{Path(infer_cfg.output_file).stem}_chunks"
        output_chunk_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 24000
    for idx, (voice_name, text) in enumerate(parse_voice_chunks(infer_cfg.gen_text)):
        voice_cfg = voices.get(voice_name, voices["main"])
        if voice_name not in voices:
            print(f"Voice '{voice_name}' not found, using main.")

        print(f"Generating chunk {idx} with voice '{voice_name}'")
        audio_segment, sample_rate, spectrogram = infer_process(
            voice_cfg["ref_audio"],
            voice_cfg["ref_text"],
            text,
            model,
            vocoder,
            mel_spec_type=infer_cfg.vocoder_name,
            target_rms=infer_cfg.target_rms,
            cross_fade_duration=infer_cfg.cross_fade_duration,
            nfe_step=infer_cfg.nfe_step,
            cfg_strength=infer_cfg.cfg_strength,
            sway_sampling_coef=infer_cfg.sway_sampling_coef,
            speed=infer_cfg.speed,
            fix_duration=infer_cfg.fix_duration,
        )

        if audio_segment is None:
            continue
        segments.append(audio_segment)
        if spectrogram is not None:
            mel_specs.append(spectrogram)

        if output_chunk_dir is not None:
            safe_name = re.sub(r"[^\w\-]+", "_", text[:80]).strip("_") or f"chunk_{idx}"
            sf.write(output_chunk_dir / f"{idx:03d}_{safe_name}.wav", audio_segment, sample_rate)

    return segments, mel_specs, sample_rate if segments else 24000


def _concat_segments_with_crossfade(audio_segments: list[np.ndarray], sample_rate: int, cross_fade_duration: float) -> np.ndarray:
    if not audio_segments:
        return np.array([], dtype=np.float32)
    if len(audio_segments) == 1:
        return np.asarray(audio_segments[0], dtype=np.float32)

    output = np.asarray(audio_segments[0], dtype=np.float32)
    overlap = max(0, int(sample_rate * max(0.0, cross_fade_duration)))
    for segment in audio_segments[1:]:
        segment = np.asarray(segment, dtype=np.float32)
        current_overlap = min(overlap, len(output), len(segment))
        if current_overlap <= 0:
            output = np.concatenate([output, segment])
            continue
        fade_out = np.linspace(1.0, 0.0, current_overlap, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, current_overlap, dtype=np.float32)
        mixed = output[-current_overlap:] * fade_out + segment[:current_overlap] * fade_in
        output = np.concatenate([output[:-current_overlap], mixed, segment[current_overlap:]])
    return output


def save_outputs(app_cfg, audio_segments: list[np.ndarray], mel_specs: list[np.ndarray], sample_rate: int):
    infer_cfg = app_cfg.infer
    if not audio_segments:
        raise RuntimeError("Inference returned no audio segments.")

    output_dir = Path(infer_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_wave = _concat_segments_with_crossfade(audio_segments, sample_rate, infer_cfg.cross_fade_duration)
    wave_path = output_dir / infer_cfg.output_file
    sf.write(wave_path, final_wave, sample_rate)

    mel_path = None
    if mel_specs:
        try:
            mel_path = output_dir / f"{wave_path.stem}_mel.npy"
            np.save(mel_path, np.concatenate(mel_specs, axis=1))
        except Exception:
            mel_path = output_dir / f"{wave_path.stem}_mel_list.npy"
            np.save(mel_path, np.array(mel_specs, dtype=object), allow_pickle=True)

    if infer_cfg.remove_silence:
        remove_silence_for_generated_wav(str(wave_path))

    return wave_path, mel_path


def main():
    args = build_argparser().parse_args()
    cfg = build_infer_app_config_from_args(args)
    validate_infer_config(cfg)

    infer_cfg = cfg.infer
    print_infer_summary(cfg)

    vocoder = load_vocoder(
        vocoder_name=infer_cfg.vocoder_name,
        is_local=infer_cfg.load_vocoder_from_local,
        local_path=default_vocoder_local_path(infer_cfg.vocoder_name),
    )
    model = load_model(
        resolve_model_cls(cfg.model.backbone),
        cfg.model.to_transformer_kwargs(),
        infer_cfg.ckpt_file,
        mel_spec_type=infer_cfg.vocoder_name,
        vocab_file=infer_cfg.vocab_file,
    )

    voices = build_voices(cfg)
    segments, mel_specs, sample_rate = infer_segments(cfg, voices, model, vocoder)
    wave_path, mel_path = save_outputs(cfg, segments, mel_specs, sample_rate)

    print(f"Saved wav: {wave_path}")
    if mel_path is not None:
        print(f"Saved mel: {mel_path}")


if __name__ == "__main__":
    main()
