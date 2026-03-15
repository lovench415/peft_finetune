import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import modelscope_studio.components.antd as antd
import modelscope_studio.components.base as ms

from f5_tts.infer.infer_cli import (
    _default_vocoder_local_path,
    _normalize_model_name,
    _resolve_model,
)
from f5_tts.infer.utils_infer import (
    cfg_strength as default_cfg_strength,
    cross_fade_duration as default_cross_fade_duration,
    fix_duration as default_fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    nfe_step as default_nfe_step,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    speed as default_speed,
    sway_sampling_coef as default_sway_sampling_coef,
    target_rms as default_target_rms,
)

APP_TITLE = "PEFT-TTS · ModelScope Studio GUI"
DEFAULT_MODEL = "PEFT-TTS_base"
DEFAULT_OUTPUT_DIR = Path("gui_outputs")

_RUNTIME_CACHE: dict[str, Any] = {
    "signature": None,
    "model": None,
    "vocoder": None,
}


CSS = """
footer {display:none !important}
#app-header h1 {margin-bottom: 0.25rem;}
#app-header p {opacity: 0.8; margin: 0;}
"""


def _make_signature(model_name: str, ckpt_file: str, vocab_file: str, vocoder_name: str, local_vocoder: bool) -> tuple:
    return (model_name, ckpt_file, vocab_file, vocoder_name, bool(local_vocoder))


def _ensure_runtime(model_name: str, ckpt_file: str, vocab_file: str, vocoder_name: str, local_vocoder: bool):
    normalized_model = _normalize_model_name(model_name)
    signature = _make_signature(normalized_model, ckpt_file, vocab_file, vocoder_name, local_vocoder)

    if _RUNTIME_CACHE["signature"] == signature and _RUNTIME_CACHE["model"] is not None and _RUNTIME_CACHE["vocoder"] is not None:
        return _RUNTIME_CACHE["model"], _RUNTIME_CACHE["vocoder"], normalized_model

    _, model_cls, model_cfg = _resolve_model(normalized_model, explicit_model_cfg=None)
    model = load_model(
        model_cls=model_cls,
        model_cfg=model_cfg,
        ckpt_path=ckpt_file,
        mel_spec_type=vocoder_name,
        vocab_file=vocab_file,
    )
    vocoder = load_vocoder(
        vocoder_name=vocoder_name,
        is_local=local_vocoder,
        local_path=_default_vocoder_local_path(vocoder_name),
    )

    _RUNTIME_CACHE.update(signature=signature, model=model, vocoder=vocoder)
    return model, vocoder, normalized_model


def _save_outputs(final_wave, sample_rate, spectrogram, output_dir: str, prefix: str, remove_silence: bool):
    target_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = target_dir / f"{prefix}_{stamp}.wav"
    png_path = target_dir / f"{prefix}_{stamp}_mel.png"
    npy_path = target_dir / f"{prefix}_{stamp}_mel.npy"

    import soundfile as sf
    import numpy as np

    sf.write(wav_path, final_wave, sample_rate)
    mel_plot = None
    mel_npy = None
    if spectrogram is not None:
        np.save(npy_path, spectrogram)
        save_spectrogram(spectrogram, png_path)
        mel_plot = str(png_path)
        mel_npy = str(npy_path)

    if remove_silence:
        remove_silence_for_generated_wav(str(wav_path))

    return str(wav_path), mel_npy, mel_plot


def run_single(
    model_name,
    ckpt_file,
    vocab_file,
    vocoder_name,
    local_vocoder,
    ref_audio,
    ref_text,
    gen_text,
    output_dir,
    remove_silence,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
):
    if not ckpt_file:
        raise gr.Error("Укажи путь к checkpoint (.pt или .safetensors).")
    if not vocab_file:
        raise gr.Error("Укажи путь к vocab-файлу.")
    if ref_audio is None:
        raise gr.Error("Загрузи эталонное аудио.")
    if not gen_text or not gen_text.strip():
        raise gr.Error("Введи текст для синтеза.")

    model, vocoder, normalized_model = _ensure_runtime(model_name, ckpt_file, vocab_file, vocoder_name, local_vocoder)
    ref_audio_path, normalized_ref_text = preprocess_ref_audio_text(ref_audio, ref_text or "")
    final_wave, sample_rate, spectrogram = infer_process(
        ref_audio=ref_audio_path,
        ref_text=normalized_ref_text,
        gen_text=gen_text,
        model_obj=model,
        vocoder=vocoder,
        mel_spec_type=vocoder_name,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
    )
    if final_wave is None:
        raise gr.Error("Инференс не вернул аудио.")

    wav_path, mel_npy, mel_plot = _save_outputs(
        final_wave,
        sample_rate,
        spectrogram,
        output_dir,
        prefix="single",
        remove_silence=remove_silence,
    )

    status = (
        f"Готово. Model: {normalized_model}\n"
        f"Checkpoint: {ckpt_file}\n"
        f"Vocoder: {vocoder_name}\n"
        f"Нормализованный ref_text: {normalized_ref_text}\n"
        f"WAV: {wav_path}"
    )
    return wav_path, mel_plot, mel_npy, normalized_ref_text, status


MULTI_VOICE_HELP = {
    "voices": {
        "main": {"ref_audio": "path/to/main.wav", "ref_text": "Основной голос."},
        "alice": {"ref_audio": "path/to/alice.wav", "ref_text": "Голос Алисы."},
        "bob": {"ref_audio": "path/to/bob.wav", "ref_text": "Голос Боба."},
    },
    "script": "[alice] Привет! [bob] Здравствуй! [main] Это основной голос.",
}


def run_multi(
    model_name,
    ckpt_file,
    vocab_file,
    vocoder_name,
    local_vocoder,
    voices_json,
    script,
    output_dir,
    remove_silence,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
):
    if not ckpt_file:
        raise gr.Error("Укажи путь к checkpoint (.pt или .safetensors).")
    if not vocab_file:
        raise gr.Error("Укажи путь к vocab-файлу.")
    if not script or not script.strip():
        raise gr.Error("Введи script с тегами голосов.")

    try:
        payload = json.loads(voices_json)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Невалидный JSON: {exc}") from exc

    voices = payload.get("voices") if isinstance(payload, dict) else None
    if not voices or not isinstance(voices, dict):
        raise gr.Error("JSON должен содержать объект voices.")

    model, vocoder, normalized_model = _ensure_runtime(model_name, ckpt_file, vocab_file, vocoder_name, local_vocoder)

    import re
    import numpy as np

    prepared_voices = {}
    for name, cfg in voices.items():
        audio_path = cfg.get("ref_audio")
        if not audio_path:
            raise gr.Error(f"У голоса '{name}' отсутствует ref_audio.")
        ref_audio_path, normalized_ref_text = preprocess_ref_audio_text(audio_path, cfg.get("ref_text", ""))
        prepared_voices[name] = {"ref_audio": ref_audio_path, "ref_text": normalized_ref_text}

    pattern = r"(?=\[\w+\])"
    tag_pattern = r"\[(\w+)\]"
    segments = []
    spectrograms = []
    normalized_parts = []
    sample_rate = 24000

    for chunk in re.split(pattern, script):
        if not chunk.strip():
            continue
        match = re.match(tag_pattern, chunk)
        if match:
            voice_name = match.group(1)
            text = re.sub(tag_pattern, "", chunk).strip()
        else:
            voice_name = "main"
            text = chunk.strip()
        if not text:
            continue
        if voice_name not in prepared_voices:
            raise gr.Error(f"Голос '{voice_name}' не найден в JSON.")

        voice = prepared_voices[voice_name]
        audio_part, sample_rate, spectrogram = infer_process(
            ref_audio=voice["ref_audio"],
            ref_text=voice["ref_text"],
            gen_text=text,
            model_obj=model,
            vocoder=vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
        )
        if audio_part is not None:
            segments.append(audio_part)
            normalized_parts.append(f"[{voice_name}] {text}")
            if spectrogram is not None:
                spectrograms.append(spectrogram)

    if not segments:
        raise gr.Error("Инференс не вернул аудио для multi-voice сценария.")

    final_wave = np.concatenate(segments)
    combined_spec = np.concatenate(spectrograms, axis=1) if spectrograms else None
    wav_path, mel_npy, mel_plot = _save_outputs(
        final_wave,
        sample_rate,
        combined_spec,
        output_dir,
        prefix="multi",
        remove_silence=remove_silence,
    )
    status = (
        f"Готово. Model: {normalized_model}\n"
        f"Checkpoint: {ckpt_file}\n"
        f"Vocoder: {vocoder_name}\n"
        f"Сегментов: {len(segments)}\n"
        f"WAV: {wav_path}"
    )
    return wav_path, mel_plot, mel_npy, "\n".join(normalized_parts), status


DEFAULT_MULTI_JSON = json.dumps(MULTI_VOICE_HELP, ensure_ascii=False, indent=2)


def build_demo():
    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
        with ms.Application(), antd.ConfigProvider():
            gr.HTML(
                """
                <div id='app-header'>
                  <h1>PEFT-TTS · ModelScope Studio GUI</h1>
                  <p>Single-speaker and tagged multi-voice inference for the refactored PEFT-TTS project.</p>
                </div>
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    model_name = gr.Dropdown(
                        choices=["PEFT-TTS_base", "PEFT-TTS_v1", "F5TTS_Base", "E2TTS_Base"],
                        value=DEFAULT_MODEL,
                        label="Model",
                    )
                    ckpt_file = gr.Textbox(label="Checkpoint path", placeholder=r"F:\python\petf_tts\ckpts\KSS\PEFT-TTS_base\model_355000.pt")
                    vocab_file = gr.Textbox(label="Vocab path", placeholder=r"F:\python\petf_tts\data\KSS_pinyin\vocab_ko.txt")
                    vocoder_name = gr.Dropdown(choices=["vocos", "bigvgan"], value="vocos", label="Vocoder")
                    local_vocoder = gr.Checkbox(value=False, label="Load vocoder from local checkpoint folder")
                    output_dir = gr.Textbox(label="Output directory", value=str(DEFAULT_OUTPUT_DIR))
                    remove_silence = gr.Checkbox(value=False, label="Remove silence in final WAV")

                    with gr.Accordion("Advanced generation settings", open=False):
                        target_rms = gr.Slider(minimum=0.01, maximum=0.5, value=default_target_rms, step=0.01, label="Target RMS")
                        cross_fade_duration = gr.Slider(minimum=0.0, maximum=1.0, value=default_cross_fade_duration, step=0.01, label="Cross-fade duration")
                        nfe_step = gr.Slider(minimum=4, maximum=96, value=default_nfe_step, step=1, label="NFE steps")
                        cfg_strength = gr.Slider(minimum=0.0, maximum=5.0, value=default_cfg_strength, step=0.1, label="CFG strength")
                        sway_sampling_coef = gr.Slider(minimum=-2.0, maximum=2.0, value=default_sway_sampling_coef, step=0.1, label="Sway sampling coef")
                        speed = gr.Slider(minimum=0.3, maximum=2.0, value=default_speed, step=0.05, label="Speed")
                        fix_duration = gr.Number(value=default_fix_duration, label="Fix duration (seconds, optional)")

                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.Tab("Single voice"):
                            ref_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference audio")
                            ref_text = gr.Textbox(lines=3, label="Reference text (optional)")
                            gen_text = gr.Textbox(lines=8, label="Text to synthesize")
                            run_single_btn = gr.Button("Generate single-voice speech", variant="primary")

                        with gr.Tab("Tagged multi voice"):
                            voices_json = gr.Code(value=DEFAULT_MULTI_JSON, language="json", label="Voices JSON")
                            script = gr.Textbox(
                                lines=8,
                                label="Script with [voice] tags",
                                value="[alice] Привет! [bob] Здравствуй! [main] Это основной голос.",
                            )
                            run_multi_btn = gr.Button("Generate multi-voice speech", variant="primary")

                    normalized_text = gr.Textbox(lines=6, label="Normalized / effective text", interactive=False)
                    status = gr.Textbox(lines=6, label="Status", interactive=False)
                    audio_out = gr.Audio(type="filepath", label="Generated audio")
                    mel_plot = gr.Image(type="filepath", label="Mel spectrogram")
                    mel_npy = gr.File(label="Mel numpy file")

            run_single_btn.click(
                fn=run_single,
                inputs=[
                    model_name,
                    ckpt_file,
                    vocab_file,
                    vocoder_name,
                    local_vocoder,
                    ref_audio,
                    ref_text,
                    gen_text,
                    output_dir,
                    remove_silence,
                    target_rms,
                    cross_fade_duration,
                    nfe_step,
                    cfg_strength,
                    sway_sampling_coef,
                    speed,
                    fix_duration,
                ],
                outputs=[audio_out, mel_plot, mel_npy, normalized_text, status],
            )

            run_multi_btn.click(
                fn=run_multi,
                inputs=[
                    model_name,
                    ckpt_file,
                    vocab_file,
                    vocoder_name,
                    local_vocoder,
                    voices_json,
                    script,
                    output_dir,
                    remove_silence,
                    target_rms,
                    cross_fade_duration,
                    nfe_step,
                    cfg_strength,
                    sway_sampling_coef,
                    speed,
                    fix_duration,
                ],
                outputs=[audio_out, mel_plot, mel_npy, normalized_text, status],
            )

    return demo


def main():
    demo = build_demo()
    demo.queue().launch()


if __name__ == "__main__":
    main()
