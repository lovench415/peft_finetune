import json
from pathlib import Path
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_dataset as hf_load_dataset, load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default



class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self._resamplers = {}

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

        self.valid_indices = []
        for idx in range(len(self.data)):
            row = self.data[idx]
            audio = row["audio"]["array"]
            sample_rate = row["audio"]["sampling_rate"]
            duration = audio.shape[-1] / sample_rate
            if 0.3 <= duration <= 30:
                self.valid_indices.append(idx)

        if not self.valid_indices:
            raise RuntimeError("No valid audio samples found in HFDataset")

    def _resolve_index(self, index):
        return self.valid_indices[index]

    def get_frame_len(self, index):
        row = self.data[self._resolve_index(index)]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return int(audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length)

    def get_speaker(self, index):
        row = self.data[self._resolve_index(index)]
        return row.get("speaker", row.get("speaker_id", "default"))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        row = self.data[self._resolve_index(index)]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        if sample_rate != self.target_sample_rate:
            if sample_rate not in self._resamplers:
                self._resamplers[sample_rate] = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = self._resamplers[sample_rate](audio_tensor)
        mel_spec = self.mel_spectrogram(audio_tensor).squeeze(0)
        return {"mel_spec": mel_spec, "text": row["text"]}



class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel
        self._resamplers = {}

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

        self.valid_indices = []
        for idx in range(len(self.data)):
            row = self.data[idx]
            duration = self.durations[idx] if self.durations is not None else row["duration"]
            if 0.3 <= duration <= 30:
                self.valid_indices.append(idx)

        if not self.valid_indices:
            raise RuntimeError("No valid samples found in CustomDataset")

    def _resolve_index(self, index):
        return self.valid_indices[index]

    def get_frame_len(self, index):
        raw_index = self._resolve_index(index)
        if self.durations is not None:
            return int(self.durations[raw_index] * self.target_sample_rate / self.hop_length)
        return int(self.data[raw_index]["duration"] * self.target_sample_rate / self.hop_length)

    def get_speaker(self, index):
        row = self.data[self._resolve_index(index)]
        if "speaker" in row:
            return row["speaker"]
        if "speaker_id" in row:
            return row["speaker_id"]
        if "audio_path" in row:
            return Path(row["audio_path"]).parent.name
        return "default"

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        row = self.data[self._resolve_index(index)]

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
            text = row["text"]
            return {"mel_spec": mel_spec, "text": text}

        audio_path = row["audio_path"]
        text = row["text"]
        audio, source_sample_rate = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if source_sample_rate != self.target_sample_rate:
            if source_sample_rate not in self._resamplers:
                self._resamplers[source_sample_rate] = torchaudio.transforms.Resample(
                    source_sample_rate, self.target_sample_rate
                )
            audio = self._resamplers[source_sample_rate](audio)

        mel_spec = self.mel_spectrogram(audio).squeeze(0)
        return {"mel_spec": mel_spec, "text": text}


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = None,
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """
    if mel_spec_kwargs is None:
        mel_spec_kwargs = {}


    print("Loading dataset ...")


    preprocessed_mel = False

    if dataset_type == "CustomDataset":
        # BUG-25 FIX: include tokenizer suffix in path (e.g. data/KSS_pinyin)
        rel_data_path = str(files("f5_tts").joinpath(f"../../../data/{dataset_name}_{tokenizer}"))
        print(f"rel_data_path : {rel_data_path}")
        #exit()
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]



        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            hf_load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
            **mel_spec_kwargs,
        )

    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )
