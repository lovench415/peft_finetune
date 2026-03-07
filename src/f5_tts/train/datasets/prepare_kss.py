import os
import sys

sys.path.append(os.getcwd())

import random
import csv
from pathlib import Path
import argparse
import json
import shutil
from importlib.resources import files
from tqdm import tqdm
import torchaudio
from datasets.arrow_writer import ArrowWriter
from f5_tts.model.utils import convert_char_to_pinyin_orig


# Korean Single Speaker (KSS Dataset)
# download from : https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
PRETRAINED_VOCAB_PATH = Path("/workspace/PEFT-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt")
NEW_VOCAB_PATH = Path("/workspace/PEFT-TTS/data/vocab_ko.txt")


def is_csv_wavs_format(input_dataset_dir):
    fpath = Path(input_dataset_dir)
    metadata = fpath / "metadata.csv"
    wavs = fpath / "wavs"
    return metadata.exists() and metadata.is_file() and wavs.exists() and wavs.is_dir()

# converting .txt to .csv
def convert_txt_to_csv(txt_file, csv_file):
    print(f"Converting {txt_file} to {csv_file}...")
    with open(txt_file, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter="|")  # Use | as the delimiter
        csv_writer.writerow(["audio_file", "text"])  # Add header
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                audio_file = "wavs/" + parts[0]  # Add 'wavs/' before the audio file
                text = parts[2]
                csv_writer.writerow([audio_file, text])

def ensure_metadata_csv_exists(input_dir):
    input_dir = Path(input_dir)
    metadata_path = input_dir / "metadata.csv"
    txt_path = input_dir / "metadata.txt"

    if metadata_path.exists():
        print(f"Found existing CSV file: {metadata_path}. No conversion needed.")
        return metadata_path
    elif txt_path.exists():
        print(f"No CSV found. Converting TXT to CSV...")
        convert_txt_to_csv(txt_path, metadata_path)
        return metadata_path
    else:
        raise FileNotFoundError(f"No metadata.csv or metadata.txt found in {input_dir}.")

def prepare_csv_wavs_dir(input_dir):
    input_dir = Path(input_dir)
    metadata_path = ensure_metadata_csv_exists(input_dir)
    audio_path_text_pairs = read_audio_text_pairs(metadata_path.as_posix())

    sub_result, durations, vocab_set = [], [], set()
    polyphone = True
    for audio_path, text in audio_path_text_pairs:
        if not Path(audio_path).exists():
            print(f"Audio {audio_path} not found, skipping.")
            continue
        audio_duration = get_audio_duration(audio_path)

        text = convert_char_to_pinyin_orig([text], polyphone=polyphone)[0]
        print("".join(text))
        sub_result.append({"audio_path": audio_path, "text": text, "duration": audio_duration})
        durations.append(audio_duration)
        vocab_set.update(list(text))# updating vocab

    return sub_result, durations, vocab_set

def get_audio_duration(audio_path):
    audio, sample_rate = torchaudio.load(audio_path)
    print(f"sample_rate  ={sample_rate}")
    return audio.shape[1] / sample_rate

def read_audio_text_pairs(csv_file_path):
    audio_text_pairs = []
    parent = Path(csv_file_path).parent
    with open(csv_file_path, mode="r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) >= 2:
                audio_file = row[0].strip()
                text = row[1].strip()
                audio_file_path = parent / audio_file
                audio_text_pairs.append((audio_file_path.as_posix(), text))
    return audio_text_pairs

def save_prepped_dataset(out_dir, result, duration_list, text_vocab_set, is_finetune, add_vocab, tokenizer_path):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=1) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    voca_out_path = out_dir / "vocab.txt"
    if is_finetune:
        file_vocab_finetune = tokenizer_path.as_posix()
        shutil.copy2(file_vocab_finetune, voca_out_path)
        if add_vocab:
            # TODO : compare pretrained vocab.txt and text_vocab_set
            with open(file_vocab_finetune, "r", encoding='utf-8') as f:
                pretrained_vocab_set = set(f.read().splitlines())

            new_vocab_set = text_vocab_set - pretrained_vocab_set
            
            print(f"new {len(new_vocab_set)}vocabs found : {new_vocab_set}")
            # TODO : add new vocab
            if new_vocab_set :
                with open(voca_out_path, "a", encoding='utf-8') as f:
                    for vocab in sorted(new_vocab_set):
                        f.write(vocab + "\n")



    else:
        with open(voca_out_path, "w") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")

    dataset_name = out_dir.stem
    print(f"\nFor {dataset_name}, train sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list)/3600:.2f} hours")

def save_test_lst(lst_dir, result):

    lst_dir = Path(lst_dir)
    lst_dir.mkdir(exist_ok=True, parents=True)
    lst_path = lst_dir / "KSS_test_200_metadata.lst"
    print(f"\nSaving to {lst_dir} ...")

    with open(lst_path, "w", encoding='utf-8') as lst_file:

        for i in range(0, len(result), 2):  
            sample1 = result[i]
            sample2 = result[i + 1] if i + 1 < len(result) else {}  

            lst_file.write(sample1["audio_path"] + "|" + str(sample1["duration"]) + "|" + ''.join(sample1["text"]) + "|")
            lst_file.write(sample2["audio_path"] + "|" + str(sample2["duration"]) + "|" + ''.join(sample2["text"] )+ "\n")


    print(f"\nFor KSS, test sample count : {len(result)}, {len(result)/2} rows")


def prepare_and_save_set(inp_dir, out_dir, lst_dir, tokenizer_path, is_finetune=True, add_vocab = True, test_count = 200):
    if is_finetune:
        assert tokenizer_path.exists(), f"Pretrained vocab.txt not found: {tokenizer_path}"
    sub_result, durations, vocab_set = prepare_csv_wavs_dir(inp_dir)

    paired_data = list(zip(sub_result, durations))
    random.shuffle(paired_data)

    test_pairs = paired_data[:test_count]
    train_pairs = paired_data[test_count:]

    test_samples, test_durations = zip(*test_pairs)
    train_samples, train_durations = zip(*train_pairs)

    save_test_lst(lst_dir, test_samples)
    save_prepped_dataset(out_dir, train_samples, train_durations, vocab_set, is_finetune, add_vocab, tokenizer_path)

def cli():
    input_path = "/workspace/PEFT-TTS/KSS"
    output_path = "/workspace/PEFT-TTS/data/KSS_pinyin"
    lst_path = "/workspace/PEFT-TTS/data"
    print(input_path)
    print(output_path)
    print(lst_path)
    parser = argparse.ArgumentParser(description="Prepare and save dataset.")
    parser.add_argument("--inp_dir", type=str, default = input_path, help="Input directory containing the data.")
    parser.add_argument("--out_dir", type=str, default = output_path, help="Output directory to save the prepared data.")
    parser.add_argument("--lst_dir", type=str, default = lst_path, help =".lst directory to save the test_meta.lst")
    parser.add_argument("--pretrain", action="store_true", help="Enable for new pretrain, otherwise is a fine-tune")
    parser.add_argument("--add_vocab", type=bool, default =None, help= "Add new vocabs of your own datasets not found in pre-trained vocab.txt.")
    parser.add_argument("--test_count", type=int, default= 200, help= "Amount of samples you want to split for you test set")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to custom tokenizer vocab file (only used if tokenizer = 'custom')",
    )

    args = parser.parse_args()
    prepare_and_save_set(inp_dir = args.inp_dir, out_dir = args.out_dir, lst_dir = args.lst_dir, is_finetune = not args.pretrain, add_vocab = args.add_vocab, test_count = args.test_count, tokenizer_path = args.tokenizer_path)

if __name__ == "__main__":

    cli()

"""
For KSS, test sample count : 200, 100.0 rows
For KSS_pinyin, train sample count: 12653
For KSS_pinyin, vocab size is: 106
For KSS_pinyin, total 12.65 hours
"""
# $ python3 prepare_kss.py --add_vocab True --test_count 200