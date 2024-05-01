import argparse
import glob
import math
import os
import re

import torch
import torchvision
import whisper
from tqdm import tqdm
from transforms import TextTransform

parser = argparse.ArgumentParser(description="Transcribe into text from media")
parser.add_argument(
    "--audio-dir",
    type=str,
    required=True,
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing)",
)
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing)",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel",
)
args = parser.parse_args()

# Constants
chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"]"


text_transform = TextTransform()

# Load video files
all_files = sorted(glob.glob(os.path.join(args.audio_dir, "**", "*.wav"), recursive=True))
print(len(all_files))
unit = math.ceil(len(all_files) / args.groups)
files_to_process = all_files[args.job_index * unit : (args.job_index + 1) * unit]

# Load ASR model
model = whisper.load_model("tiny.en", device=f'cuda:{str(args.device)}')

# Transcription
for filename in tqdm(files_to_process):
    # Prepare destination filename
    dst_filename = filename.replace('aud_dir','txt_dir')[:-4] + ".txt"
    os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
    try:
        with torch.no_grad():
            result = model.transcribe(filename)
            transcript = (
                re.sub(chars_to_ignore_regex, "", result["text"])
                .upper()
                .replace("’", "'")
            )
            transcript = " ".join(transcript.split())
    except RuntimeError:
        continue

    # Write transcript to a text file
    if transcript:
        with open(dst_filename, "w") as k:
            k.write(f'Text:  {transcript}\nConf:  5')
    else:
        with open(dst_filename, "w") as k:
            k.write(f'Text:  {transcript}\nConf:  5')
