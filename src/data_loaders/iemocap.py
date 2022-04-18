# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
# ----------------------------- Datascience stuff ---------------------------- #

# from torch.utils.data import Dataset, DataLoader
import torch.utils.data as tdata

# from sklearn.model_selection import train_test_split
import numpy as np
import collections
import pandas as pd

# from datasets import load_dataset
# import torchaudio

# -------------------------- Signal and media stuff -------------------------- #
import scipy

# from scipy.io import wavfile
import scipy.signal

# import cv2

# ----------------------------------- Other ---------------------------------- #
import os
from glob import glob

# import json
import regex as re

# from tqdm import tqdm

# --------------------------- Cross platform stuff --------------------------- #
# from dotenv import load_dotenv
# load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env_consts'))


# ---------------------------------- Warning --------------------------------- #
print("This version of the dataloader is loading :")
print("\t\t-Label for each sequence")
print("\t\t-Audio Path for each sequence")
print("\t\t-Start and end time for each sequence")
print("\t\t-text for each sequence")
print("Future versions might include :")
print("\t\t-Audio for each sequence")
print("\t\t-Video Path for each sequence")


# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #
F16 = np.float16
F32 = np.float32
F64 = np.float64
FTYPE = F32

I8 = np.uint8  # unsigned integer 8 bits

TRAIN_SPLIT = 0.8
BATCH_SIZE = 4

# ---------------------------------------------------------------------------- #
#                           Verify data availability                           #
# ---------------------------------------------------------------------------- #
SAVE_TMP_PATH = "tmp"
# read local variable if they contain data_path
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/")
# DEFAULT_DATA_PATH = "/mnt/d/OneDrive/OneDrive - CentraleSupelec/"  # TODO
DATA_PATH = os.path.join(os.getenv("TMPDIR", DEFAULT_DATA_PATH), "iemocap/")
# DATA_PATH = os.path.join(os.getenv("DATADIR", DEFAULT_DATA_PATH), "iemocap/")
AUDIO_PATH = os.path.join(DATA_PATH, "session1-sentences-wav")
VIDEO_PATH = os.path.join(DATA_PATH, "session1-dialog-avi")
TRANSCRIPT_PATH = os.path.join(DATA_PATH, "session1-dialog-transcriptions")
LABELS_PATH = os.path.join(DATA_PATH, "session1-dialog-EmoEvaluation/Categorical")

if not os.path.exists(DATA_PATH):
    raise Exception(f"Data path {DATA_PATH} does not exist, donwload the data please, it is under licence")
else:
    for path in [AUDIO_PATH, LABELS_PATH]:
        if not os.path.exists(path):
            raise Exception(f"Path {path} does not exist, make sure the data is correctly extracted")


def most_frequent(List):
    occurence_count = collections.Counter(List)
    return occurence_count.most_common(1)[0][0]


def load_audio(audio_path):
    sample_rate, audio = scipy.io.wavfile.read(os.path.join(AUDIO_PATH, audio_path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 2**15
    elif audio.dtype != np.float32:
        raise ValueError("Unexpected datatype. Model expects sound samples to lie in [-1, 1]")
    if len(audio.shape) == 2:  # stereo
        audio = audio.mean(axis=1)
    return audio, sample_rate


# ---------------------------------------------------------------------------- #
#                            Define main dataloader                            #
# ---------------------------------------------------------------------------- #


class IEMOCAP(tdata.Dataset):
    def __init__(
        self,
        output_type=FTYPE,
        debugging=False,
        best_label_only=True,
        return_path=True,
        # annotators = 0, # 0 for all, 1 for annotator 1, 2 for annotator 2, 3 for annotator 3
        **kwargs,
    ):
        if not return_path:
            raise Exception("Not yet implemented")
        self.best_label_only = True
        # self.annotators = annotators
        # self.devide_images_with = 255.0
        self.output_type = output_type

        self.all_labels_names = sorted(glob(os.path.join(LABELS_PATH, "*.txt")))
        self.text_paths = sorted(glob(os.path.join(TRANSCRIPT_PATH, "*.txt")))
        self.audio_paths = sorted(glob(os.path.join(AUDIO_PATH, "*/*.wav")))
        self.video_paths = sorted(glob(os.path.join(VIDEO_PATH, "*/*.avi")))

        self.all_labels = collections.defaultdict(list)

        # self.audio = []
        self.data = {}

        # ------------------------------ Extract labels ------------------------------ #
        for label_file in self.all_labels_names:
            with open(label_file, "r") as f:
                all_lines = f.readlines()
                for line in all_lines:
                    line = line.strip()
                    annotation_count = line.count(":")
                    seq_id = line.split(" ")[0]
                    for i in range(annotation_count):
                        annotation = line.split(":")[i + 1]
                        annotation = annotation.split(";")[0]
                        self.all_labels[seq_id].append(annotation)

                    labelisations = self.all_labels[seq_id]
                    if self.best_label_only:
                        label = most_frequent(labelisations)

                    self.data[seq_id] = {
                        "name": seq_id,
                        "emotion": label,
                    }

        # ----------------------------------- load audio ---------------------------------- #
        for audio_path in self.audio_paths:
            base_name = os.path.basename(audio_path).split(".")[0]
            self.data[base_name] = {
                "audio_path": audio_path,
            }

        # # -------------------------------- load text -------------------------------- #
        for text_path in self.text_paths:
            with open(text_path, "r") as f:
                for line in f.readlines():  # list of "<seq_id> [<start_time>-<end_time>]: Excuse me.\n"
                    line = line.strip()
                    if re.match(r"(\w+)\s+\[([\d\.]+)-([\d\.]+)\]:\s+(.*)", line):
                        seq_id, start_time, end_time, text = re.match(
                            r"(\w+)\s+\[([\d\.]+)-([\d\.]+)\]:\s+(.*)", line
                        ).groups()
                        self.data[seq_id]["start_time"] = start_time
                        self.data[seq_id]["end_time"] = end_time
                        self.data[seq_id]["text"] = text

        # # -------------------------------- load video -------------------------------- #
        # TODO: need to cut dialog videos according to sequence start and end time
        # TODO: check https://github.com/IoannisKansizoglou/Iemocap-preprocess/blob/f2329ac892f1334ac743b985b2751d9b1431b3c8/extractionmapCreator.py
        # for video_path in self.video_paths:
        #     base_name = os.path.basename(video_path).split(".")[0]
        #     self.data[base_name] = {
        #         "video_path": video_path,
        #     }

        # self.video_sequences = []
        # self.audio_sequences = []
        self.indexer = {i: seq_id for i, seq_id in enumerate(self.data.keys())}

    def __getitem__(self, index):
        return self.data[self.indexer[index]]

    def __len__(self):
        return len(self.data)


def get_data(**kwargs):
    dataset = IEMOCAP(**kwargs)
    df = pd.DataFrame(dataset.data.values())
    print("Found labels : ", df.emotion.unique())
    # print(df.head())

    # train_df, test_df = train_test_split(df, test_size=1-TRAIN_SPLIT, random_state=42, stratify=["emotion"])

    # train_df = train_df.reset_index(drop=True)
    # test_df = test_df.reset_index(drop=True)

    # train_df.to_csv(f"{SAVE_TMP_PATH}/train.csv", sep="\t", encoding="utf-8", index=False)
    # test_df.to_csv(f"{SAVE_TMP_PATH}/test.csv", sep="\t", encoding="utf-8", index=False)

    # data_files = {
    #     "train": f"{SAVE_TMP_PATH}/train.csv",
    #     "validation": f"{SAVE_TMP_PATH}/test.csv",
    # }

    # dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["validation"]
    return df


if __name__ == "__main__":
    print("Debuging ...")
    dataset = IEMOCAP()
    print("Dataset loaded")
    df = get_data()
    print("__main__ done")
