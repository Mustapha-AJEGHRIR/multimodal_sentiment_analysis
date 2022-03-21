# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
# ----------------------------- Datascience stuff ---------------------------- #
# import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# -------------------------- Signal and media stuff -------------------------- #
import scipy
from scipy.io import wavfile
import scipy.signal
import cv2

# ----------------------------------- Other ---------------------------------- #
import os
import h5py
import json
from tqdm import tqdm


# ---------------------------------- Warning --------------------------------- #
print("This version of the dataloader is loading :")
print("\t\t-Audio for each sequence")
print("\t\t-Label for each sequence")
print("Future versions might include :")
print("\t\t-Video for each sequence")
print("\t\t-text for each sequence")


# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #
F16 = np.float16
F32 = np.float32
F64 = np.float64
FTYPE = F32

I8 = np.uint8 # unsigned integer 8 bits

TRAIN_SPLIT = 0.8
BATCH_SIZE = 4

# ---------------------------------------------------------------------------- #
#                           Verify data availability                           #
# ---------------------------------------------------------------------------- #
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/iemocap/')
AUDIO_PATH = os.path.join(DATA_PATH, 'session1-sentences-wav')
LABELS_PATH = os.path.join(DATA_PATH, '')

if not os.path.exists(DATA_PATH):
    raise Exception("Data path does not exist, donwload the data using the 'data/get_CMU_MOSI.sh' script")
else :
    for path in [VIDEO_PATH, LABELS_PATH]:
        if not os.path.exists(path):
            raise Exception("Data not correctly downloaded, please follow the correct instructions for 'data/get_CMU_MOSI.sh' script")



# ---------------------------------------------------------------------------- #
#                                     Utils                                    #
# ---------------------------------------------------------------------------- #

def load_audio(audio_path):
    sample_rate, audio = scipy.io.wavfile.read(os.path.join(AUDIO_PATH, audio_path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 2**15
    elif audio.dtype != np.float32:
        raise ValueError('Unexpected datatype. Model expects sound samples to lie in [-1, 1]')
    if len(audio.shape) == 2: # stereo
        audio = audio.mean(axis=1)
    return audio, sample_rate
# ---------------------------------------------------------------------------- #
#                            Define main dataloader                            #
# ---------------------------------------------------------------------------- #

class IEMOCAP(Dataset):
    def __init__(self,
                split = None, # Could be "training" or "val"
                train_split = TRAIN_SPLIT,
                output_type = FTYPE,
                debugging = False,
                annotators = 0, # 0 for all, 1 for annotator 1, 2 for annotator 2, 3 for annotator 3
                **kwargs
                ):
        self.split = split
        self.annotators = annotators
        self.train_split = train_split
        self.devide_images_with = 255.0
        self.output_type = output_type

        
        self.sequence_names, self.sequence_labels = extract_labels()
        if debugging :
            self.sequence_labels = self.sequence_labels[:50]
            self.sequence_names = self.sequence_names[:50]
        train_sequence_names, val_sequence_names, train_sequence_labels, val_sequence_labels = train_test_split(self.sequence_names, self.sequence_labels, train_size=train_split, random_state=42)
        if self.split == "training":
            self.sequence_names = train_sequence_names
            self.sequence_labels = train_sequence_labels
        elif self.split == "validation":
            self.sequence_names = val_sequence_names
            self.sequence_labels = val_sequence_labels
        elif self.split == None:
            pass
        else :
            raise Exception("Unknown split name, please use : 'training', 'validation' or None")

        self.video_sequences = []
        self.audio_sequences = []
        for seq_name in tqdm(self.sequence_names, desc="Caching videos for " + str(self.split) + " split"):
            frames = load_video(seq_name + ".mp4", max_frames=self.max_frames, resize=self.resize, color_space=self.color_space)
            # print("frames: ", frames.max(), frames.min())
            self.video_sequences.append(frames)

            audio, sample_rate = load_audio(seq_name + ".wav")
            assert sample_rate == 16000 # FIXME : remove me later, note efficient !
            self.audio_sequences.append(audio)
        
        
    def __getitem__(self, index):
        video_seq = self.video_sequences[index] / self.devide_images_with
        return (video_seq.astype(self.output_type),
                self.audio_sequences[index].astype(self.output_type),
                self.sequence_labels[index])

    def __len__(self):
        return len(self.sequence_names)


def get_train_val(**kwargs):
    if "BATCH_SIZE" in kwargs:
        batch_size = kwargs["BATCH_SIZE"]
    else :
        batch_size = BATCH_SIZE

    train = CMU_MOSI(split="training", **kwargs)
    val = CMU_MOSI(split="validation", **kwargs)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader
    



if  __name__ == "__main__":
    print("Debuging ...")
    dataset = CMU_MOSI()
    print("Dataset loaded")