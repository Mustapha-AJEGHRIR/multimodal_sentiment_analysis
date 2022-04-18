import pickle
from torch.utils.data import Dataset
import os

AUDIO = b"covarep"
VISUAL = b"facet"
TEXT = b"glove"
LABEL = b"label"

TRAIN = b"train"
VALID = b"valid"
TEST = b"test"


def load_iemocap(data_path, split="all"):
    # parse the input args
    class IEMOCAP(Dataset):
        """
        PyTorch Dataset for IEMOCAP, don't need to change this
        """

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    # iemocap_data = pickle.load(open(os.path.join(data_path, "iemocap_data.pkl"), "rb"), encoding="latin1")
    iemocap_data = pickle.load(
        open(os.path.join(data_path, "iemocap/IEMOCAP_features_raw.pkl"), "rb"), encoding="latin1"
    )
    audio = iemocap_data[AUDIO]
    visual = iemocap_data[VISUAL]
    text = iemocap_data[TEXT]
    labels = iemocap_data[LABEL]
    train_idx = iemocap_data[TRAIN]
    valid_idx = iemocap_data[VALID]
    test_idx = iemocap_data[TEST]

    if split == "train":
        return IEMOCAP(audio[train_idx, :], visual[train_idx, :], text[train_idx, :, :], labels[train_idx])
    elif split == "valid":
        return IEMOCAP(audio[valid_idx, :], visual[valid_idx, :], text[valid_idx, :, :], labels[valid_idx])
    elif split == "test":
        return IEMOCAP(audio[test_idx, :], visual[test_idx, :], text[test_idx, :, :], labels[test_idx])

    iemocap_train, iemocap_valid, iemocap_test = (
        iemocap_data[split][TRAIN],
        iemocap_data[split][VALID],
        iemocap_data[split][TEST],
    )

    train_audio, train_visual, train_text, train_labels = (
        iemocap_train[AUDIO],
        iemocap_train[VISUAL],
        iemocap_train[TEXT],
        iemocap_train[LABEL],
    )
    valid_audio, valid_visual, valid_text, valid_labels = (
        iemocap_valid[AUDIO],
        iemocap_valid[VISUAL],
        iemocap_valid[TEXT],
        iemocap_valid[LABEL],
    )
    test_audio, test_visual, test_text, test_labels = (
        iemocap_test[AUDIO],
        iemocap_test[VISUAL],
        iemocap_test[TEXT],
        iemocap_test[LABEL],
    )

    # code that instantiates the Dataset objects
    train_set = IEMOCAP(train_audio, train_visual, train_text, train_labels)
    valid_set = IEMOCAP(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = IEMOCAP(test_audio, test_visual, test_text, test_labels)

    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims


if __name__ == "__main__":
    DATADIR = "/mnt/d/Nouamane/Desktop/iemocap"
    train_set, valid_set, test_set, input_dims = load_iemocap(DATADIR, "happy")
    print(train_set[0][0].shape)
    print(train_set[0][1].shape)
    print(train_set[0][2].shape)
    print(train_set[0][3])
