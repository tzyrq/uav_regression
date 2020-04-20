from torch.utils.data import Dataset
import numpy as np
import sys


class MaxDatasetTuple(Dataset):
    def __init__(self, input_path, init_max_path, label_max_path, labelSize):
        self.input_path = input_path
        self.init_max_path = init_max_path
        self.label_max_path = label_max_path
        self.labelSize = labelSize

        self.label_max_md = []
        self.init_max_md = []
        self.input_md = []
        self._get_tuple()

    def __len__(self):
        return len(self.label_max_md)

    def _get_tuple(self):
        self.init_max_md = np.load(self.init_max_path).astype(float)
        self.label_max_md = np.load(self.label_max_path).astype(float)
        self.input_md = np.load(self.input_path).astype(float)

    def __getitem__(self, idx):
        try:
            inputs = self._prepare_input(idx)
            init_max = self._prepare_init(idx)
            label_max = self._prepare_label(idx)
            # init_min = np.expand_dims(init_min, axis=0)
        except Exception as e:
            print('error encountered while loading {}'.format(idx))
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
            raise

        return {'input': inputs, 'init_max': init_max, 'label_max': label_max}

    def _prepare_init(self, idx):
        return self.init_max_md[idx]

    def _prepare_input(self, idx):
        return self.input_md[idx]

    def _prepare_label(self, idx):
        return self.label_max_md[idx].reshape(self.labelSize, self.labelSize)



