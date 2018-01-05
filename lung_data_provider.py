import numpy as np
from image_util import BaseDataProvider
import img_processing as proc



class LungDataProvider(BaseDataProvider):
    channels = 1
    classes = 2
    img_list = []
    label_list = []
    img_idx = 0
    n_cur_img = -1
    num_slices = 40
    idx = 0

    def _load_data_and_label(self):
        data, label = self._next_data()

        train_data = self._process_data(data)
        labels = self._process_labels(label)

        #train_data, labels = self._post_process(train_data, labels)

        nz = train_data.shape[1]
        nx = train_data.shape[2]
        ny = train_data.shape[3]

        return train_data.reshape(1, nz, nx, ny, self.channels), labels.reshape(1, nz, nx, ny, self.n_class)

    def _process_labels(self, label):
        if self.n_class == 2:
            nz = label.shape[1]
            nx = label.shape[2]
            ny = label.shape[3]
            labels = np.zeros((1, nz, nx, ny, self.n_class), dtype=np.float32)
            #print(labels.shape)
            labels[..., 1] = label[..., 0]
            labels[..., 0] = 1 - label[..., 0]
            return labels

        return label

    def __call__(self, *args, **kwargs):
        return self._load_data_and_label()

    def __init__(self, img_folder, label_folder, nx=512, ny=512, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        super(BaseDataProvider, self).__init__()
        self.img_list = proc.get_mhd_list(img_folder)
        self.label_list = proc.get_mhd_list(label_folder)
        self.n_examples = len(self.label_list)
        self.nx=nx
        self.ny=ny

    def _next_data(self):
        print("idx = %d"%self.idx)
        if (self.img_idx>=self.n_examples):
            print("No more images left.")
            return
        if (self.n_cur_img<self.idx + self.num_slices):
            img_arr = proc.get_image_array(self.img_list[self.img_idx], normalize=True)
            label_arr = proc.get_image_array(self.label_list[self.img_idx], normalize=True)
            self.idx += 1

        nz = self.num_slices#img_arr.shape[0]
        nx = img_arr.shape[1]
        ny = img_arr.shape[2]

        if (self.nx!=nx or self.ny!=ny):
            print("Warning: Img size doesn't match to the model (nx=%d, ny=%d)."%(nx, ny))

        X = np.reshape(img_arr[self.idx:self.idx + self.num_slices], (1, nz, nx, ny, 1))
        y = np.reshape(label_arr[self.idx:self.idx + self.num_slices, ...], (1, nz, nx, ny, 1))
        self.idx+=self.num_slices
        #del img_arr, label_arr

        return X, y


