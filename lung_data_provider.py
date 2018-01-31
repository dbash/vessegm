import numpy as np
from image_util import BaseDataProvider
import img_processing as proc
from scipy.misc import imresize



class LungDataProvider(BaseDataProvider):
    channels = 1
    classes = 2
    img_list = []
    label_list = []
    img_idx = 0
    n_cur_img = -1
    num_slices = 100
    max_patches = 3
    id_patch = 0
    lim_x = 100
    lim_y = 100


    def _load_data_and_label(self, balanced=False):
        data, label = self.next_data_whole() #self._next_data(balanced)

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

            labels[..., 1] = label[..., 0]
            labels[..., 0] = 1 - labels[..., 1]
            return labels

        return label


    def __call__(self, *args, **kwargs):
        balanced = kwargs.get("balanced", False)
        return self._load_data_and_label(balanced)

    def __init__(self, img_folder, label_folder, nx=352, ny=480, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        super(BaseDataProvider, self).__init__()
        self.img_list = proc.get_file_list(img_folder, pattern='*.mhd')

        self.label_list = proc.get_file_list(label_folder, pattern='*.mhd')
        self.n_examples = len(self.label_list)
        self.img_arr = proc.get_image_array(self.img_list[0], normalize=True)
        self.label_arr = proc.get_image_array(self.label_list[0], normalize=True)
        self.img_idx = 1
        self.id_patch = 0
        self.nz = self.img_arr.shape[0]
        self.nx = self.img_arr.shape[1]
        self.ny = self.img_arr.shape[2]
        print("Number of images = %d" % self.n_examples)

    def _get_rnd_idx(self):
        idz = np.random.randint(0, self.nz - self.num_slices - 1)
        idx = np.random.randint(0, self.nx - 2*self.lim_x - 1)
        idy = np.random.randint(0, self.ny - 2*self.lim_y - 1)
        return (idz, idx, idy)

    def _next_data(self, balanced=False):
        print("img #%d" % self.img_idx)
        print("patch #%d" % self.id_patch)
        #if self.img_idx >= self.n_examples:
        #    print("No more images left.")
        #    return
        if self.max_patches < self.id_patch:
            self.img_idx = np.random.randint(len(self.img_list))
            self.img_arr = proc.get_image_array(self.img_list[self.img_idx], normalize=True)
            self.label_arr = proc.get_image_array(self.label_list[self.img_idx], normalize=True)
            #self.img_idx += 1
            self.id_patch = 0
            self.nz = self.img_arr.shape[0]
            self.nx = self.img_arr.shape[1]
            self.ny = self.img_arr.shape[2]

        self.id_patch += 1
        idz, idx, idy = self._get_rnd_idx()

        if balanced:
            search = True
            while search:
                label_patch = self.label_arr[idz:idz + self.num_slices,
                              idx:idx + 2*self.lim_x, idy:idy + 2*self.lim_y]
                label_patch /= label_patch.max() + 0.00001
                if label_patch.mean() > 0.1:
                    print("patch mean = %f" % label_patch.mean())
                    # proc.show_img_arr(label_patch, slice=50)
                    search = False
                else:
                    idz, idx, idy = self._get_rnd_idx()
        print("idz = %d, idx = %d, idy = %d" % (idz, idx, idy))
        x_full = self.img_arr[idz:idz + self.num_slices, idx:idx + 2*self.lim_x, idy:idy + 2*self.lim_y]
        y_full = self.label_arr[idz:idz + self.num_slices, idx:idx + 2*self.lim_x, idy:idy + 2*self.lim_y]

        x_resized = np.zeros((self.num_slices, self.lim_x, self.lim_y))
        y_resized = np.zeros((self.num_slices, self.lim_x, self.lim_y))

        for i in range(self.num_slices):
            x_resized[i, ...] = imresize(x_full[i, ...], 50)
            y_resized[i, ...] = imresize(y_full[i, ...], 50)
        del x_full, y_full

        X = np.reshape(x_resized, (1, self.num_slices, self.lim_x, self.lim_y, 1))
        y = np.reshape(y_resized, (1, self.num_slices, self.lim_x, self.lim_y, 1))
        y = np.ceil(y/(y.max() + 0.0001))
        del x_resized, y_resized

        return X, y


    def next_data_whole(self):
        idz = np.random.randint(0, self.nz - self.num_slices - 1)
        in_img = np.random.randint(len(self.img_list))
        print("img # %d, idz = %d" % (in_img, idz))
        x_full = proc.get_image_array(self.img_list[in_img], normalize=True)[idz:idz + self.num_slices, ...]
        y_full = proc.get_image_array(self.label_list[in_img], normalize=True)[idz:idz + self.num_slices, ...]
        x_resized = np.zeros((self.num_slices, self.lim_x, self.lim_y))
        y_resized = np.zeros((self.num_slices, self.lim_x, self.lim_y))
        for i in range(self.num_slices):
            x_resized[i] = imresize(x_full[i], (self.lim_x, self.lim_y))
            y_resized[i] = imresize(y_full[i], (self.lim_x, self.lim_y))

        X = np.reshape(x_resized, (1, self.num_slices, self.lim_x, self.lim_y, 1))
        y = np.reshape(y_resized, (1, self.num_slices, self.lim_x, self.lim_y, 1))
        y = np.ceil(y / (y.max() + 0.0001))
        del x_full, y_full, x_resized, y_resized
        return X, y


