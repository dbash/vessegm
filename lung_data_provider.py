import numpy as np
from tf_unet.image_util import BaseDataProvider
import img_processing as proc



class LungDataProvider(BaseDataProvider):
    channels = 1
    classes = 2
    img_list = []
    label_list = []
    idx = 0
    n_examples = 0

    def __init__(self, img_folder, label_folder, nx=512, ny=512):
        super(BaseDataProvider, self).__init__()
        self.img_list = proc.get_mhd_list(img_folder)
        self.label_list = proc.get_mhd_list(label_folder)
        self.n_examples = len(self.label_list)
        self.nx=nx
        self.ny=ny

    def _next_data(self):
        if (self.idx>=self.n_examples):
            print("No more images left.")
            return
        img_arr = proc.get_image_array(self.img_list[self.idx], normalize=True)
        label_arr = proc.get_image_array(self.label_list[self.idx], normalize=True)
        self.idx+=1

        nz = img_arr.shape[0]
        nx = img_arr.shape[1]
        ny = img_arr.shape[2]

        if (self.nx!=nx or self.ny!=ny):
            print("Warning: Img size doesn't match to the model (nx=%d, ny=%d)."%(nx, ny))
        X = np.reshape(img_arr, (1, nz, nx, ny, 1))
        y = np.reshape(label_arr, (1, nz, nx, ny, 1))

        del img_arr, label_arr

        return X, y


