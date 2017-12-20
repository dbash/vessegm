import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import SimpleITK
import glob
import os
from fnmatch import fnmatch


img_path = "/scratch/VESSEL12/images/01/VESSEL12_01.raw"
mhd_path = "/scratch/VESSEL12/images/01/VESSEL12_01.mhd"
mhd_mask_path = "/scratch/VESSEL12/masks/VESSEL12_01.mhd"
idx_slice = 12
#image = np.empty((512, 512), np.uint16)
img = SimpleITK.ReadImage(mhd_mask_path)
#plt.imshow(x)
#print(img[0:30, 0:30, 12])
#SimpleITK.Show(img)
#size = list(img.GetSize())
img_arr = SimpleITK.GetArrayFromImage(img)
#print(img_arr.max(), img_arr.min())
#max, min  = img_arr.max(), img_arr.min()
#diff = max - min
#img_rescale = (img_arr - min)*255.0/diff
#print(plt.isinteractive())
#f, subplot = plt.subplots(2)
#subplot[0].imshow(img_arr[120], cmap='gray')
#subplot[1].imshow(img_rescale[120], cmap='gray')
plt.imshow(img_arr[120], cmap='gray')
#plt.imshow(img_rescale[120], cmap='gray')
plt.show()


def get_image_array(mhd_file, normalize=False):
    img = SimpleITK.ReadImage(mhd_file)
    img_arr = SimpleITK.GetArrayFromImage(img)
    if (normalize):
        max, min = img_arr.max(), img_arr.min()
        diff = max - min
        img_arr = (img_arr - min)*255.0/diff
    return img_arr

def get_images(folder_name, normalize=False):
    pattern = "*.mhd"
    f_list = []
    X_dict = {}
    for path, subdirs, files in os.walk(folder_name):
        for name in files:
            if fnmatch(name, pattern):
                fpath = os.path.join(path, name)
                print("found file %s"%fpath)
                X_dict[name] = get_image_array(fpath, normalize)
                f_list.append(fpath)

    return X_dict

def get_dataset(img_folder, mask_folder, normalize=False):
    pattern = "*.mhd"
    img_dict = {}
    mask_dict = {}
    for path, subdirs, files in os.walk(img_folder):
        for name in files:
            if fnmatch(name, pattern):
                fpath = os.path.join(path, name)
                img_dict[name] = get_image_array(fpath, normalize)

    for path, subdirs, files in os.walk(mask_folder):
        for name in files:
            if fnmatch(name, pattern):
                fpath = os.path.join(path, name)
                mask_dict[name] = get_image_array(fpath, normalize)
    if not (img_dict.keys()==mask_dict.keys()):
        print('Warning: images dict and mask dict do not match.')
    return img_dict, mask_dict


def get_dataset_tuples(img_folder, mask_folder, normalize=False):
    data = []
    X, y = get_dataset(img_folder, mask_folder, normalize)
    for key in X.keys():
        data.append((X[key], y[key]))
    del X, y
    return data

#X, y = get_dataset('/scratch/VESSEL12/images', '/scratch/VESSEL12/masks')
#print(y['VESSEL12_01.mhd'].shape)
