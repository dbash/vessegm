import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import SimpleITK
import glob
import os
import imageio
from fnmatch import fnmatch


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

def get_mhd_list(folder):
    pattern = "*.mhd"
    f_list = []

    for path, subdirs, files in os.walk(folder):
        for name in files:
            if fnmatch(name, pattern):
                fpath = os.path.join(path, name)
                f_list.append(fpath)
    return f_list

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

def show_img(filepath, slice_idx = 0):
    img = SimpleITK.ReadImage(filepath)
    img_arr = SimpleITK.GetArrayFromImage(img)
    plt.imshow(img_arr[slice_idx], cmap='gray')
    plt.show()

def save_as_gif(filepath,gifpath):
    img = SimpleITK.ReadImage(filepath)
    img_arr = SimpleITK.GetArrayFromImage(img)
    n_slices = img.GetSize()[2]
    durn = n_slices*0.0003
    with imageio.get_writer(gifpath, mode='I', duration=durn) as writer:
        for i in range(n_slices):
            writer.append_data(img_arr[i])
    print("Saved image to %s."%gifpath)
    del img_arr, img


if __name__ == '__main__':
    pass


"""testing"""
mhd_path = "/scratch/VESSEL12/images/01/VESSEL12_01.mhd"
gif_path = "/scratch/vessel_1.gif"
#show_img(mhd_path, 100)
#save_as_gif(mhd_path, gif_path)
#X, y = get_dataset('/scratch/VESSEL12/images', '/scratch/VESSEL12/masks')
#print(y['VESSEL12_01.mhd'].shape)

