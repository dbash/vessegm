import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import SimpleITK
import glob
import os
import imageio
from fnmatch import fnmatch
import csv


def get_image_array(mhd_file, normalize=False):
    img = SimpleITK.ReadImage(mhd_file)
    img_arr = SimpleITK.GetArrayFromImage(img)
    if (normalize):
        max, min = img_arr.max(), img_arr.min()
        diff = max - min
        img_arr = (img_arr - min)*255.0/diff
    return img_arr

def normalize(img):
    min = img.min()
    max = img.max()
    dif = max - min
    st_img = (img - min) * 255.0 / dif
    return st_img


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


def get_mhd_dict(img_folder, label_folder):
    pattern = "*.mhd"
    img_dict = {}
    label_dict = {}
    for path, subdirs, files in os.walk(img_folder):
        for name in files:
            if fnmatch(name, pattern):
                fpath = os.path.join(path, name)
                img_dict[name] = fpath

    for path, subdirs, files in os.walk(label_folder):
        for name in files:
            if fnmatch(name, pattern):
                fpath = os.path.join(path, name)
                label_dict[name] = fpath
    data_dict = {}
    for key in img_dict.keys():
        data_dict[img_dict[key]] = label_dict[key]
    del label_dict, img_dict
    return data_dict


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

def save_array_as_gif(img_arr,gifpath):
    n_slices = img_arr.shape[0]
    durn = n_slices*0.0003
    with imageio.get_writer(gifpath, mode='I', duration=durn) as writer:
        for i in range(n_slices):
            writer.append_data(img_arr[i])
    print("Saved image to %s."%gifpath)
    del img_arr


def crop_ROI(img):
    step = 16

    nx = img.shape[1]
    ny = img.shape[2]
    k = 50
    avg_center = img[0].mean()
    print("avg_center = ", avg_center)
    gr_x, gr_y = nx // step, ny // step
    grid = np.zeros((gr_x, gr_y))

    for i in range(gr_x):
        for j in range(gr_y):
            grid[i, j] = int(img[:, i * step:(i + 1) * step, j * step:(j + 1) * step].mean() > avg_center)

    sum_x = grid.sum(axis=1)
    sum_y = grid.sum(axis=0)
    x_start = 0
    y_start = 0
    x_end = sum_x.shape[0] - 1
    y_end = sum_y.shape[0] - 1
    print("sum_x = ", sum_x)
    print("sum_y = ", sum_y)
    while sum_x[x_start] < 2:
        x_start += 1

    while sum_y[y_start] < 2:
        y_start += 1

    while sum_x[x_end] < 2:
        x_end -= 1

    while sum_y[y_end] < 2:
        y_end -= 1

    new_img = img[:, x_start * step:x_end * step, y_start * step:y_end * step]
    del grid, sum_x, sum_y
    # print(x_start, y_start, x_end, y_end)
    return new_img


def crop_ROI_limited(img, label, lim_x, lim_y):
    step = 16

    nx = img.shape[1]
    ny = img.shape[2]
    k = 50
    avg_center = img[0].mean()
    print("avg_center = ", avg_center)
    gr_x, gr_y = nx // step, ny // step
    grid = np.zeros((gr_x, gr_y))

    for i in range(gr_x):
        for j in range(gr_y):
            grid[i, j] = int(img[:, i * step:(i + 1) * step, j * step:(j + 1) * step].mean() > avg_center)

    sum_x = grid.sum(axis=1)
    sum_y = grid.sum(axis=0)
    x_start = 0
    y_start = 0
    x_end = sum_x.shape[0] - 1
    y_end = sum_y.shape[0] - 1
    while sum_x[x_start] < 2:
        x_start += 1

    while sum_y[y_start] < 2:
        y_start += 1

    while sum_x[x_end] < 2:
        x_end -= 1

    while sum_y[y_end] < 2:
        y_end -= 1

    x_start = x_start * step
    y_start = y_start * step
    x_end = x_end * step
    y_end = y_end * step

    cx = (x_start + x_end) // 2
    cy = (y_start + y_end) // 2

    kx = lim_x // 2
    ky = lim_y // 2
    cropped_img = img[:, cx - kx:cx + kx, cy - ky: cy + ky]
    cropped_label = label[:, cx - kx:cx + kx, cy - ky: cy + ky]
    return cropped_img, cropped_label


def crop_pair(img_mhd, label_mhd,new_root, lim_x, lim_y, save_gif=False):
    if not os.path.exists(new_root):
        os.mkdir(new_root)
    crop_img_folder = os.path.join(new_root, "images")
    crop_label_folder = os.path.join(new_root, "masks")
    if not os.path.exists(crop_img_folder):
        os.mkdir(crop_img_folder)
    if not os.path.exists(crop_label_folder):
        os.mkdir(crop_label_folder)

    img = get_image_array(img_mhd, normalize=True)
    label = get_image_array(label_mhd, normalize=True)

    cropped_img, cropped_label = crop_ROI_limited(img, label, lim_x, lim_y)
    new_img_path = os.path.join(crop_img_folder, os.path.basename(img_mhd))
    new_label_mhd = os.path.join(crop_label_folder, os.path.basename(label_mhd))

    new_sitk_img = SimpleITK.GetImageFromArray(cropped_img)
    new_sitk_label = SimpleITK.GetImageFromArray(cropped_label)
    SimpleITK.WriteImage(new_sitk_img, new_img_path)
    SimpleITK.WriteImage(new_sitk_label, new_label_mhd)

    if save_gif:
        img_gif = ''.join([os.path.splitext(os.path.basename(img_mhd))[0], '.gif'])
        label_gif = ''.join([os.path.splitext(os.path.basename(label_mhd))[0], '.gif'])
        save_array_as_gif(cropped_img, os.path.join(crop_img_folder, img_gif))
        save_array_as_gif(cropped_label, os.path.join(crop_label_folder, label_gif))
    del cropped_img, cropped_label, img, label, new_sitk_label, new_sitk_img
    print("saved files: ", new_img_path, new_label_mhd)


def crop_dataset(img_folder, label_folder, new_root, lim_x, lim_y, save_gif=True):
    data_dict = get_mhd_dict(img_folder, label_folder)
    for img, label in data_dict.items():
        crop_pair(img, label, new_root, lim_x, lim_y, save_gif)
    print("saved all cropped files in %s. " % new_root)


def save_label_as_img(img_mhd, label_csv, out_path, save_as_gif=False):
    img_arr = get_image_array(img_mhd)
    #print(img_arr.shape)
    label_arr = np.zeros(shape=img_arr.shape)
    with open(label_csv) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            x = int(row[0])
            y = int(row[1])
            z = int(row[2])
            val = int(row[3])
            label_arr[z, x, y] = val

    new_file = os.path.join(out_path, os.path.basename(img_mhd))
    new_sitk_img = SimpleITK.GetImageFromArray(label_arr)
    SimpleITK.WriteImage(new_sitk_img, new_file)
    print("saved labels to file %s." % new_file)
    if save_as_gif:
        new_gif = new_file.replace('mhd', 'gif')
        save_array_as_gif(label_arr, new_gif)
        print("saved labels into gif file %s." % new_gif)
    del label_arr, img_arr


if __name__ == '__main__':
    pass


"""testing"""

img_mhd = "/scratch/VESSEL12/images/01/VESSEL12_01.mhd"
label_mhd = "/scratch/VESSEL12/masks/VESSEL12_01.mhd"
new_root = "/scratch/cropped_data"
gif_path = "/scratch/mask_1.gif"
save_label_as_img("/scratch/VESSEL/Scans/VESSEL12_21.mhd",
                  "/scratch/VESSEL/Annotations/VESSEL12_21_Annotations.csv",
                  "/scratch/VESSEL/Labels", save_as_gif=True)
#crop_dataset("/scratch/VESSEL12/images", "/scratch/VESSEL12/masks", new_root, 352, 480)
#crop_pair(img_mhd, label_mhd, new_root, 352, 480, save_gif=True)
#show_img(mhd_path, 100)
#save_as_gif(mhd_path, gif_path)
#X, y = get_dataset('/scratch/VESSEL12/images', '/scratch/VESSEL12/masks')
#print(y['VESSEL12_01.mhd'].shape)

