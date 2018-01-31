# import remote_config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
import Unet3D as unet
import img_processing as proc
import lung_data_provider as ldprovider
import matplotlib.pyplot as plt


def main():
    img_folder = "/scratch/VESSEL/cropped_data/images"
    label_folder = "/scratch/VESSEL/cropped_data/masks"
    provider = ldprovider.LungDataProvider(img_folder, label_folder)
    X, y = provider()

    fig, img_arr = plt.subplots(1, 2, sharey='all')

    img_arr[0].imshow(X[0, 5, :, :, 0], cmap='gray')
    img_arr[0].set_title("Input image")
    img_arr[1].imshow(y[0, 5, :, :, 0], cmap='gray')
    img_arr[1].set_title("Label")
    plt.show()
    #print("0")
    net = unet.Unet3D(channels=1, n_class=2, cost="cross_entropy", summaries=False)
    trainer = unet.Trainer(net, batch_size=1, optimizer="adam",  norm_grads=True,
                           opt_kwargs=dict(learning_rate=0.0000005))
    #print("1")
    if not os.path.exists("unet3d_trained"):
        os.path.mkdir("/home/guest/dbash/unet3d_trained")
    trainer.train(provider, "/home/guest/dbash/unet3d_trained", epochs=50,
                  training_iters=10, restore=True)
    #print("2")

def test_patch(img_folder,label_folder, new_img_path, net, model_path, coord=(0,0,0), size=100):
    z, x, y = coord
    provider = ldprovider.LungDataProvider(img_folder, label_folder)
    img_arr, label_arr = provider()
    prediction = net.predict(model_path, img_arr)
    label_patch = unet.crop_to_shape(label_arr, prediction.shape)
    err_rate = unet.error_rate(predictions=prediction, labels=label_patch)
    print("Error rate = %f" % err_rate)
    cropped_img = 255*unet.crop_to_shape(img_arr, prediction.shape)[0,...,0]

    prediction = np.argmax(prediction, 4)[0]
    label = np.argmax(label_patch, 4)[0]
    print(np.sum(prediction))
    proc.double_mask_arr(cropped_img,
                        prediction, label, new_img_path)
    #proc.mask_image_arr(cropped_img,
    #                    label,
    #                    "/home/guest/dbash/masked_img/VESSEL12_01_gt_patch.gif")
    #proc.show_img_arr(prediction, slice=20)
    return prediction



def predict_image(img_path, model_path, new_mhd_path, new_gif_path):
    net = unet.Unet3D(channels=1, n_class=2, cost="cross_entropy", summaries=False)
    #provider = ldprovider.LungDataProvider(img_folder, label_folder)
    x_dummy = np.zeros((1, 100, 100, 100, 1))
    pred = net.predict(model_path, x_dummy)
    ps = pred.shape[1]
    del x_dummy, pred

    img_arr = proc.get_image_array(img_path, normalize=True)
    img_arr = (img_arr - img_arr.min()) / img_arr.max()
    nz = int(img_arr.shape[0]//ps)
    dz = int(np.ceil(0.5*(img_arr.shape[0] % ps)))
    img_arr = img_arr[dz:dz + ps*nz]

    diff = 100 - ps
    new_shape = (img_arr.shape[0] + diff, img_arr.shape[1] + diff, img_arr.shape[2] + diff)
    ext_img_arr = np.zeros(new_shape)
    d2 = diff//2
    ext_img_arr[d2:-d2, d2:-d2, d2:-d2] = img_arr
    id_0, id_1, id_2 = 0, 0, 0


    res_img = np.zeros(img_arr.shape)
    nz, nx, ny = img_arr.shape[0]//ps, img_arr.shape[1]//ps, img_arr.shape[2]//ps
    for i in range(nz):
        for j in range(nx):
            for k in range(ny):
                img_slice = np.reshape(ext_img_arr[id_0:id_0 + 100, id_1:id_1 + 100, id_2:id_2 + 100],
                                       (1, 100, 100, 100, 1))
                pred_slice = net.predict(model_path, img_slice)
                res_img[id_0:id_0 + ps, id_1: id_1 + ps, id_2: id_2 + ps] = np.argmax(pred_slice, 4)[0]
                id_2 += ps

            id_2 = 0
            id_1 += ps
        id_1 = 0
        id_0 += ps

    proc.save_array_as_gif(255*res_img, new_gif_path)
    proc.save_image_as_mhd(res_img, new_mhd_path)





def test(n_ex, model_path):
    avg_acc = 0.0
    #n_ex = 50
    img_folder = "/scratch/VESSEL/cropped_data/images"
    label_folder = "/scratch/VESSEL/cropped_data/masks"
    net = unet.Unet3D(channels=1, n_class=2, cost="cross_entropy", summaries=False)
    provider = ldprovider.LungDataProvider(img_folder, label_folder)
    for _ in range(n_ex):
        X_test, y_test = provider._next_data()
        prediction = net.predict(model_path, X_test)
        new_shape = (prediction.shape[0], prediction.shape[1], prediction.shape[2],prediction.shape[3], 1)
        prediction = np.reshape(prediction[..., 1], new_shape)
        y_cropped = unet.crop_to_shape(y_test, new_shape)
        n_pix = new_shape[0]*new_shape[1]*new_shape[2]*new_shape[3]
        #print(prediction.shape, y_test.shape)
        acc = np.sum((prediction>0.5)==y_cropped)/(n_pix)
        #unet.error_rate(predictions=prediction, labels= unet.crop_to_shape(y_test, prediction.shape))
        print(acc)
        avg_acc += acc
    avg_acc/=n_ex
    print("average accuracy = ", avg_acc)


img_path = "/scratch/vessel_data/VESSEL/cropped_data/images/VESSEL12_01.mhd"
mask_path = "/scratch/VESSEL/cropped_data/masks/"
new_img_path = "/scratch/vessel_data/VESSEL/cropped_data/masked_img/MASKED.mhd"
new_gif_path = "/scratch/vessel_data/VESSEL/cropped_data/masked_img/MASKED.gif"
model_path = "/home/guest/dbash/unet3d_trained_25012018/round1/model.cpkt"
#net = unet.Unet3D(channels=1, n_class=2, cost="cross_entropy", summaries=False)
#coord = (np.random.randint(0, 250), np.random.randint(0, 250), np.random.randint(0, 250))
#print("coord =  ", coord)
#test_patch(img_path, mask_path, new_img_path, net, model_path, coord=coord, size=100)
#(215, 162, 270
# test(n_ex, model_path)

predict_image(img_path, model_path, new_img_path, new_gif_path)
#main()

"""fig1, ax1 = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
ax1[0].imshow(X_test[0, 100, :, :, 0], aspect="auto")
ax1[1].imshow(y_test[0, 100, :, :, 0], aspect="auto")
mask = prediction[0, 100, :, :, 0] > 0.9
ax1[2].imshow(mask, aspect="auto")
ax1[0].set_title("Input")
ax1[1].set_title("Ground truth")
ax1[2].set_title("Prediction")
fig1.tight_layout()
fig1.savefig("../example_lung.png")
plt.show()"""
