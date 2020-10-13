# methods to recover the model and some helper functions

from keras.models import load_model
from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Input

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))


def get_depth(depth_file, init_offset):
    mat = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1 : val = 1200
                mat[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat = np.asarray(mat)
    if init_offset:
        mat_small = mat[140:340, 236:436]
    else:
        mat_small = mat[140:340, 220:420]
    return mat_small


def get_rgb(depth_file):
    img = Image.open(depth_file[:-5] + "c.bmp")
    img.thumbnail((640, 480))
    img = np.asarray(img)
    img = img[140:340, 220:420]
    return img


# combine a rgb photo and a depth picture to a matrix
# when init_offset = true, generate a rgbd couple with init_offset to match the RGB-D database
def create_input_rgbd(depth_file, init_offset=False):

    mat_small = get_depth(depth_file, init_offset)
    mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)
    img = get_rgb(depth_file)

    full1 = np.zeros((200, 200, 4))
    full1[:, :, :3] = img[:, :, :3]
    full1[:, :, 3] = mat_small

    return np.array([full1])


# plot the rgb and depth seperately.
# img is in the same format as the output of create_input_rgbd()
# rescale_rgb and rescale_d for plotting noise only
def plot_rgbd(img, plot=True, save=False, path=None, file_name=None, rgb_title="", depth_title=""):
    # plot depth
    depth = img[0][:, :, 3]
    plt.figure(figsize=(8, 8))
    plt.title(depth_title)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(depth)
    # plt.grid(color='r')

    if save:
        plt.savefig(path+file_name+"_D.png")
    if plot:
        plt.show()
    plt.close()

    # plot rgb
    rgb = img[0][:, :, :3]
    plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])
    plt.title(rgb_title)
    plt.imshow(rgb.astype('uint8'))
    # plt.grid(color='r')
    if save:
        plt.savefig(path+file_name+"_RGB.png")
    if plot:
        plt.show()
    plt.close()


def plot_test(ref, user, adv, original_similarity, new_similarity, l2_norm, time_cost):
    rgb_ref = ref[0][:, :, :3]
    depth_ref = ref[0][:, :, 3]
    rgb_user = user[0][:, :, :3]
    depth_user = user[0][:, :, 3]
    rgb_adv = adv[0][:, :, :3]
    depth_adv = adv[0][:, :, 3]

    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Demo Result of VVSec\nTime cost ' + time_cost[-9:-3]+'s\n')
    axs[0, 0].imshow(rgb_ref.astype('uint8'))
    axs[0, 0].set_title('Reference Input\n')
    axs[0, 0].set(ylabel='RGB')
    axs[0, 1].imshow(rgb_user.astype('uint8'))
    axs[0, 1].set_title('User Input\n'+'Similarity %.3f' % original_similarity+'\n')
    axs[0, 2].imshow(rgb_adv.astype('uint8'))
    axs[0, 2].set_title('Perturbed User Input\n'+'Similarity %.3f' % new_similarity+'\nL2 norm %.3f' % l2_norm)
    axs[1, 0].imshow(depth_ref)
    axs[1, 0].set(ylabel='Depth')
    axs[1, 1].imshow(depth_user)
    axs[1, 2].imshow(depth_adv)
    for ax in axs.flat:
        ax.set(xticks=[], yticks=[])
    plt.show()


# get the model for the entire face id system.
def get_model():
    im_in1 = Input(shape=(200, 200, 4))
    im_in2 = Input(shape=(200, 200, 4))

    model_top = load_model("./model/model_top.model")
    feat_x1 = model_top(im_in1)
    feat_x2 = model_top(im_in2)

    lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])

    model_recover = Model(inputs=[im_in1, im_in2], outputs=lambda_merge)
    model_recover.summary()
    return model_recover
