import os
import sys
import utils  # local import
import skimage
import numpy as np
from numpy import log
from numpy import std
from numpy import exp
from math import floor
from numpy import mean
from numpy import cov
from numpy import trace
import tensorflow as tf
from numpy import asarray
from model import Generator  # local import
from numpy import expand_dims
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

EPOCHS = 100
LAMBDA = 100
BATCH_SIZE = 8
IMG_WIDTH = 256
IMG_HEIGHT = 256
BUFFER_SIZE = 400
DATASET = 'cityscapes'

num_of_samples = 100  # number of samples to test the model

# model params
ff_dim = 32
num_heads = 2
patch_size = 8
embed_dim = 64
projection_dim = 64
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
num_patches = (IMG_HEIGHT // patch_size) ** 2

path_to_weights = sys.argv[1]
device = '/device:GPU:0' if utils.check_cuda else '/cpu:0'


_URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{DATASET}.tar.gz'

path_to_zip = tf.keras.utils.get_file(f'{DATASET}.tar.gz',
                                      origin=_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), f'{DATASET}/')


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


inp, re = load(PATH+'train/100.jpg')


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return real_image, input_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


def generate_samples(model, dataset, device, num_of_samples):
    with tf.device(device):
        outs = list()
        targets = list()

        for n, (input_image, target) in dataset.enumerate():
            
            target = np.array(target)
            targets.append(target)

            input_image = np.array(input_image)
            model_out = np.squeeze(np.array(model(input_image, training=False)).reshape((-1, 256, 256, 3)))
            outs.append(model_out)

            if (n + 1) % num_of_samples == 0:
                break

        return outs, targets


def pre_process(outs, targets):
    outs = np.array(outs)    
    targets = np.array(targets)
    
    outs = outs.reshape((-1, 3, 256, 256))
    targets = targets.reshape(-1, 3, 256, 256)
    
    outs = outs * 0.5 + 0.5
    targets = targets * 0.5 + 0.5
    
    outs = outs * 255
    targets = targets * 255

    return outs, targets

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
    # load inception v3 model
    model = InceptionV3()
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (299,299,3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std


# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = skimage.transform.resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# calculate frechet inception distance
def calculate_fid(images1, images2):
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    images1 = scale_images(images1, (299,299,3))
    images2 = scale_images(images2, (299,299,3))

    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


model = Generator(input_shape, patch_size, num_patches, projection_dim, num_heads, ff_dim)
model.load_weights(path_to_weights)


# generate and process samples from the model
outs, targets = generate_samples(model, train_dataset, device, num_of_samples)
outs, targets = pre_process(outs, targets)


# calculate fid, ssim, inception score
fid_score = calculate_fid(targets, outs)
ssim_score = ssim(targets.reshape(-1, 256, 256, 3), outs.reshape(-1, 256, 256, 3), data_range=targets.max() - targets.min(), multichannel=True)
inception_score = calculate_inception_score(outs)

print('----------------|-------------')
print(f'ssim score      | {ssim_score}')
print(f'FID             | {fid_score}')
print(f'Inception score | mean: {inception_score[0]} std: {inception_score[1]}')
print('----------------|-------------')
