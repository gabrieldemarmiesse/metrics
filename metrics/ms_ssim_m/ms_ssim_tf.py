# !/usr/bin/python
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of MS-SSIM.
Usage:
"""
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from keras import backend as K
#tfe.enable_eager_execution()


def broadcast_to(tensor, shape):
    return tensor + tf.zeros(dtype=tensor.dtype, shape=shape)


def pad(im):
    return tf.pad(im, np.array([[0, 0], [0, 1], [0, 1], [0, 0]]))


def conv(img, window, size):
    custom_window = tf.reshape(window, (size, size, 1, 1))
    custom_window = broadcast_to(custom_window, (size, size, 3, 1))
    return tf.nn.depthwise_conv2d(tf.cast(img, tf.float32),
                                  custom_window,
                                  strides=[1, 1, 1, 1],
                                  padding='VALID')


def _f_special_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    g /= g.sum()
    return tf.constant(g, dtype=tf.float32)


def _ssim_for_multiscale(img1, img2, max_val=255, filter_size=11,
                         filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.
    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """

    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    #size = tf.min(filter_size, height, width)
    size = filter_size

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = tf.reshape(_f_special_gauss(size, sigma), (1, size, size, 1))
        mu1 = conv(img1, window, size)
        mu2 = conv(img2, window, size)
        sigma11 = conv(img1 * img1, window, size)
        sigma22 = conv(img2 * img2, window, size)
        sigma12 = conv(img1 * img2, window, size)
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = tf.reduce_mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = tf.reduce_mean(v1 / v2)
    return ssim, cs


def ms_ssim_tf(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
            k1=0.01, k2=0.03, weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
        img1: Numpy array holding the first RGB image batch.
        img2: Numpy array holding the second RGB image batch.
        max_val: the dynamic range of the images (i.e., the difference between the
            maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
          for small images).
        k1: Constant used to maintain stability in the SSIM calculation (0.01 in
            the original paper).
        k2: Constant used to maintain stability in the SSIM calculation (0.03 in
            the original paper).
        weights: List of weights for each level; if none, use five levels and the
            weights from the original paper.
    Returns:
        MS-SSIM score between `img1` and `img2`.
    Raises:
        RuntimeError: If input images don't have the same shape or don't have four
            dimensions: [batch_size, height, width, depth].
    """

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else
                       [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1), dtype=np.float32) / 4.0
    im1, im2 = [tf.cast(x, tf.float32) for x in [img1, img2]]
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _ssim_for_multiscale(
            im1, im2, max_val=max_val, filter_size=filter_size,
            filter_sigma=filter_sigma, k1=k1, k2=k2)
        mssim.append(ssim)
        mcs.append(cs)
        filtered = [conv(pad(im), downsample_filter, 2)
                    for im in [im1, im2]]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return (tf.reduce_prod(tf.convert_to_tensor(mcs[0:levels - 1]) ** weights[0:levels - 1]) *
            (mssim[levels - 1] ** weights[levels - 1]))


class MS_SSIM_TF:

    def __init__(self):
        pl1 = K.placeholder((None, None, None, 3), dtype=tf.float32)
        pl2 = K.placeholder((None, None, None, 3), dtype=tf.float32)
        out = ms_ssim_tf(pl1, pl2)
        self._func = K.function([pl1, pl2], [out])

    def ms_ssim(self, img1, img2):
        return self._func([img1, img2])[0]
