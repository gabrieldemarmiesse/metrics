import os

import imageio
import numpy as np

from metrics.tests_utils_cy import time_func
from metrics.ms_ssim_m.ms_ssim import ms_ssim
from metrics.ms_ssim_m.ms_ssim_cy import ms_ssim as ms_ssim_cy

image = imageio.imread(os.path.join(__file__, '../..', 'corgi.jpg')).astype(np.int32)
noisy = (image + np.random.uniform(-5, 5, image.shape)).astype(np.int32)

time_func(ms_ssim, 'ms_ssim', image[None], noisy[None])
time_func(ms_ssim_cy, 'ms_ssim_cy cython', image[None], noisy[None])
