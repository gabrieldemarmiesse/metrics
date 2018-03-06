import os

import imageio
import numpy as np

from metrics.tests_utils_cy import time_func
from metrics.mse_m.mse import mse
from metrics.mse_m.mse_cy import mse as mse_cy

image = imageio.imread(os.path.join(__file__, '../..', 'corgi.jpg')).astype(np.int32)
noisy = (image + np.random.uniform(-5, 5, image.shape)).astype(np.int32)

time_func(mse, 'mse', image, noisy)
time_func(mse_cy, 'mse cython', image, noisy)
