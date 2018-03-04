import os
import time

import imageio
import numpy as np

from metrics.mse.mse import mse
from metrics.mse.mse_cy import mse as mse_cy

image = imageio.imread(os.path.join(__file__, '../..', 'corgi.jpg')).astype(np.int32)

noisy = (image + np.random.uniform(-5, 5, image.shape)).astype(np.int32)

t = time.time()
for _ in range(100):
    mse(image, noisy)
print(time.time() - t, mse(image, noisy))

t = time.time()
for _ in range(100):
    mse_cy(image, noisy)
print(time.time() - t, mse_cy(image, noisy))
