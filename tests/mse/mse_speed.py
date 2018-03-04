import os
import timeit

import imageio
import numpy as np

from metrics.mse.mse import mse
from metrics.mse.mse_cy import mse as mse_cy

image = imageio.imread(os.path.join(__file__, '../..', 'corgi.jpg')).astype(np.int32)

noisy = (image + np.random.uniform(-5, 5, image.shape)).astype(np.int32)
n = 1000
t = timeit.timeit('mse(image, noisy)', number=n, globals=globals())/n
print(t*1000, 'ms per call')
print(mse(image, noisy))

t = timeit.timeit('mse_cy(image, noisy)', number=n, globals=globals())/n
print(t*1000, 'ms per call')
print(mse_cy(image, noisy))
