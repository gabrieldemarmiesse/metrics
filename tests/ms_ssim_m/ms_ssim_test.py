from metrics.ms_ssim_m.ms_ssim_tf import ms_ssim as ms_ssim_tf
from metrics.ms_ssim_m.ms_ssim import ms_ssim

import imageio
import numpy as np
import os

image = imageio.imread(os.path.join(__file__, '../..', 'corgi.jpg')).astype(np.int32)
noisy = (image + np.random.uniform(-5, 5, image.shape)).astype(np.int32)

print(ms_ssim(image[None], noisy[None]))
print(ms_ssim_tf(image[None], noisy[None]))
