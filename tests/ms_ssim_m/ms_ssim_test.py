from metrics.ms_ssim_m.ms_ssim_tf import MS_SSIM_TF
from metrics.ms_ssim_m.ms_ssim import ms_ssim

import imageio
import numpy as np
import os

image = imageio.imread(os.path.join(__file__, '../..', 'corgi.jpg')).astype(np.int32)
noisy = (image + np.random.uniform(-10, 10, image.shape)).astype(np.int32)

obj = MS_SSIM_TF()
print(ms_ssim(image[None], noisy[None]))
print(obj.ms_ssim(image[None], noisy[None]))
