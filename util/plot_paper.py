from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def image_grid(x, image_size, num_channels):
  img = x.reshape(-1, image_size, image_size, num_channels)
  w = int(np.sqrt(img.shape[0]))
  img = img[:w**2, :, :, :]
  return img.reshape((w, w, image_size, image_size, num_channels)).transpose((0, 2, 1, 3, 4)).reshape((w * image_size, w * image_size, num_channels))


device = 'cuda:0'
noises = ['0.01', '0.05', '0.1', '0.2']
# noises = ['0.01', '0.05', '0.1']
image_size = 256
if np.size(noises) == 3:
  gs_right = 2./3
elif np.size(noises) == 4:
  gs_right = 2./4

  
num_channel = 3

# task = 'super_resolution8'
# task = 'motion_blur'
task = 'gaussian_blur'
# task = 'super_resolution'
# task = 'inpainting'
# task = 'inpaintingrandom'

if task=='super_resolution':
  input_image_size = image_size//4
elif task=='super_resolution8':
  input_image_size = image_size//8
else:
  input_image_size = image_size

save_fname = "appendix_ffhq_" + task + "_VE"

num_eval = 9

figure_sizes = {
  'full_width' : 8,
}
fig = plt.figure(figsize=(figure_sizes['full_width'],0.4*figure_sizes['full_width']))
gs = GridSpec(np.size(noises), 5, height_ratios=[1] * np.size(noises), width_ratios=[1] * 5)
# gs = GridSpec(3, 5, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1])
# gs.update(left=0.0,
#           right=1.,
#           top=1.,
#           bottom=0.,
#           wspace=-.705125,hspace=0.0)
gs.update(left=0.0,
          right=gs_right,
          top=1.,
          bottom=0.,
          wspace=0.,hspace=0.0)

ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(np.size(noises))])

for i, _ in enumerate(noises):
  for j in range(5):
    ax[i, j].axis('off')

for i, noise in enumerate(noises):
  label_root = Path(f'/home/bb515/code/tmpdtorch/results/{task}/label/{noise}')
  input_root = Path(f'/home/bb515/code/tmpdtorch/results/{task}/input/{noise}')
  tmpd_recon_root = Path(f'/home/bb515/code/tmpdtorch/results/{task}/tmpd/{noise}')
  dps_recon_root = Path(f'/home/bb515/code/tmpdtorch/results/{task}/dps/{noise}')
  pigdm_recon_root = Path(f'/home/bb515/code/tmpdtorch/results/{task}/pigdm/{noise}')

  labels_x = np.empty((num_eval, image_size, image_size, num_channel))
  input_x = np.empty((num_eval, input_image_size, input_image_size, num_channel))
  dps_x = np.empty((num_eval, image_size, image_size, num_channel))
  pigdm_x = np.empty((num_eval, image_size, image_size, num_channel))
  tmpd_x = np.empty((num_eval, image_size, image_size, num_channel))

  for idx in tqdm(range(num_eval)):
    fname = str(idx).zfill(5)
    label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
    input = plt.imread(input_root / f'{fname}.png')[:, :, :3]
    pigdm_recon = plt.imread(pigdm_recon_root / f'{fname}.png')[:, :, :3]
    tmpd_recon = plt.imread(tmpd_recon_root / f'{fname}.png')[:, :, :3]
    dps_recon = plt.imread(dps_recon_root / f'{fname}.png')[:, :, :3]

    # As we are reading, need to save them.
    labels_x[idx, :, :, :] = label
    input_x[idx, :, :, :] = input
    dps_x[idx, :, :, :] = dps_recon
    pigdm_x[idx, :, :, :] = pigdm_recon
    tmpd_x[idx, :, :, :] = tmpd_recon

  labels_img = image_grid(labels_x, image_size, num_channel)
  input_img = image_grid(input_x, input_image_size, num_channel)
  dps_img = image_grid(dps_x, image_size, num_channel)
  pigdm_img = image_grid(pigdm_x, image_size, num_channel)
  tmpd_img = image_grid(tmpd_x, image_size, num_channel)

  ax[i, 0].imshow(labels_img, interpolation=None)
  ax[i, 1].imshow(input_img, interpolation=None)
  ax[i, 2].imshow(tmpd_img, interpolation=None)
  ax[i, 3].imshow(dps_img, interpolation=None)
  ax[i, 4].imshow(pigdm_img, interpolation=None)

plt.savefig(save_fname + '.png', bbox_inches='tight', pad_inches=0.0, dpi=1000)
plt.close()