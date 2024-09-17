from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch

import gc
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub
import tensorflow_probability as tfp
from tensorflow.image import ssim as tf_ssim
import six


INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
  INCEPTION_OUTPUT: tf.float32,
  INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def mse(a, b):
  return np.mean((a - b)**2)


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
  """Returns a function that can be as a classifier function.

  Copied from tfgan but avoid loading the model each time calling _classifier_fn

  Args:
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    inception_model: A model loaded from TFHub.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]

  def _classifier_fn(images):
    output = inception_model(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

  return _classifier_fn


@tf.function
def run_inception_jit(inputs,
                      inception_model,
                      num_batches=1,
                      inceptionv3=False):
  """Running the inception network. Assuming input is within [0, 255]."""
  if not inceptionv3:
    inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
  else:
    inputs = tf.cast(inputs, tf.float32) / 255.

  return tfgan.eval.run_classifier_fn(
    inputs,
    num_batches=num_batches,
    classifier_fn=classifier_fn_from_tfhub(None, inception_model),
    dtypes=_DEFAULT_DTYPES)


@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
  """Distribute the inception network computation to all available TPUs.

  Args:
    input_tensor: The input images. Assumed to be within [0, 255].
    inception_model: The inception network model obtained from `tfhub`.
    num_batches: The number of batches used for dividing the input.
    inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

  Returns:
    A dictionary with key `pool_3` and `logits`, representing the pool_3 and
      logits of the inception network respectively.
  """
  num_tpus = 1
  input_tensors = tf.split(input_tensor, num_tpus, axis=0)
  pool3 = []
  logits = [] if not inceptionv3 else None
  device_format = '/GPU:{}'
  for i, tensor in enumerate(input_tensors):
    with tf.device(device_format.format(i)):
      tensor_on_device = tf.identity(tensor)
      res = run_inception_jit(
        tensor_on_device, inception_model, num_batches=num_batches,
        inceptionv3=inceptionv3)

      if not inceptionv3:
        pool3.append(res['pool_3'])
        logits.append(res['logits'])  # pytype: disable=attribute-error
      else:
        pool3.append(res)

  with tf.device('/CPU'):
    return {
      'pool_3': tf.concat(pool3, axis=0),
      'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
    }


def get_inception_model(inceptionv3=False):
  if inceptionv3:
    return tfhub.load(
      'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
  else:
    return tfhub.load(INCEPTION_TFHUB)


def load_dataset_stats(dataset):
  """Load the pre-computed dataset statistics."""
  if dataset.lower() == 'cifar10':
    filename = 'assets/cifar10_stats.npz'
  elif dataset.lower() == 'ffhq':
    filename = '../assets/ffhq_clean_trainval_256.npz'
  elif dataset.lower() == 'imagenet':
    filename = '../assets/adm_in256_stats.npz'
  else:
    raise ValueError(f'Dataset {dataset} stats not found.')

  with tf.io.gfile.GFile(filename, 'rb') as fin:
    stats = np.load(fin)
    return stats


def compute_metrics_inner(samples, labels):
  samples = np.clip(samples * 255, 0, 255).astype(np.uint8)


def compute_fid_from_stats(samples, dataset, idx=None, report_KID=False):
    """https://github.com/tensorflow/gan/blob/656e4332d1e6d7f398f0968966c753e44397fc60/tensorflow_gan/python/eval/classifier_metrics.py#L689"""

    def calculate_fid_helper(activations1, m_w, sigma_w):
        activations1 = tf.convert_to_tensor(value=activations1)
        activations1.shape.assert_has_rank(2)
        m_w = tf.convert_to_tensor(value=m_w)
        m_w.shape.assert_has_rank(1)
        sigma_w = tf.convert_to_tensor(value=sigma_w)
        sigma_w.shape.assert_has_rank(2)

        activations_dtype = activations1.dtype
        if activations_dtype != tf.float64:
            activations1 = tf.cast(activations1, tf.float64)
            m_w = tf.cast(m_w, tf.float64)
            sigma_w = tf.cast(sigma_w, tf.float64)

        m = (tf.reduce_mean(input_tensor=activations1, axis=0),)
        m_w = (m_w,)
        # Calculate the unbiased covariance matrix of first activations.
        num_examples_real = tf.cast(tf.shape(input=activations1)[0], tf.float64)
        sigma = (num_examples_real / (num_examples_real - 1) *
                tfp.stats.covariance(activations1),)
        sigma_w = (sigma_w,)
        # m, m_w, sigma, sigma_w are tuples containing one or two elements: the first
        # element will be used to calculate the score value and the second will be
        # used to create the update_op. We apply the same operation on the two
        # elements to make sure their value is consistent.

        def _symmetric_matrix_square_root(mat, eps=1e-10):
            """Compute square root of a symmetric matrix.

            Note that this is different from an elementwise square root. We want to
            compute M' where M' = sqrt(mat) such that M' * M' = mat.

            Also note that this method **only** works for symmetric matrices.

            Args:
                mat: Matrix to take the square root of.
                eps: Small epsilon such that any element less than eps will not be square
                rooted to guard against numerical instability.

            Returns:
                Matrix square root of mat.
            """
            # Unlike numpy, tensorflow's return order is (s, u, v)
            s, u, v = tf.linalg.svd(mat)
            # sqrt is unstable around 0, just use 0 in such case
            si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
            # Note that the v returned by Tensorflow is v = V
            # (when referencing the equation A = U S V^T)
            # This is unlike Numpy which returns v = V^T
            return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)

        def trace_sqrt_product(sigma, sigma_v):
            """Find the trace of the positive sqrt of product of covariance matrices.

            '_symmetric_matrix_square_root' only works for symmetric matrices, so we
            cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
            ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

            Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
            We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
            Note the following properties:
            (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
                => eigenvalues(A A B B) = eigenvalues (A B B A)
            (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
                => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
            (iii) forall M: trace(M) = sum(eigenvalues(M))
                => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                            = sum(sqrt(eigenvalues(A B B A)))
                                            = sum(eigenvalues(sqrt(A B B A)))
                                            = trace(sqrt(A B B A))
                                            = trace(sqrt(A sigma_v A))
            A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
            use the _symmetric_matrix_square_root function to find the roots of these
            matrices.

            Args:
                sigma: a square, symmetric, real, positive semi-definite covariance matrix
                sigma_v: same as sigma

            Returns:
                The trace of the positive square root of sigma*sigma_v
            """

            # Note sqrt_sigma is called "A" in the proof above
            sqrt_sigma = _symmetric_matrix_square_root(sigma)

            # This is sqrt(A sigma_v A) above
            sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))

            return tf.linalg.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

        def _calculate_fid(m, m_w, sigma, sigma_w):
            """Returns the Frechet distance given the sample mean and covariance."""
            # Find the Tr(sqrt(sigma sigma_w)) component of FID
            sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

            # Compute the two components of FID.

            # First the covariance component.
            # Here, note that trace(A + B) = trace(A) + trace(B)
            trace = tf.linalg.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

            # Next the distance between means.
            mean = tf.reduce_sum(input_tensor=tf.math.squared_difference(
                m, m_w))  # Equivalent to L2 but more stable.
            fid = trace + mean
            if activations_dtype != tf.float64:
                fid = tf.cast(fid, activations_dtype)
            return fid

        result = tuple(
            _calculate_fid(m_val, m_w_val, sigma_val, sigma_w_val)
            for m_val, m_w_val, sigma_val, sigma_w_val in zip(m, m_w, sigma, sigma_w))
        return result[0]

    # Compute FID scores
    uint8_samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
    samples = np.clip(samples, 0., 1.)
    print("sample range: ", np.min(samples), np.max(samples))
    print("uint8 sample range: ", np.min(uint8_samples), np.max(uint8_samples))

    # Use inceptionV3 for images with resolution higher than 256.
    # inceptionv3 = image_size >= 256
    inceptionv3 = False
    inception_model = get_inception_model(inceptionv3=inceptionv3)
    # Load pre-computed dataset statistics.
    data_stats = load_dataset_stats(dataset)
    data_mu = data_stats["mu"]
    data_sigma = data_stats["sigma"]

    gc.collect()
    latents = run_inception_distributed(uint8_samples, inception_model, inceptionv3=inceptionv3)
    # Force garbage collection again before returning to JAX code
    gc.collect()

    # tmp_logits = latents["logits"].numpy()
    tmp_pool_3 = latents["pool_3"].numpy()

    print("tmpd_pool_3.shape: {}".format(tmp_pool_3.shape))
    # must have rank 2 to calculate distribution distances
    assert tmp_pool_3.shape[0] > 1

    # # Compute FID/KID/IS on individual inverse problem
    # if not inceptionv3:
    #     _inception_score = tfgan.eval.classifier_score_from_logits(tmp_logits)
    #     if idx:
    #         stable_inception_score = tfgan.eval.classifier_score_from_logits(tmp_logits[idx])
    # else:
    #     _inception_score = -1

    _fid = calculate_fid_helper(
        tmp_pool_3, data_mu, data_sigma)
    if idx:
        _stable_fid = tfgan.eval.frechet_classifier_distance_from_activations(
            tmp_pool_3[idx], data_mu, data_sigma)
    else: _stable_fid = None

    # if report_KID:
    #   # Hack to get tfgan KID work for eager execution.
    #   _tf_data_pools = tf.convert_to_tensor(data_pools)
    #   _tf_tmp_pools = tf.convert_to_tensor(tmp_pool_3)
    #   stable_tf_tmp_pools = tf.convert_to_tensor(tmp_pool_3[idx])
    #   _kid = tfgan.eval.kernel_classifier_distance_from_activations(
    #       _tf_data_pools, _tf_tmp_pools).numpy()
    #   stable_kid = tfgan.eval.kernel_classifier_distance_from_activations(
    #       _tf_data_pools, stable_tf_tmp_pools).numpy()
    #   del _tf_data_pools, _tf_tmp_pools, stable_tf_tmp_pools

    # print(f'{dataset} FID: {_fid}')
    # print(f'{dataset} KID: {_fid}')
    # if report_KID:
    #   return (_fid, _stable_fid), (_kid, stable_kid)
    # else:
    #   return _fid, _stable_fid
    return _fid, _stable_fid


def compute_fid_from_activations(samples, dataset=None, labels=None, idx=None, report_KID=False):
    # Compute FID scores
    uint8_samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
    samples = np.clip(samples, 0., 1.)
    print("sample range: ", np.min(samples), np.max(samples))
    print("uint8 sample range: ", np.min(uint8_samples), np.max(uint8_samples))

    # Use inceptionV3 for images with resolution higher than 256.
    # inceptionv3 = image_size >= 256
    inceptionv3 = False
    inception_model = get_inception_model(inceptionv3=inceptionv3)
    if dataset is not None:
        # Load pre-computed dataset statistics.
        data_stats = load_dataset_stats(dataset)
        data_pools = data_stats["pool_3"]
    elif dataset is None and labels is not None:
        uint8_labels = np.clip(labels * 255., 0, 255).astype(np.uint8)
        print("label range: ", np.min(labels), np.max(labels))
        print("uint8 label range: ", np.min(uint8_labels), np.max(uint8_labels))
        gc.collect()
        data_latents = run_inception_distributed(uint8_labels, inception_model, inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        data_pools = data_latents["pool_3"]
    else: raise ValueError("must supply dataset statistics or samples")

    gc.collect()
    latents = run_inception_distributed(uint8_samples, inception_model, inceptionv3=inceptionv3)
    # Force garbage collection again before returning to JAX code
    gc.collect()

    tmp_pool_3 = latents["pool_3"].numpy()

    # print("tmp_pool_3.shape: {}".format(tmp_pool_3.shape))
    # must have rank 2 to calculate distribution distances
    assert tmp_pool_3.shape[0] > 1

    # Compute FID/KID/IS on individual inverse problem
    _fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, tmp_pool_3)
    if idx:
        _stable_fid = tfgan.eval.frechet_classifier_distance_from_activations(
            data_pools, tmp_pool_3[idx])
    else: _stable_fid = None

    if report_KID:
      # Hack to get tfgan KID work for eager execution.
      _tf_data_pools = tf.convert_to_tensor(data_pools)
      _tf_tmp_pools = tf.convert_to_tensor(tmp_pool_3)
      _kid = tfgan.eval.kernel_classifier_distance_from_activations(
          _tf_data_pools, _tf_tmp_pools).numpy()
      if idx:
        stable_tf_tmp_pools = tf.convert_to_tensor(tmp_pool_3[idx])
        stable_kid = tfgan.eval.kernel_classifier_distance_from_activations(
            _tf_data_pools, stable_tf_tmp_pools).numpy()
        del stable_tf_tmp_pools
      else:
         stable_kid = _kid
      del _tf_data_pools, _tf_tmp_pools

    print(f'{dataset} FID: {_fid} SFID: {_stable_fid}')
    print(f'{dataset} KID: {_kid} SKID: {stable_kid}')
    if report_KID:
      return (_fid, _stable_fid), (_kid, stable_kid)
    return _fid, _stable_fid

device = 'cuda:0'
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

noise = '0.0'
image_size = 256
num_channel = 3

# task = 'gaussian_blur'
task = 'super_resolution4'
# task = 'uniform_blur'
# task = 'inpaintingbox'
# task = 'inpaintingrandom'
num_eval = 1000

root = './'
dataset = 'FFHQ'
# dataset = 'ImageNet'

return_tmpd = True
return_others = True
report_KID = True

label_root = Path(root + f'label')
# label_root = Path(root + f'{task}/label/{noise}')
tmpd_recon_root = Path(root + f'{task}/tmpd/{noise}')
dps_recon_root = Path(root + f'{task}/dps/{noise}')
pigdm_recon_root = Path(root + f'{task}/pigdm/{noise}')

labels_x = np.empty((num_eval, image_size, image_size, num_channel))
dps_x = np.empty((num_eval, image_size, image_size, num_channel))
pigdm_x = np.empty((num_eval, image_size, image_size, num_channel))
tmpd_x = np.empty((num_eval, image_size, image_size, num_channel))

psnr_tmpd_list = []
psnr_pigdm_list = []
psnr_dps_list = []

lpips_pigdm_list = []
lpips_tmpd_list = []
lpips_dps_list = []

mse_pigdm_list = []
mse_tmpd_list = []
mse_dps_list = []

for idx in tqdm(range(num_eval)):
    fname = str(idx).zfill(5)

    label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
    # delta_recon = plt.imread(delta_recon_root / f'{fname}.png')[:, :, :3]
    # normal_recon = plt.imread(normal_recon_root / f'{fname}.png')[:, :, :3]
    if return_tmpd:
      tmpd_recon = plt.imread(tmpd_recon_root / f'{fname}.png')[:, :, :3]
    if return_others:
      pigdm_recon = plt.imread(pigdm_recon_root / f'{fname}.png')[:, :, :3]
      dps_recon = plt.imread(dps_recon_root / f'{fname}.png')[:, :, :3]

    # As we are reading, need to save them.
    labels_x[idx, :, :, :] = label

    if return_tmpd: tmpd_x[idx, :, :, :] = tmpd_recon
    
    if return_others:
      dps_x[idx, :, :, :] = dps_recon
      pigdm_x[idx, :, :, :] = pigdm_recon

    if return_tmpd: psnr_tmpd = peak_signal_noise_ratio(label, tmpd_recon)
    if return_others:
      psnr_pigdm = peak_signal_noise_ratio(label, pigdm_recon)
      psnr_dps = peak_signal_noise_ratio(label, dps_recon)

    if return_tmpd: psnr_tmpd_list.append(psnr_tmpd)
    if return_others:
      psnr_dps_list.append(psnr_dps)
      psnr_pigdm_list.append(psnr_pigdm)

    if return_tmpd: mse_tmpd_recon = mse(tmpd_recon, label)
    if return_others:
      mse_pigdm_recon = mse(pigdm_recon, label)
      mse_dps_recon = mse(dps_recon, label)

    if return_tmpd: mse_tmpd_list.append(mse_tmpd_recon)
    if return_others:
      mse_pigdm_list.append(mse_pigdm_recon)
      mse_dps_list.append(mse_dps_recon)

    label = torch.from_numpy(label).permute(2, 0, 1).to(device)
    if return_tmpd: tmpd_recon = torch.from_numpy(tmpd_recon).permute(2, 0, 1).to(device)
    if return_others:
      pigdm_recon = torch.from_numpy(pigdm_recon).permute(2, 0, 1).to(device)
      dps_recon = torch.from_numpy(dps_recon).permute(2, 0, 1).to(device)

    label = label.view(1, 3, 256, 256) * 2. - 1.
    if return_tmpd: tmpd_recon = tmpd_recon.view(1, 3, 256, 256) * 2. - 1.
    if return_others:
      pigdm_recon = pigdm_recon.view(1, 3, 256, 256) * 2. - 1.
      dps_recon = dps_recon.view(1, 3, 256, 256) * 2. - 1.

    if return_tmpd: lpips_tmpd_list.append(loss_fn_vgg(tmpd_recon, label).detach().cpu().numpy())
    if return_others:
      lpips_pigdm_list.append(loss_fn_vgg(pigdm_recon, label).detach().cpu().numpy())
      lpips_dps_list.append(loss_fn_vgg(dps_recon, label).detach().cpu().numpy())

if return_tmpd:
  psnr_tmpd_list = np.array(psnr_tmpd_list)
  lpips_tmpd_list = np.array(lpips_tmpd_list)
  mse_tmpd_list = np.array(mse_tmpd_list)
  psnr_tmpd_avg = np.mean(psnr_tmpd_list)
  lpips_tmpd_avg = np.mean(lpips_tmpd_list)
  mse_tmpd_avg = np.mean(mse_tmpd_list)
  psnr_tmpd_std = np.std(psnr_tmpd_list)
  lpips_tmpd_std = np.std(lpips_tmpd_list)
  mse_tmpd_std = np.std(mse_tmpd_list)

if return_others:
  psnr_dps_list = np.array(psnr_dps_list)
  lpips_dps_list = np.array(lpips_dps_list)
  mse_dps_list = np.array(mse_dps_list)
  psnr_dps_avg = np.mean(psnr_dps_list)
  lpips_dps_avg = np.mean(lpips_dps_list)
  mse_dps_avg = np.mean(mse_dps_list)
  psnr_dps_std = np.std(psnr_dps_list)
  lpips_dps_std = np.std(lpips_dps_list)
  mse_dps_std = np.std(mse_dps_list)

  psnr_pigdm_list = np.array(psnr_pigdm_list)
  lpips_pigdm_list = np.array(lpips_pigdm_list)
  mse_pigdm_list = np.array(mse_pigdm_list)
  psnr_pigdm_avg = np.mean(psnr_pigdm_list)
  lpips_pigdm_avg = np.mean(lpips_pigdm_list)
  mse_pigdm_avg = np.mean(mse_pigdm_list)
  psnr_pigdm_std = np.std(psnr_pigdm_list)
  lpips_pigdm_std = np.std(lpips_pigdm_list)
  mse_pigdm_std = np.std(mse_pigdm_list)

if return_tmpd:
  idx_tmpd = np.argwhere(np.array(mse_tmpd_list) < 1.0).flatten()
  fraction_stable_tmpd = len(idx_tmpd) / len(mse_tmpd_list)

if return_others:
  idx_pigdm = np.argwhere(np.array(mse_pigdm_list) < 1.0).flatten()
  fraction_stable_pigdm = len(idx_pigdm) / len(mse_pigdm_list)

  idx_dps = np.argwhere(np.array(mse_dps_list) < 1.0).flatten()
  fraction_stable_dps = len(idx_dps) / len(mse_dps_list)

if report_KID:
  if return_tmpd:
    (tmpd_fid, _), (tmpd_kid, _) = compute_fid_from_activations(tmpd_x, dataset=None, labels=labels_x, idx=None, report_KID=True)
  if return_others:
    (dps_fid, _), (dps_kid, _) = compute_fid_from_activations(dps_x, dataset=None, labels=labels_x, idx=None, report_KID=True)
    (pigdm_fid, _), (pigdm_kid, _) = compute_fid_from_activations(pigdm_x, dataset=None, labels=labels_x, idx=None, report_KID=True)

else:
  if return_tmpd:
    tmpd_fid, _ = compute_fid_from_stats(tmpd_x, dataset, idx=None)
  if return_others:
    dps_fid, _ = compute_fid_from_stats(dps_x, dataset, idx=None)
    pigdm_fid, _ = compute_fid_from_stats(pigdm_x, dataset, idx=None)

if return_tmpd:
  tmpd_ssim = tf_ssim(tmpd_x, labels_x, max_val=1.0)
  ssim_tmpd_avg = np.mean(tmpd_ssim)
  ssim_tmpd_std = np.std(tmpd_ssim)

if return_others:
  dps_ssim = tf_ssim(dps_x, labels_x, max_val=1.0)
  pigdm_ssim = tf_ssim(pigdm_x, labels_x, max_val=1.0)
  ssim_dps_avg = np.mean(dps_ssim)
  ssim_dps_std = np.std(dps_ssim)
  ssim_pigdm_avg = np.mean(pigdm_ssim)
  ssim_pigdm_std = np.std(pigdm_ssim)

if report_KID:
  if return_others:
    print("PiGDM - stable: {}, FID: {:6e}, KID: {:6e}, LPIPS: {:6e}+/-{:3e}, PSNR: {:6e}+/-{:3e}, SSIM: {:6e}+/-{:3e}, MSE: {:6e}+/-{:3e}".format(
        fraction_stable_pigdm, pigdm_fid, pigdm_kid,
        lpips_pigdm_avg, lpips_pigdm_std, psnr_pigdm_avg, psnr_pigdm_std, ssim_pigdm_avg, ssim_pigdm_std, mse_pigdm_avg, mse_pigdm_std,
        ))
    print("DPSDDPM - stable: {}, FID: {:6e}, KID: {:6e}, LPIPS: {:6e}+/-{:3e}, PSNR: {:6e}+/-{:3e}, SSIM: {:6e}+/-{:3e}, MSE: {:6e}+/-{:3e}".format(
        fraction_stable_dps, dps_fid, dps_kid,
        lpips_dps_avg, lpips_dps_std, psnr_dps_avg, psnr_dps_std, ssim_dps_avg, ssim_dps_std, mse_dps_avg, mse_dps_std,
        ))
  if return_tmpd:
    print("TMPDDDPM - stable: {}, FID: {:6e}, KID: {:6e}, LPIPS: {:6e}+/-{:3e}, PSNR: {:6e}+/-{:3e}, SSIM: {:6e}+/-{:3e}, MSE: {:6e}+/-{:3e}".format(
        fraction_stable_tmpd, tmpd_fid, tmpd_kid,
        lpips_tmpd_avg, lpips_tmpd_std, psnr_tmpd_avg, psnr_tmpd_std, ssim_tmpd_avg, ssim_tmpd_std, mse_tmpd_avg, mse_tmpd_std,
        ))
else:
  if return_others:
    print("PiGDM - stable: {}, FID: {:6e}, LPIPS: {:6e}+/-{:3e}, PSNR: {:6e}+/-{:3e}, SSIM: {:6e}+/-{:3e}, MSE: {:6e}+/-{:3e}".format(
        fraction_stable_pigdm, pigdm_fid,
        lpips_pigdm_avg, lpips_pigdm_std, psnr_pigdm_avg, psnr_pigdm_std, ssim_pigdm_avg, ssim_pigdm_std, mse_pigdm_avg, mse_pigdm_std,
        ))
    print("DPSDDPM - stable: {}, FID: {:6e}, LPIPS: {:6e}+/-{:3e}, PSNR: {:6e}+/-{:3e}, SSIM: {:6e}+/-{:3e}, MSE: {:6e}+/-{:3e}".format(
        fraction_stable_dps, dps_fid,
        lpips_dps_avg, lpips_dps_std, psnr_dps_avg, psnr_dps_std, ssim_dps_avg, ssim_dps_std, mse_dps_avg, mse_dps_std,
        ))
  if return_tmpd:
    print("TMPDDDPM - stable: {}, FID: {:6e}, LPIPS: {:6e}+/-{:3e}, PSNR: {:6e}+/-{:3e}, SSIM: {:6e}+/-{:3e}, MSE: {:6e}+/-{:3e}".format(
        fraction_stable_tmpd, tmpd_fid,
        lpips_tmpd_avg, lpips_tmpd_std, psnr_tmpd_avg, psnr_tmpd_std, ssim_tmpd_avg, ssim_tmpd_std, mse_tmpd_avg, mse_tmpd_std,
        ))
