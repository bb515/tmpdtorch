from functools import partial
import os
import yaml

import torch
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import _get_dataset
from util.img_utils import mask_generator, to_numpy, clear_color
from util.logger import get_logger

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os

import numpy as np


FLAGS = flags.FLAGS

flags.DEFINE_string("model_config", None, "Model config.")
flags.DEFINE_string("diffusion_config", None, "Diffusion config.")
flags.DEFINE_string("task_config", None, "Task config.")
flags.DEFINE_integer("gpu", 0, "GPU")
flags.DEFINE_string("save_dir", "./results", "Save directory.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "Dataset and training configuration (from tmpdjax code).",
    lock_config=True,
)

flags.mark_flags_as_required(
    ["model_config", "diffusion_config", "task_config", "config"]
)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(argv):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_config', type=str)
    # parser.add_argument('--diffusion_config', type=str)
    # parser.add_argument('--task_config', type=str)
    # parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--save_dir', type=str, default='./results')
    # BB add some config so that I can load the dataset

    # FLAGS = parser.parse_args()

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(FLAGS.model_config)
    diffusion_config = load_yaml(FLAGS.diffusion_config)
    task_config = load_yaml(FLAGS.task_config)

    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    # "learn_sigma must be the same for model and diffusion configuartion."

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config["measurement"]
    operator = get_operator(device=device, **measure_config["operator"])
    noiser = get_noise(**measure_config["noise"])
    logger.info(
        f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}"
    )

    # out_path = os.path.join(FLAGS.save_dir, measure_config['operator']['name'] + measure_config['mask_opt']['mask_type'])  # for inpainting
    # out_path = os.path.join(FLAGS.save_dir, measure_config['operator']['name'] + str(measure_config['operator']['scale_factor']))   # for superresolution
    out_path = os.path.join(FLAGS.save_dir, measure_config['operator']['name'])  # everything else

    return_tmpd = True
    return_dps = True
    return_pigdm = True

    if return_tmpd:
        # Prepare tmpd condition method
        tmpd_cond_config = task_config["tmpdconditioning"]
        tmpd_cond_method = get_conditioning_method(
            tmpd_cond_config["method"], operator, noiser
        )
        tmpd_measurement_cond_fn = tmpd_cond_method.conditioning
        logger.info(
            f"tmpd Conditioning method : {task_config['tmpdconditioning']['method']}"
        )

    if return_dps:
        # Prepare dps conditioning method
        dps_cond_config = task_config["dpsconditioning"]
        dps_cond_method = get_conditioning_method(
            dps_cond_config["method"], operator, noiser, **dps_cond_config["params"]
        )
        dps_measurement_cond_fn = dps_cond_method.conditioning
        logger.info(f"Conditioning method : {task_config['dpsconditioning']['method']}")

    if return_pigdm:
        # Prepare pigdm condition method
        pigdm_cond_config = task_config["pigdmconditioning"]
        pigdm_cond_method = get_conditioning_method(
            pigdm_cond_config["method"], operator, noiser
        )
        pigdm_measurement_cond_fn = pigdm_cond_method.conditioning
        logger.info(
            f"pigdm Conditioning method : {task_config['pigdmconditioning']['method']}"
        )

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    # sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
    if return_tmpd:
        tmpd_sample_fn = partial(
            sampler.tmpd_sample_loop,
            config=measure_config,
            model=model,
            measurement_cond_fn=tmpd_measurement_cond_fn,
        )
    if return_dps:
        dps_sample_fn = partial(
            sampler.p_sample_loop,
            config=FLAGS.config,
            model=model,
            measurement_cond_fn=dps_measurement_cond_fn,
        )
    if return_pigdm:
        pigdm_sample_fn = partial(
            # sampler.pigdm_sample_loop,
            sampler.reddiff_pigdm_sample_loop,
            config=measure_config,
            model=model,
            measurement_cond_fn=pigdm_measurement_cond_fn,
        )

    # os.makedirs(os.path.join(out_path, 'label'), exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ["input", "dps", "pigdm", "tmpd", "label"]:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
        for noise in ["0.01", "0.05", "0.1", "0.2"]:
            os.makedirs(os.path.join(out_path, img_dir, noise), exist_ok=True)

    os.makedirs(os.path.join(out_path, "progress"), exist_ok=True)

    # Prepare TF dataloader
    tmpd_config = FLAGS.config
    num_devices = 1
    # print(tmpd_config.data.tfrecords_path)
    # print(tmpd_config.eval.batch_size)
    _, eval_ds, _ = _get_dataset(num_devices, tmpd_config)

    # Exception) In case of inpainting, we need to generate a mask
    if measure_config["operator"]["name"] == "inpainting":
        mask_gen = mask_generator(**measure_config["mask_opt"])

    # Do Inference
    for i, batch in enumerate(iter(eval_ds)):
        print(i)
        if i==1000: assert 0  # break at 1k to evaluate FID-1k
        if tmpd_config.data.dataset == "ImageNet":
            ref_img = batch[0].to(device='cuda:0')
        else:
            ref_img = batch['image'][0]
            # Convert to torch.Tensor
            ref_img = torch.Tensor(np.array(ref_img).transpose(0, 3, 1, 2)).to(device='cuda:0')
        print("min ", ref_img.min(), "max", ref_img.max())

        # Exception) In case of inpainging,
        if measure_config["operator"]["name"] == "inpainting":
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            if return_tmpd:
                tmpd_measurement_cond_fn = partial(tmpd_cond_method.conditioning, mask=mask)
                tmpd_sample_fn = partial(
                    tmpd_sample_fn, measurement_cond_fn=tmpd_measurement_cond_fn
                )
            if return_dps:
                dps_measurement_cond_fn = partial(dps_cond_method.conditioning, mask=mask)
                dps_sample_fn = partial(
                    dps_sample_fn, measurement_cond_fn=dps_measurement_cond_fn
                )
            if return_pigdm:
                pigdm_measurement_cond_fn = partial(
                    pigdm_cond_method.conditioning, mask=mask
                )
                pigdm_sample_fn = partial(
                    pigdm_sample_fn, measurement_cond_fn=pigdm_measurement_cond_fn
                )

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else:
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)

        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()

        fname = str(i).zfill(5) + '.png'
        noise = str(measure_config['noise']['sigma'])
        # plt.imsave(os.path.join(out_path, 'input', noise, fname), to_numpy(y_n))
        plt.imsave(os.path.join(out_path, 'input', noise, fname), clear_color(y_n))
        # plt.imsave(os.path.join(out_path, 'label', noise, fname), to_numpy(ref_img))
        plt.imsave(os.path.join(out_path, 'label', noise, fname), clear_color(ref_img))

        if return_tmpd:
            tmpd_sample = tmpd_sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)
            # plt.imsave(os.path.join(out_path, 'tmpd', noise, fname), to_numpy(tmpd_sample))
            plt.imsave(os.path.join(out_path, 'tmpd', noise, fname), clear_color(tmpd_sample))
        if return_dps:
            dps_sample = dps_sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)
            # plt.imsave(os.path.join(out_path, 'dps', noise, fname), to_numpy(dps_sample))
            plt.imsave(os.path.join(out_path, 'dps', noise, fname), clear_color(dps_sample))
        if return_pigdm:
            pigdm_sample = pigdm_sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)
            # plt.imsave(os.path.join(out_path, 'pigdm', noise, fname), to_numpy(pigdm_sample))
            plt.imsave(os.path.join(out_path, 'pigdm', noise, fname), clear_color(pigdm_sample))


if __name__ == "__main__":
    app.run(main)
