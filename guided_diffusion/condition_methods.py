from abc import ABC, abstractmethod
import torch
import functorch

__CONDITIONING_METHOD__ = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == "gaussian":
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        elif self.noiser.__name__ == "poisson":
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


@register_conditioning_method(name="vanilla")
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t


@register_conditioning_method(name="projection")
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name="mcg")
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(
        self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs
    ):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale

        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm


@register_conditioning_method(name="ps")
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale
        del norm_grad
        return x_t, norm


@register_conditioning_method(name="ps+")
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get("num_sampling", 5)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling

        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name="pig")
class PsuedoInverseGuided(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        """
        NOTE: To be used with `tmpdtorch.gaussian_diffusion.pigdm_sample_loop`.
        NOTE: In this class, I have tried to reproduce https://openreview.net/forum?id=9_gsMA8MRKQ
        although the authors do not release code, and do not mention key aspects such as use of
        static thresholding (clipping) required for the stability of the algorithm.
        I have followed all the implementation details that their reproducibility statement gives, namely
        " (ii) We include a detailed description of our algorithm in Algorithm 1. (iii) We discuss
        all the key hyperparameters and evaluation metrics to reproduce our experiments in Sec. 3.3 and
        App. B. (iv) We provide more explanations to some statements in the main paper in App. A.2 to A.5."
        Whilst this seems to produce reasonable numerical results, there are some artifacts on the images
        that are not shown in the examples in the publication.
        """
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get("num_sampling", 5)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_t, measurement, estimate_x_0, r, v, noise_std, **kwargs):
        def estimate_h_x_0(x_t):
            x_0, eps = estimate_x_0(x_t)
            return self.operator.forward(x_0, **kwargs), (x_0, eps)

        if self.noiser.__name__ == "gaussian":
            h_x_0, vjp_estimate_h_x_0, (x_0, eps) = functorch.vjp(
                estimate_h_x_0, x_t, has_aux=True
            )
            r = v * self.scale / (v + self.scale)
            C_yy = 1.0 + noise_std**2 / r
            difference = measurement - h_x_0
            norm = torch.linalg.norm(difference)
            ls = vjp_estimate_h_x_0(difference / C_yy)[0]
            del C_yy
            del difference
        else:
            raise NotImplementedError

        return x_0, eps, ls, norm


@register_conditioning_method(name="altpig")
class AltPsuedoInverseGuided(ConditioningMethod):
    """
    NOTE: To be used with `tmpdtorch.gaussian_diffusion.tmpd_sample_loop`.
    NOTE: An alternative implementation of PiGDM that is more numerically stable, and does not require
    clipping, although different in some aspects to the authors' Algorithm 1 in that it more closely
    follows the TMPD algorithm.
    """

    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get("num_sampling", 5)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_t, measurement, estimate_x_0, r, v, noise_std, **kwargs):
        def estimate_h_x_0(x_t):
            x_0 = estimate_x_0(x_t)
            return self.operator.forward(x_0, **kwargs), x_0

        if self.noiser.__name__ == "gaussian":
            h_x_0, vjp_estimate_h_x_0, x_0 = functorch.vjp(
                estimate_h_x_0, x_t, has_aux=True
            )
            r = v * self.scale / (v + self.scale)
            C_yy = 1.0 + noise_std**2 / r
            difference = measurement - h_x_0
            norm = torch.linalg.norm(difference)
            ls = vjp_estimate_h_x_0(difference / C_yy)[0]
            x_0 = x_0 + ls
        else:
            raise NotImplementedError

        return x_0, norm


@register_conditioning_method(name="tmp")
class TweedieMomentProjection(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get("num_sampling", 5)
        # self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_t, measurement, estimate_x_0, r, v, noise_std, **kwargs):
        def estimate_h_x_0(x_t):
            x_0, model_var_values = estimate_x_0(x_t)
            return self.operator.forward(x_0, **kwargs), (x_0, model_var_values)
            # return self.operator.forward(x_0, **kwargs)

        if self.noiser.__name__ == "gaussian":
            # Due to the structure of this code, the condition operator is not accesible unless inside from in the conditioning method. That's why the analysis is here
            # Since functorch 1.1.1 is not compatible with this
            # functorch 0.1.1 (unstable; works with PyTorch 1.11) does not work with autograd.Function, which is what the model is written in. It can be rewritten, or package environment needs to be solved.
            # h_x_0, vjp = torch.autograd.functional.vjp(estimate_h_x_0, x_t, self.operator.forward(torch.ones_like(x_t), **kwargs))
            # difference = measurement - h_x_0
            # norm = torch.linalg.norm(difference)
            # C_yy = self.operator.forward(vjp, **kwargs) + noise_std**2 / ratio
            # _, ls = torch.autograd.functional.vjp(estimate_h_x_0, x_t, difference / C_yy)
            # x_0 = estimate_x_0(x_t)

            # NOTE: This standing functorch way seems to be only slightly faster (163 seconds instead of 188 seconds)
            # NOTE: In torch, usually our method is up to 2x slower than dps due to the extra vjp
            # # h_x_0, vjp_estimate_h_x_0, x_0 = torch.func.vjp(estimate_h_x_0, x_t, has_aux=True)
            h_x_0, vjp_estimate_h_x_0, (x_0, model_var_values) = functorch.vjp(
                estimate_h_x_0, x_t, has_aux=True
            )
            C_yy = (
                self.operator.forward(
                    vjp_estimate_h_x_0(
                        self.operator.forward(torch.ones_like(x_0), **kwargs)
                    )[0],
                    **kwargs,
                )
                + noise_std**2 / r
            )
            difference = measurement - h_x_0
            norm = torch.linalg.norm(difference)
            ls = vjp_estimate_h_x_0(difference / C_yy)[0]
            del C_yy

            x_0 = x_0 + ls  # commenting it out shows that rest of the code works
            del ls
        else:
            raise NotImplementedError

        return x_0, norm, model_var_values

