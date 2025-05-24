# Implementation of various ODE solvers for diffusion models.

import torch
from solver_utils import *

#----------------------------------------------------------------------------
# Get the denoised output from the pre-trained diffusion models.

def get_denoised(net, x, t, class_labels=None, condition=None, unconditional_condition=None):
    if hasattr(net, 'guidance_type'):       # models from LDM and Stable Diffusion
        denoised = net(x, t, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = net(x, t, class_labels=class_labels)
    return denoised

#----------------------------------------------------------------------------

@torch.no_grad()
def euler_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_eps=False, 
    t_steps=None,
    **kwargs
):  
    """
    Euler sampler (equivalent to the DDIM sampler: https://arxiv.org/abs/2010.02502).

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    xt_list = []
    x_next = latents * t_steps[0]
    xt_list.append(x_next)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):   # 0, ..., N-1
        x_cur = x_next

        # Euler step.
        denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_cur - denoised) / t_cur

        if i == 0:
            d_first = d_cur

        x_next = x_cur + (t_next - t_cur) * d_cur

        xt_list.append(x_next)
    
    return xt_list, d_first, t_steps

#----------------------------------------------------------------------------

@torch.no_grad()
def heun_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False,
    return_eps=False, 
    t_steps=None, 
    **kwargs
):
    """
    Heun's second sampler. Introduced in EDM: https://arxiv.org/abs/2206.00364.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    xt_list = []
    x_next = latents * t_steps[0]
    xt_list.append(x_next)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next

        # Euler step.
        denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_cur - denoised) / t_cur

        if i == 0:
            d_first = d_cur

        x_next = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction.
        denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_prime = (x_next - denoised) / t_next
        x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        xt_list.append(x_next)

    return xt_list, d_first, t_steps

#----------------------------------------------------------------------------

@torch.no_grad()
def dpm_2_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_eps=False, 
    r=0.5, 
    t_steps=None,
    **kwargs
):
    """
    DPM-Solver-2 sampler: https://arxiv.org/abs/2206.00927.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
        r: A `float`. The hyperparameter controlling the location of the intermediate time step. r=0.5 recovers the original DPM-Solver-2.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    
    # Main sampling loop.
    xt_list = []
    x_next = latents * t_steps[0]
    xt_list.append(x_next)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next

        denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_cur - denoised) / t_cur

        if i == 0:
            d_first = d_cur

        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur

        # Apply 2nd order correction.
        denoised = get_denoised(net, x_next, t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_prime = (x_next - denoised) / t_mid
        x_next = x_cur + (t_next - t_cur) * ((1 / (2*r)) * d_prime + (1 - 1 / (2*r)) * d_cur)

        xt_list.append(x_next)

    return xt_list, d_first, t_steps