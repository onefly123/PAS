# Implementation of various ODE solvers for diffusion models.

import torch
from solver_utils import *
from pca_utils import correction_d
from torch import autocast
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
    coordinates_dict, PCA_num, use_xT,  # pas_solver parameters
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
    t_steps=None,
    **kwargs
):  
    """
    Euler sampler (equivalent to the DDIM sampler: https://arxiv.org/abs/2010.02502).

    Args:
        coordinates_dict: A `dict`. The coordinates of the points to be corrected.
        PCA_num: A `int`. The number of PCA components.
        use_xT: A `str`. Whether to use the initial sample `xT` to span the sampling space.
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
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    d_list = []
    x_next = latents * t_steps[0]
    if use_xT == "yes":
        d_list.append(x_next)
    x_start = x_next
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):   # 0, ..., N-1
        x_cur = x_next

        if kwargs['model_source'] == 'ldm':
            with autocast("cuda"):  # mixed precision
                denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            
        d_cur = (x_cur - denoised) / t_cur

        if i in coordinates_dict:
            coordinates = coordinates_dict[i].to(d_cur.device)
            d_cur = correction_d(x_cur, d_cur, d_list, x_start, coordinates, PCA_num)
            d_list.append(d_cur)
        else:
            d_list.append(d_cur)

        # Euler step.
        x_next = x_cur + (t_next - t_cur) * d_cur
    
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def ipndm_sampler(
    coordinates_dict, PCA_num, use_xT,  # pas_solver parameters
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
    max_order=4, 
    t_steps=None,
    **kwargs
):
    """
    Improved PNDM sampler: https://arxiv.org/abs/2204.13902.

    Args:
        coordinates_dict: A `dict`. The coordinates of the points to be corrected.
        PCA_num: A `int`. The number of PCA components.
        use_xT: A `str`. Whether to use the initial sample `xT` to span the sampling space.
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
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 4
    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    d_list = []
    x_next = latents * t_steps[0]
    if use_xT == "yes":
        d_list.append(x_next)
    x_start = x_next
    buffer_model = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):        # 0, ..., N-1
        x_cur = x_next

        if kwargs['model_source'] == 'ldm':
            with autocast("cuda"):
                denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_cur - denoised) / t_cur

        order = min(max_order, i+1)
        if order == 1:      # First Euler step.
            d_update = d_cur
        elif order == 2:    # Use one history point.
            d_update = (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:    # Use two history points.
            d_update = (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:    # Use three history points.
            d_update = (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        
        if i in coordinates_dict:
            coordinates = coordinates_dict[i].to(d_cur.device)
            d_update = correction_d(x_cur, d_update, d_list, x_start, coordinates, PCA_num)
            d_list.append(d_update)
        else:
            d_list.append(d_update)

        # Euler step.
        x_next = x_cur + (t_next - t_cur) * d_update

        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur
        else:
            buffer_model.append(d_cur)
        
    return x_next


#----------------------------------------------------------------------------
@torch.no_grad()
def deis_sampler(
    coordinates_dict, PCA_num, use_xT,  # pas_solver parameters
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
    max_order=4, 
    coeff_list=None, 
    t_steps=None,
    **kwargs
):
    """
    A pytorch implementation of DEIS: https://arxiv.org/abs/2204.13902.

    Args:
        coordinates_dict: A `dict`. The coordinates of the points to be corrected.
        PCA_num: A `int`. The number of PCA components.
        use_xT: A `str`. Whether to use the initial sample `xT` to span the sampling space.
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
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
        coeff_list: A `list`. The pre-calculated coefficients for DEIS sampling.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 4
    assert coeff_list is not None
    
    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    d_list = []
    x_next = latents * t_steps[0]
    if use_xT == "yes":
        d_list.append(x_next)
    x_start = x_next
    buffer_model = []

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        
        if kwargs['model_source'] == 'ldm':
            with autocast("cuda"):
                denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_cur - denoised) / t_cur
        
        order = min(max_order, i+1)
        if order == 1:          # First Euler step.
            d_update = d_cur
        elif order == 2:        # Use one history point.
            coeff_cur, coeff_prev1 = coeff_list[i]
            d_update = (coeff_cur * d_cur + coeff_prev1 * buffer_model[-1]) / (t_next - t_cur)
        elif order == 3:        # Use two history points.
            coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
            d_update = (coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]) / (t_next - t_cur)
        elif order == 4:        # Use three history points.
            coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
            d_update = (coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]) / (t_next - t_cur)
        
        if i in coordinates_dict:
            coordinates = coordinates_dict[i].to(d_cur.device)
            d_update = correction_d(x_cur, d_update, d_list, x_start, coordinates, PCA_num)
            d_list.append(d_update)
        else:
            d_list.append(d_update)

        # Euler step.
        x_next = x_cur + (t_next - t_cur) * d_update

        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur
        else:
            buffer_model.append(d_cur)

    return x_next