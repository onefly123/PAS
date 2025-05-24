#----------------------------------------------------------------------------
# Get the denoised output from the pre-trained diffusion models.

def get_denoised(net, x, t, class_labels=None, condition=None, unconditional_condition=None):
    if hasattr(net, 'guidance_type'):       # models from LDM and Stable Diffusion
        denoised = net(x, t, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = net(x, t, class_labels=class_labels)
    return denoised

#----------------------------------------------------------------------------
def euler_step(device_, next_xt, next_time, model_source, net, class_labels=None, condition=None, unconditional_condition=None):
    next_xt = next_xt.to(device_)
    if model_source == 'ldm':
        denoised = get_denoised(net, next_xt, next_time, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = get_denoised(net, next_xt, next_time, class_labels=class_labels)
    
    next_d = (next_xt - denoised) / next_time

    return next_d.to('cpu')

#----------------------------------------------------------------------------
def ipndm_step(device_, next_xt, next_time, model_source, net, time_point, buffer_ipndm, max_order=4, class_labels=None, condition=None, unconditional_condition=None):
    next_xt = next_xt.to(device_)
    assert max_order >= 1 and max_order <= 4
    if model_source == 'ldm':
        denoised = get_denoised(net, next_xt, next_time, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = get_denoised(net, next_xt, next_time, class_labels=class_labels)
    
    next_d = (next_xt - denoised) / next_time
    order = min(max_order, time_point+2)  # Next time point

    next_d = next_d.to('cpu')
    if order == 1:      # First Euler step. will not appear
        next_d_update = next_d
    elif order == 2:    # Use one history point.
        next_d_update = (3 * next_d - buffer_ipndm[-1]) / 2
    elif order == 3:    # Use two history points.
        next_d_update = (23 * next_d - 16 * buffer_ipndm[-1] + 5 * buffer_ipndm[-2]) / 12
    elif order == 4:    # Use three history points.
        next_d_update = (55 * next_d - 59 * buffer_ipndm[-1] + 37 * buffer_ipndm[-2] - 9 * buffer_ipndm[-3]) / 24
    
    if len(buffer_ipndm) == max_order - 1:
        for k in range(max_order - 2):
            buffer_ipndm[k] = buffer_ipndm[k+1]
        buffer_ipndm[-1] = next_d
    else:
        buffer_ipndm.append(next_d)

    return next_d_update


def deis_step(coeff_list, device_, next_xt, next_time, next_next_time, model_source, net, time_point, buffer_ipndm, max_order=4, class_labels=None, condition=None, unconditional_condition=None):
    next_xt = next_xt.to(device_)
    assert max_order >= 1 and max_order <= 4
    assert coeff_list is not None
    
    if model_source == 'ldm':
        denoised = get_denoised(net, next_xt, next_time, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = get_denoised(net, next_xt, next_time, class_labels=class_labels)
    
    next_d = (next_xt - denoised) / next_time
    order = min(max_order, time_point+2)  # Next time point

    next_d = next_d.to('cpu')
    next_next_time, next_time = next_next_time.to('cpu'), next_time.to('cpu')
    if order == 1:
        next_d_update = next_d
    elif order == 2:
        coeff_cur, coeff_prev1 = coeff_list[time_point+1]
        coeff_cur, coeff_prev1 = coeff_cur.to('cpu'), coeff_prev1.to('cpu')
        next_d_update = (coeff_cur * next_d + coeff_prev1 * buffer_ipndm[-1]) / (next_next_time - next_time)
    elif order == 3:
        coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[time_point+1]
        coeff_cur, coeff_prev1, coeff_prev2 = coeff_cur.to('cpu'), coeff_prev1.to('cpu'), coeff_prev2.to('cpu')
        next_d_update = (coeff_cur * next_d + coeff_prev1 * buffer_ipndm[-1] + coeff_prev2 * buffer_ipndm[-2]) / (next_next_time - next_time)
    elif order == 4:
        coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[time_point+1]
        coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_cur.to('cpu'), coeff_prev1.to('cpu'), coeff_prev2.to('cpu'), coeff_prev3.to('cpu')
        next_d_update = (coeff_cur * next_d + coeff_prev1 * buffer_ipndm[-1] + coeff_prev2 * buffer_ipndm[-2] + coeff_prev3 * buffer_ipndm[-3]) / (next_next_time - next_time)


    if len(buffer_ipndm) == max_order - 1:
        for k in range(max_order - 2):
            buffer_ipndm[k] = buffer_ipndm[k+1]
        buffer_ipndm[-1] = next_d
    else:
        buffer_ipndm.append(next_d)
    
    return next_d_update