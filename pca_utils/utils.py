import torch
import torch.nn as nn

class CoordinateSearcher(nn.Module):
    def __init__(self, cur_d, PCA_num):
        super(CoordinateSearcher, self).__init__()
        # batch, [batch_size, c, r, r] -> [idx, c, r, r], idx represents the number of all samples
        d_cur_tensor = torch.concat(cur_d, dim=0)
        # [idx, c, r, r] -> [idx, c*r*r]
        d_cur_tensor = d_cur_tensor.reshape(d_cur_tensor.shape[0], -1)  
        # [idx, c*r*r] -> [idx,]
        d_cur_norm = torch.norm(d_cur_tensor, p=2, dim=1)  # d_cur_tensor L2 norm as the first coordinate initial value
        # [idx,] -> [1,]
        d_cur_norm_mean = torch.mean(d_cur_norm)

        coordinates_init = torch.zeros(PCA_num)  # [PCA_num,]
        coordinates_init[0] = d_cur_norm_mean
        self.coordinates = nn.Parameter(coordinates_init)  # [PCA_num,]

    def forward(self, U):
        # self.coordinates: [PCA_num,]; U: [batch_size, PCA_num, c*r*r]
        d_cur_searched = torch.matmul(U.transpose(1, 2), self.coordinates).squeeze(-1)  # [batch_size, c*r*r], corrected d_cur
        return d_cur_searched


# Correction d_cur during sampling
def correction_d(xt_cur, d_cur, d_buffer, xt_0, coordinates, PCA_num):
    # xt_cur, d_cur, xt_0: [batch_size, c, r, r]
    # d_buffer: before_NFE, [batch_size, c, r, r]
    U, batch_size, c, r = PCA_process(xt_cur, d_cur, d_buffer, xt_0, PCA_num)  # [batch_size, PCA_num, c*r*r]
    d_cur_searched = torch.matmul(U.transpose(1, 2), coordinates).squeeze(-1)  # [batch_size, c*r*r], corrected d_cur
    d_cur_searched = d_cur_searched.reshape(batch_size, c, r, r)

    return d_cur_searched


def early_stopping(running_loss_list, early_stop, tolerance=1e-5):
    if len(running_loss_list) < early_stop:
        return False
    
    # Get the most recent loss value
    recent_losses = running_loss_list[-early_stop:]
    # Check if there are consecutive losses that are not decreasing
    return all(abs(recent_losses[i] - recent_losses[i + 1]) <= tolerance for i in range(len(recent_losses) - 1))


def PCA_process(xt_cur, d_cur, d_buffer, xt_0, PCA_num, device_=None):
    # xt, d_cur, xt_0: [batch_size, c, r, r]
    # d_buffer: before_NFE, [batch_size, c, r, r]
    batch_size = xt_cur.shape[0]
    c = xt_cur.shape[1]
    r = xt_cur.shape[2]

    d_buffer.append(d_cur)  # The current d_cur is also put into the buffer to calculate PCA

    if device_ is not None:  # If it is a training process
        for index_i in range(len(d_buffer)):  # Put the buffer on the GPU and perform PCA later
            d_buffer[index_i] = d_buffer[index_i].to(device_)

    before_NFE_add1 = len(d_buffer)

    # before_NFE+1, [batch_size, c, r, r] -> [before_NFE+1, batch_size, c, r, r] -> [batch_size, before_NFE+1, c, r, r] -> [batch_size, before_NFE+1, c*r*r]
    d_buffer = torch.stack(d_buffer).permute(1, 0, 2, 3, 4).reshape(batch_size, before_NFE_add1, -1) 

    xt_cur = xt_cur.reshape(batch_size, -1)  # [batch_size, c*r*r]
    d_cur = d_cur.reshape(batch_size, -1)  # [batch_size, c*r*r]
    xt_0 = xt_0.reshape(batch_size, -1)  # [batch_size, c*r*r]

    v = d_cur  # [batch_size, c*r*r]
    v = v / torch.norm(v, p=2, dim=1, keepdim=True)  # [batch_size, c*r*r]

    # Used to unify the coordinate direction of each track
    tensor_in_flip = xt_cur - xt_0  # [batch_size, c*r*r]

    data_proj = d_buffer  # [batch_size, before_NFE+1, c*r*r]
    if PCA_num == 1:
        v1 = v  # [batch_size, c*r*r]
        u1 = v1
        U = torch.stack([u1], dim=1)  # [batch_size, 1, c*r*r]
    elif PCA_num == 2:
        U, S, V = torch.pca_lowrank(data_proj, q=1)
        v1, v2 = v, V[:, :, 0]  # [batch_size, c*r*r]
        if device_ is not None:  # If it is a training process
            v2 = v2.to('cpu')  # Stored in the CPU to prevent out of memory

        # Schmidt orthogonalization
        u1 = v1
        u2 = v2 - (torch.sum(u1 * v2, dim=1, keepdim=True) / torch.sum(u1 * u1, dim=1, keepdim=True)) * u1
        u2 = u2 / torch.norm(u2, p=2, dim=1, keepdim=True)
        # Flip the basis
        u2 = torch.where(torch.sum(tensor_in_flip * u2, dim=1, keepdim=True) > 0, u2, -u2)
        U = torch.stack([u1, u2], dim=1)  # [batch_size, 2, c*r*r]
    elif PCA_num == 3:
        U, S, V = torch.pca_lowrank(data_proj, q=2)
        v1, v2, v3 = v, V[:, :, 0], V[:, :, 1]  # [batch_size, c*r*r]
        if device_ is not None:  # If it is a training process
            v2, v3 = v2.to('cpu'), v3.to('cpu')  # Stored in the CPU to prevent out of memory

        # Schmidt orthogonalization
        u1 = v1
        u2 = v2 - (torch.sum(u1 * v2, dim=1, keepdim=True) / torch.sum(u1 * u1, dim=1, keepdim=True)) * u1
        u3 = v3 - (torch.sum(u1 * v3, dim=1, keepdim=True) / torch.sum(u1 * u1, dim=1, keepdim=True)) * u1 - (torch.sum(u2 * v3, dim=1, keepdim=True) / torch.sum(u2 * u2, dim=1, keepdim=True)) * u2
        u2 = u2 / torch.norm(u2, p=2, dim=1, keepdim=True)
        u3 = u3 / torch.norm(u3, p=2, dim=1, keepdim=True)
        # Flip the basis
        u2 = torch.where(torch.sum(tensor_in_flip * u2, dim=1, keepdim=True) > 0, u2, -u2)  # [batch_size, c*r*r]
        u3 = torch.where(torch.sum(tensor_in_flip * u3, dim=1, keepdim=True) > 0, u3, -u3)  # [batch_size, c*r*r]
        U = torch.stack([u1, u2, u3], dim=1)  # [batch_size, 3, c*r*r]
    elif PCA_num == 4:
        U, S, V = torch.pca_lowrank(data_proj, q=3)
        v1, v2, v3, v4 = v, V[:, :, 0], V[:, :, 1], V[:, :, 2]  # [batch_size, c*r*r]
        if device_ is not None:  # If it is a training process
            v2, v3, v4 = v2.to('cpu'), v3.to('cpu'), v4.to('cpu')  # Stored in the CPU to prevent out of memory

        # Schmidt orthogonalization
        u1 = v1
        u2 = v2 - (torch.sum(u1 * v2, dim=1, keepdim=True) / torch.sum(u1 * u1, dim=1, keepdim=True)) * u1
        u3 = v3 - (torch.sum(u1 * v3, dim=1, keepdim=True) / torch.sum(u1 * u1, dim=1, keepdim=True)) * u1 \
                - (torch.sum(u2 * v3, dim=1, keepdim=True) / torch.sum(u2 * u2, dim=1, keepdim=True)) * u2
        u4 = v4 - (torch.sum(u1 * v4, dim=1, keepdim=True) / torch.sum(u1 * u1, dim=1, keepdim=True)) * u1 \
                - (torch.sum(u2 * v4, dim=1, keepdim=True) / torch.sum(u2 * u2, dim=1, keepdim=True)) * u2 \
                - (torch.sum(u3 * v4, dim=1, keepdim=True) / torch.sum(u3 * u3, dim=1, keepdim=True)) * u3
        
        u2 = u2 / torch.norm(u2, p=2, dim=1, keepdim=True)
        u3 = u3 / torch.norm(u3, p=2, dim=1, keepdim=True)
        u4 = u4 / torch.norm(u4, p=2, dim=1, keepdim=True)

        # Flip the basis
        u2 = torch.where(torch.sum(tensor_in_flip * u2, dim=1, keepdim=True) > 0, u2, -u2)
        u3 = torch.where(torch.sum(tensor_in_flip * u3, dim=1, keepdim=True) > 0, u3, -u3)
        u4 = torch.where(torch.sum(tensor_in_flip * u4, dim=1, keepdim=True) > 0, u4, -u4)

        U = torch.stack([u1, u2, u3, u4], dim=1)  # [batch_size, 4, c*r*r]
    else:
        raise ValueError(f'PCA_num should be >=1 and <= 4, now: {PCA_num}')
    
    return U, batch_size, c, r

def pre_PCA_process(criterion, cur_searched_xt, cur_d_cur, xt_next_gt, buffer_d_cur, x0, stu_t_step, PCA_num, time_point, device_):
    # cur_searched_xt, cur_d_cur, xt_next_gt, x0: batch, [batch_size, c, r, r]. Current xt (approximate xt_gt); d_cur corresponding to Current xt; next_xt_gt; init xt
    # buffer_d_cur: before_NFE, batch, [batch_size, c, r, r]

    if len(buffer_d_cur) == 0:  # If buffer_d_cur is empty, initialize it to batch []
        buffer_d_cur = [[] for _ in range(len(cur_searched_xt))]
    else:
        # before_NFE, batch, [batch_size, c, r, r] -> batch, before_NFE, [batch_size, c, r, r]
        buffer_d_cur = [list(before_NFE) for before_NFE in zip(*buffer_d_cur)]

    loss_origin_batch_list = []
    xt_cur_batch_list = []
    U_batch_list = []
    xt_next_batch_list = []
    for xt_cur, d_cur, xt_next, d_buffer, xt_0 in zip(cur_searched_xt, cur_d_cur, xt_next_gt, buffer_d_cur, x0):
        cur_stu_t_step = stu_t_step[time_point].to('cpu')
        next_stu_t_step = stu_t_step[time_point+1].to('cpu')
        # Euler step.
        xt_next_origin = xt_cur + d_cur * (next_stu_t_step - cur_stu_t_step)  # [batch_size, c, r, r], xt_next before correction, used to calculate the original loss
        loss_origin = criterion(xt_next_origin, xt_next)  # Original loss

        loss_origin_batch_list.append(loss_origin.item())
        xt_cur_batch_list.append(xt_cur)  # [batch_size, c, r, r]
        xt_next_batch_list.append(xt_next)  # [batch_size, c, r, r]

        U, _, _, _ = PCA_process(xt_cur, d_cur, d_buffer.copy(), xt_0, PCA_num, device_)  # [batch_size, PCA_num, c*r*r], copy() prevent the original buffer from being modified
        U_batch_list.append(U)
        
    return U_batch_list, xt_cur_batch_list, xt_next_batch_list, loss_origin_batch_list


def do_search(dist, model, cur_searched_xt, cur_d_cur, xt_next_gt, buffer_d_cur, x0, stu_t_step, epochs, early_stop, PCA_num, time_point, optimizer, criterion, device_):
    c = cur_searched_xt[0].shape[1]
    r = cur_searched_xt[0].shape[2]
    dist.print0("Start PCA to get the principal components...")
    U_batch_list, xt_cur_batch_list, xt_next_batch_list, loss_origin_batch_list = pre_PCA_process(criterion, cur_searched_xt, cur_d_cur, xt_next_gt, buffer_d_cur, x0, stu_t_step, PCA_num, time_point, device_)

    d_cur_searched_list = []
    xt_next_searched_list = []
    running_loss_list = []

    dist.print0("Start search to get the coordinates...")
    for epoch in range(epochs):
        running_loss = 0.0
        running_loss_origin = 0.0
        sum_batch = 0

        d_cur_searched_list = []
        xt_next_searched_list = []

        for idx, (U, xt_cur, xt_next, loss_origin) in enumerate(zip(U_batch_list, xt_cur_batch_list, xt_next_batch_list, loss_origin_batch_list)):
            batch_size = xt_cur.shape[0]
            xt_cur = xt_cur.to(device_)
            xt_next = xt_next.to(device_)
            U = U.to(device_)

            optimizer.zero_grad()
            d_cur_searched = model(U).reshape(batch_size, c, r, r)

            # Euler step.
            xt_next_searched = xt_cur + d_cur_searched * (stu_t_step[time_point+1] - stu_t_step[time_point])  # [batch_size, c, r, r], corrected xt_next

            loss = criterion(xt_next_searched, xt_next)
            loss.backward()
            optimizer.step()

            # Extract from the computation graph
            d_cur_searched_list.append(d_cur_searched.detach().to('cpu'))  # Returns the corrected d_cur and xt_next for the next time point, shape: [batch_size, c, r, r]
            xt_next_searched_list.append(xt_next_searched.detach().to('cpu'))
            
            # Statistical losses
            running_loss += loss.item()
            running_loss_origin += loss_origin
            sum_batch += 1

        loss_epoch = running_loss / sum_batch
        loss_origin_epoch = running_loss_origin / sum_batch

        running_loss_list.append(loss_epoch)
        if early_stopping(running_loss_list, early_stop):  # If the loss does not decrease in successive 'early_stops', stop early
            dist.print0(f"Epoch: {epoch}, STEP: {time_point}, Loss_origin: {loss_origin_epoch:.4f}, Loss: {loss_epoch:.4f}")
            dist.print0("early_stopping...")
            return d_cur_searched_list, xt_next_searched_list, (loss_origin_epoch, loss_epoch)
        else:
            if epoch % 10 == 0 or epoch == epochs-1:
                dist.print0(f"Epoch: {epoch}, STEP: {time_point}, Loss_origin: {loss_origin_epoch:.4f}, Loss: {loss_epoch:.4f}")

    return d_cur_searched_list, xt_next_searched_list, (loss_origin_epoch, loss_epoch)