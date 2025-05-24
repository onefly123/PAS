import ast
import os
import re
import csv
import click
import tqdm
import pickle
import torch
import PIL.Image
import dnnlib
import solvers
import tea_solvers
import plug_solvers
import solver_utils
from torch import autocast
from torch_utils import distributed as dist
from torchvision.utils import make_grid, save_image
from torch_utils.download_util import check_file_by_key
import time
import torch.optim as optim
import torch.nn as nn
from pca_utils import CoordinateSearcher, do_search
import lpips

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Load pre-trained models from the LDM codebase (https://github.com/CompVis/latent-diffusion) 
# and Stable Diffusion codebase (https://github.com/CompVis/stable-diffusion)

def load_ldm_model(config, ckpt, verbose=False):
    from models.ldm.util import instantiate_from_config
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        dist.print0(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

#----------------------------------------------------------------------------

def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None):
    model_path, classifier_path = check_file_by_key(dataset_name)
    dist.print0(f'Loading the pre-trained diffusion model from "{model_path}"...')

    if dataset_name in ['cifar10', 'ffhq', 'afhqv2', 'imagenet64']:         # models from EDM
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)
        net.sigma_min = 0.002
        net.sigma_max = 80.0
        model_source = 'edm'
    elif dataset_name in ['lsun_bedroom', 'lsun_cat']:                      # models from Consistency Models
        from models.cm.cm_model_loader import load_cm_model
        from models.networks_edm import CMPrecond
        net = load_cm_model(model_path)
        net = CMPrecond(net).to(device)
        model_source = 'cm'
    else:
        if guidance_type == 'cg':            # clssifier guidance           # models from ADM
            assert classifier_path is not None
            from models.guided_diffusion.cg_model_loader import load_cg_model
            from models.networks_edm import CGPrecond
            net, classifier = load_cg_model(model_path, classifier_path)
            net = CGPrecond(net, classifier, guidance_rate=guidance_rate).to(device)
            model_source = 'adm'
        elif guidance_type in ['uncond', 'cfg']:                            # models from LDM
            from omegaconf import OmegaConf
            from models.networks_edm import CFGPrecond
            if dataset_name in ['lsun_bedroom_ldm']:
                config = OmegaConf.load('./models/ldm/configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
            elif dataset_name in ['ffhq_ldm']:
                config = OmegaConf.load('./models/ldm/configs/latent-diffusion/ffhq-ldm-vq-4.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
            elif dataset_name in ['ms_coco']:
                assert guidance_type == 'cfg'
                config = OmegaConf.load('./models/ldm/configs/stable-diffusion/v1-inference.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=4, guidance_rate=guidance_rate, guidance_type='classifier-free', label_dim=True).to(device)
            model_source = 'ldm'
    if net is None:
        raise ValueError("Got wrong settings: check dataset_name and guidance_type!")
    net.eval()

    return net, model_source

#----------------------------------------------------------------------------

@click.command()
# General options
@click.option('--dataset_name',            help='Name of the dataset', metavar='STR',                               type=str, required=True)
@click.option('--model_path',              help='Network filepath', metavar='PATH|URL',                             type=str)
@click.option('--prompt',                  help='Prompt for Stable Diffusion sampling', metavar='STR',              type=str)

# Options for sampling
@click.option('--num_steps',               help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=6, show_default=True)
@click.option('--afs',                     help='Whether to use AFS', metavar='BOOL',                               type=bool, default=False, show_default=True)
@click.option('--guidance_type',           help='Guidance type',                                                    type=click.Choice(['cg', 'cfg', 'uncond', None]), default=None, show_default=True)
@click.option('--guidance_rate',           help='Guidance rate',                                                    type=float)
@click.option('--return_inters',           help='Whether to save intermediate outputs', metavar='BOOL',             type=bool, default=False)
@click.option('--use_fp16',                help='Whether to use mixed precision', metavar='BOOL',                   type=bool, default=False)
# Additional options for multi-step solvers, 1<=max_order<=4 for iPNDM, iPNDM_v and DEIS, 1<=max_order<=3 for DPM-Solver++ and UniPC
@click.option('--max_order',               help='Max order for solvers', metavar='INT',                             type=click.IntRange(min=1))
# Additional options for DPM-Solver++ and UniPC
@click.option('--predict_x0',              help='Whether to use data prediction mode', metavar='BOOL',              type=bool, default=True)
@click.option('--lower_order_final',       help='Whether to lower the order at final stages', metavar='BOOL',       type=bool, default=True)
# Additional options for UniPC
@click.option('--variant',                 help='Type of UniPC solver', metavar='STR',                              type=click.Choice(['bh1', 'bh2']), default='bh2')
# Additional options for DEIS
@click.option('--deis_mode',               help='Type of DEIS solver', metavar='STR',                               type=click.Choice(['tab', 'rhoab']), default='tab')

# Options for scheduling
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True), default=0.002)
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=80.)
@click.option('--schedule_type',           help='Time discretization schedule', metavar='STR',                      type=click.Choice(['polynomial', 'logsnr', 'time_uniform', 'discrete']), default='polynomial', show_default=True)
@click.option('--schedule_rho',            help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--t_steps',                 help='Pre-specified time schedule', metavar='STR',                       type=str, default=None)

# PAS needed
@click.option('--plug_solver', 'plug_solver',help='Plug solver', metavar='euler|ipndm...',                           type=click.Choice(['euler', 'ipndm']), required=True)
@click.option('--epochs', 'epochs',        help='ASP search rounds', metavar='INT',                                  type=int, default=100, required=True)
@click.option('--early_stop', 'early_stop',help='ASP search early stop rounds', metavar='INT',                       type=int, default=5, required=True)
@click.option('--learning_rate', 'learning_rate',  help='ASP search learning_rate', metavar='FLOAT',                 type=float, default=0.01, required=True)
@click.option('--PCA_num', 'PCA_num',      help='Number of PCA principal components used', metavar='INT',            type=int, default=3, required=True)
@click.option('--first_time_tau', 'first_time_tau',    help='Tolerance for the first start time', metavar='FLOAT',   type=float, default=0.001, required=True)
@click.option('--others_time_tau', 'others_time_tau',  help='Tolerance for other search times', metavar='FLOAT',     type=float, default=0.0001, required=True)
@click.option('--para_name', 'para_name',  help='Current method and parameter name', metavar='STR',                  type=str, required=True)
@click.option('--use_xT', 'use_xT',        help='Using xT sampling', metavar='yes|no',                               type=str, required=True)
@click.option('--loss_type', 'loss_type',  help='Loss type', metavar='L2|L1|LPIPS',                                  type=str, required=True)
@click.option('--tea_traj_dir', 'tea_traj_dir',  help='tea traj dir path', metavar='STR',                            type=str, required=True)


def main(dataset_name, plug_solver, epochs, early_stop, learning_rate, PCA_num, first_time_tau, others_time_tau, para_name, use_xT, loss_type, tea_traj_dir, device=torch.device('cuda'), **solver_kwargs):

    dist.init()

    if dataset_name in ['ms_coco'] and solver_kwargs['prompt'] is None:
        # Loading MS-COCO captions for FID-10k evaluaion
        # We use the selected 10k captions from https://github.com/boomb0om/text2image-benchmark
        prompt_path, _ = check_file_by_key('prompts')
        sample_captions = []
        with open(prompt_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['text']
                sample_captions.append(text)

    # Rank 0 goes first
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load pre-trained diffusion models.
    net, solver_kwargs['model_source'] = create_model(dataset_name, solver_kwargs['guidance_type'], solver_kwargs['guidance_rate'], device)
    # TODO: support mixed precision 
    # net.use_fp16 = solver_kwargs['use_fp16']

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    # Get: tea_num_steps
    num_steps = solver_kwargs['num_steps']

    
    # Plug solver, 2 solvers are provided
    if plug_solver == 'euler':
        plug_solver_fn = plug_solvers.euler_step
    elif plug_solver == 'ipndm':
        plug_solver_fn = plug_solvers.ipndm_step
    else:
        raise ValueError(f'Unknown method: {plug_solver}')


    # Get the ground truth trajectory of each batch
    with open(f'{tea_traj_dir}/xt_list_gt_batch.pkl', 'rb') as f:
        xt_list_gt_batch = pickle.load(f)
    
    # Get the first diffusion step of each batch, used to adjust the direction of the basis vector
    with open(f'{tea_traj_dir}/d_first_list.pkl', 'rb') as f:
        d_first_list = pickle.load(f)
    
    # Get the class labels of each batch
    with open(f'{tea_traj_dir}/class_labels_list.pkl', 'rb') as f:
        class_labels_list = pickle.load(f)
    
    # Get the conditional labels of each batch
    with open(f'{tea_traj_dir}/cond_list.pkl', 'rb') as f:
        cond_list = pickle.load(f)
    
    # Get the unconditional conditional labels of each batch
    with open(f'{tea_traj_dir}/uncond_cond_list.pkl', 'rb') as f:
        uncond_cond_list = pickle.load(f)
    
    # Get the stu_t_steps_list
    with open(f'{tea_traj_dir}/stu_t_steps_list.pkl', 'rb') as f:
        stu_t_steps_list = pickle.load(f)
    
    device_ = d_first_list[0].device

    # Move all data to the CPU to prevent out of GPU memory
    # If you have enough GPU memory, you can run it on the GPU, which can shorten the search time slightly.
    for index_i in range(len(d_first_list)):
        d_first_list[index_i] = d_first_list[index_i].to('cpu')
    
    for index_i in range(len(xt_list_gt_batch)):
        for index_j in range(len(xt_list_gt_batch[0])):
            xt_list_gt_batch[index_i][index_j] = xt_list_gt_batch[index_i][index_j].to('cpu')

    dist.print0(f"d_r: {d_first_list[0].shape[2]}")
    dist.print0(f"xt_r: {xt_list_gt_batch[0][0].shape[2]}")
    # Initialize some variables for the search
    stu_t_step = torch.stack(stu_t_steps_list)  # Get stu t_steps
    # batch, NFE, [batch_size, c, r, r] -> NFE, batch, [batch_size, c, r, r]
    xt_list_gt_batch = [list(batch) for batch in zip(*xt_list_gt_batch)]

    x0 = xt_list_gt_batch[0]  # Get x0, used to adjust the direction of the basis vector, shape: batch, [batch_size, c, r, r]

    buffer_d = []
    if use_xT == "yes":  # xT and d are in the same space
        buffer_d.append(x0)  # batch, [batch_size, c, r, r]

    cur_xt_searched = x0  # Get the current searched xt, xT no search required
    xt_list_gt_batch.pop(0)  # Remove the used states from the trajectory to prevent them from occupying cpu memory

    cur_d = d_first_list  # The d corresponding to the first xt_searched does not need to be recalculated, shape: batch, [batch_size, c, r, r]

    next_xt_gt = xt_list_gt_batch[0]  # The label of the current time point
    xt_list_gt_batch.pop(0)  # Remove the used states from the trajectory to prevent them from occupying cpu memory
    
    coordinates_dict = {}  # The coordinate dictionary finally searched
    init_coordinates_dict = {}  # The coordinate dictionary initially
    loss_origin_dict = {}
    loss_dict = {}
    flag_first_search = 1  # Flag to determine whether it is the first calibration

    # Initialize the loss function for the search
    if loss_type == "L2":
        criterion = nn.MSELoss(reduction='mean')
    elif loss_type == "L1":
        criterion = nn.L1Loss(reduction='mean')
    elif loss_type == "LPIPS":
        lpips_model = lpips.LPIPS(net='alex').to(device_)
        criterion = lambda input, target: lpips_model(input, target).mean()
    elif loss_type == "Pseudo_Huber":  # Proposed at https://arxiv.org/abs/2310.14189
        class Pseudo_Huber(nn.Module):
            def __init__(self, c):
                super(Pseudo_Huber, self).__init__()
                self.c = c
            def forward(self, x, y):
                x = x.view(x.size(0), -1)
                y = y.view(y.size(0), -1)
                loss = torch.sqrt(torch.mean((x - y) ** 2, dim=1) + self.c ** 2) - self.c
                return loss.mean()
        criterion = Pseudo_Huber(0.03)  # cifar10 0.03 https://arxiv.org/abs/2310.14189
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    if plug_solver == "ipndm":
        max_order = solver_kwargs['max_order']
    
    buffer_ipndm = [[d_first] for d_first in d_first_list]  # Initialize the buffer_ipndm, used to save the d of the previous time point (only used for ipndm), batch, [batch_size, c, r, r]

    time_s = time.time()
    dist.print0("\nStart search...")
    for time_point in range(num_steps-1):
        flag_need_search = 1  # Flag to determine whether the current time point needs to be searched
        
        if time_point == 0 and use_xT == "no" and PCA_num == 3:
            dist.print0(f"\nTime point {time_point} cannot be corrected. Because PCA_num: {PCA_num} and use_afs: {use_xT}.")
            flag_need_search = 0  # The current time point cannot be corrected
        elif time_point == 0 and PCA_num == 4:
            dist.print0(f"\nTime point {time_point} cannot be corrected. Because PCA_num: {PCA_num}.")
            flag_need_search = 0

        if time_point == 1 and use_xT == "no" and PCA_num == 4:
            dist.print0(f"\nTime point {time_point} cannot be corrected. Because PCA_num: {PCA_num} and use_afs: {use_xT}.")
            flag_need_search = 0

        if flag_need_search == 1:  # Further determine whether the current point needs correction based on the loss
            dist.print0(f"\nStart search time point: {time_point}...")
            model = CoordinateSearcher(cur_d, PCA_num).to(device_)  # Initialize the first coordinate with the L2 norm of the cur_d
            init_coordinates = model.coordinates.data.clone().cpu()  # [3,]
            dist.print0(f"Initial coordinates: {init_coordinates}")

            # Initialize the optimizer for the search
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Search coordinates
            cur_d_searched, next_xt_searched, (loss_origin, loss) = do_search(dist, model, cur_xt_searched, cur_d, next_xt_gt, buffer_d, x0, stu_t_step, epochs, early_stop, PCA_num, time_point, optimizer, criterion, device_)
            coordinates = model.coordinates.detach().cpu()  # [3,]
            dist.print0(f"Searched coordinates: {coordinates}")

            # Used to further determine whether the current time point needs to be corrected
            dist.print0(f"Loss_origin: {loss_origin}, Loss: {loss}")

            if flag_first_search == 1:
                tau_result = loss_origin - (loss + first_time_tau)  # first_time_tau: the tolerance for the first searched time point, slightly larger
                if tau_result > 0:  
                    flag_first_search = 0
                    dist.print0(f"Time point: {time_point} was added for correction, as the first searched time point. Tau_result: {tau_result}; first_time_tau: {first_time_tau}.")
                else:
                    flag_need_search = 0
                    dist.print0(f"Time point: {time_point} no need to corrected. Tau_result: {tau_result}; first_time_tau: {first_time_tau}.")
            else:
                tau_result = loss_origin - (loss + others_time_tau)  # others_time_tau: the tolerance for the other searched time points
                if tau_result > 0:
                    dist.print0(f"Time point: {time_point} was added for correction. Tau_result: {tau_result}; others_time_tau: {others_time_tau}.")
                else:
                    flag_need_search = 0
                    dist.print0(f"Time point: {time_point} no need to corrected. Tau_result: {tau_result}; others_time_tau: {others_time_tau}.")
            
            del model, optimizer

        if flag_need_search == 1:  # If the current time point needs to be corrected
            coordinates_dict[time_point] = coordinates
            init_coordinates_dict[time_point] = init_coordinates
            loss_origin_dict[time_point] = loss_origin
            loss_dict[time_point] = loss
        
        if time_point == num_steps - 2:  # All time points have been searched and the status will not be updated any more.
            break

        next_xt_gt = xt_list_gt_batch[0]
        xt_list_gt_batch.pop(0)  # Remove the used states from the trajectory to prevent them from occupying cpu memory

        if flag_need_search == 1:  # If the current time point needs to be corrected
            buffer_d.append(cur_d_searched)  # Save the searched d for the next time point (Represents the new sampling trajectory)
            cur_xt_searched = next_xt_searched

            cur_d_temp = []  # update the status of next time point cur_d
            with torch.no_grad():
                if solver_kwargs['model_source'] == 'ldm':
                    with autocast("cuda"):
                        with net.model.ema_scope():
                            for cur_xt_searched_batch, class_labels_batch, cond_batch, uncond_cond_batch, buffer_ipndm_batch in zip(cur_xt_searched, class_labels_list, cond_list, uncond_cond_list, buffer_ipndm):
                                if plug_solver == "euler":
                                    cur_d_batch = plug_solver_fn(device_, cur_xt_searched_batch, stu_t_step[time_point+1], solver_kwargs['model_source'], net, class_labels_batch, cond_batch, uncond_cond_batch)
                                elif plug_solver == "ipndm":
                                    cur_d_batch = plug_solver_fn(device_, cur_xt_searched_batch, stu_t_step[time_point+1], solver_kwargs['model_source'], net, time_point, buffer_ipndm_batch, max_order, class_labels_batch, cond_batch, uncond_cond_batch)
                                else:
                                    raise ValueError(f'Unknown method: {plug_solver}')
                                cur_d_temp.append(cur_d_batch)
                else:
                    for cur_xt_searched_batch, class_labels_batch, buffer_ipndm_batch in zip(cur_xt_searched, class_labels_list, buffer_ipndm):
                        if plug_solver == "euler":
                            cur_d_batch = plug_solver_fn(device_, cur_xt_searched_batch, stu_t_step[time_point+1], solver_kwargs['model_source'], net, class_labels_batch)
                        elif plug_solver == "ipndm":
                            cur_d_batch = plug_solver_fn(device_, cur_xt_searched_batch, stu_t_step[time_point+1], solver_kwargs['model_source'], net, time_point, buffer_ipndm_batch, max_order, class_labels_batch)
                        else:
                            raise ValueError(f'Unknown method: {plug_solver}')
                        cur_d_temp.append(cur_d_batch)
            cur_d = cur_d_temp
        else:  # If the current time point does not need to be corrected
            buffer_d.append(cur_d)  # Save cur_d before correction (which means retaining the original sampling trajectory)

            cur_xt_searched_temp = []
            cur_d_temp = []
            with torch.no_grad():
                cur_stu_t_step = stu_t_step[time_point].to('cpu')
                next_stu_t_step = stu_t_step[time_point+1].to('cpu')
                if solver_kwargs['model_source'] == 'ldm':
                    with autocast("cuda"):
                        with net.model.ema_scope():
                            for cur_xt_searched_batch, cur_d_batch, class_labels_batch, cond_batch, uncond_cond_batch, buffer_ipndm_batch in zip(cur_xt_searched, cur_d, class_labels_list, cond_list, uncond_cond_list, buffer_ipndm):
                                # Euler step.
                                next_xt_batch = cur_xt_searched_batch + cur_d_batch * (next_stu_t_step - cur_stu_t_step)
                                cur_xt_searched_temp.append(next_xt_batch)
                                if plug_solver == "euler":
                                    cur_d_batch = plug_solver_fn(device_, next_xt_batch, stu_t_step[time_point+1], solver_kwargs['model_source'], net, class_labels_batch, cond_batch, uncond_cond_batch)
                                elif plug_solver == "ipndm":
                                    cur_d_batch = plug_solver_fn(device_, next_xt_batch, stu_t_step[time_point+1], solver_kwargs['model_source'], net, time_point, buffer_ipndm_batch, max_order, class_labels_batch, cond_batch, uncond_cond_batch)
                                else:
                                    raise ValueError(f'Unknown method: {plug_solver}')
                                cur_d_temp.append(cur_d_batch)
                else:
                    for cur_xt_searched_batch, cur_d_batch, class_labels_batch, buffer_ipndm_batch in zip(cur_xt_searched, cur_d, class_labels_list, buffer_ipndm):
                        # Euler step.
                        next_xt_batch = cur_xt_searched_batch + cur_d_batch * (next_stu_t_step - cur_stu_t_step)
                        cur_xt_searched_temp.append(next_xt_batch)
                        if plug_solver == "euler":
                            cur_d_batch = plug_solver_fn(device_, next_xt_batch, stu_t_step[time_point+1], solver_kwargs['model_source'], net, class_labels_batch)
                        elif plug_solver == "ipndm":
                            cur_d_batch = plug_solver_fn(device_, next_xt_batch, stu_t_step[time_point+1], solver_kwargs['model_source'], net, time_point, buffer_ipndm_batch, max_order, class_labels_batch)
                        else:
                            raise ValueError(f'Unknown method: {plug_solver}')
                        cur_d_temp.append(cur_d_batch)
            cur_xt_searched = cur_xt_searched_temp
            cur_d = cur_d_temp
    
    time_e = time.time()
    dist.print0(f"\nSearch time: {time_e - time_s:.2f}s")

    if not coordinates_dict:
        dist.print0("\nNo time points need to be corrected.")
    else:
        result_string = ','.join(map(str, sorted(coordinates_dict.keys())))  # Get the time points that need to be corrected
                
        dist.print0(f"\ninit coordinates dict:")
        for key, value in sorted(init_coordinates_dict.items()):
            dist.print0(f"{key}: {value}")

        dist.print0(f"\nsearched coordinates dict:")
        for key, value in sorted(coordinates_dict.items()):
            dist.print0(f"{key}: {value}. loss_origin: {loss_origin_dict[key]}, loss: {loss_dict[key]}")
        
        dist.print0(f"\nall time points need to be corrected: {result_string}")
    
    save_dir = "./out/coordinates/" 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    path = save_dir + f"{para_name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(coordinates_dict, f)


    # Done.
    torch.distributed.barrier()
    dist.print0('search done.\n')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
