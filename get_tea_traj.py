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
import solver_utils
from torch import autocast
from torch_utils import distributed as dist
from torchvision.utils import make_grid, save_image
from torch_utils.download_util import check_file_by_key
import time

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
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--prompt',                  help='Prompt for Stable Diffusion sampling', metavar='STR',              type=str)

# Options for sampling
@click.option('--tea_solver',              help='Tea solver', metavar='euler|heun|dpm_solver_2',                    type=click.Choice(['euler', 'heun', 'dpm']))
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
@click.option('--tea_NFE', 'tea_NFE',      help='Approximate NFE for tea', metavar='INT',                            type=int, default=100, required=True)
@click.option('--plug_solver', 'plug_solver',help='Plug solver', metavar='euler|ipndm|deis',                         type=click.Choice(['euler', 'ipndm', 'deis']), required=True)
@click.option('--tea_traj_dir', 'tea_traj_dir',  help='tea traj dir path', metavar='STR',                            type=str, required=True)


def main(dataset_name, max_batch_size, seeds, t_steps, tea_NFE, plug_solver, tea_traj_dir, device=torch.device('cuda'), **solver_kwargs):
    dist.init()
    if os.path.exists(tea_traj_dir):
        dist.print0(f"tea_traj already exists: {tea_traj_dir}.")
        # Done.
        torch.distributed.barrier()
        dist.print0('tea_traj done.\n')
        exit()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

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
    
    # Get: tea_num_steps, M, tea_NFE_gt, stu_in_tea_index
    tea_solver = solver_kwargs['tea_solver']
    num_steps = solver_kwargs['num_steps']
    schedule_type = solver_kwargs['schedule_type']
    if schedule_type == 'polynomial':
        if plug_solver == "euler" or plug_solver == "ipndm" or plug_solver == "deis":
            if tea_solver == "euler":
                for j in range(1, tea_NFE):
                    tea_num_steps = (num_steps - 1) * (j + 1) + 1
                    tea_NFE_temp = tea_num_steps - 1
                    if tea_NFE_temp >= tea_NFE:
                        M = j
                        tea_NFE_gt = tea_NFE_temp  # the NFE of the tea_solver real use
                        break
                stu_in_tea_index = [i*(M+1) for i in range(num_steps)]
            elif tea_solver == "heun" or tea_solver == "dpm":
                for j in range(0, tea_NFE):
                    tea_num_steps = (num_steps - 1) * (j + 1) + 1
                    tea_NFE_temp = 2 * (tea_num_steps - 1)
                    if tea_NFE_temp >= tea_NFE:
                        M = j
                        tea_NFE_gt = tea_NFE_temp  # the NFE of the tea_solver real use
                        break
                stu_in_tea_index = [i*(M+1) for i in range(num_steps)]
            else:
                raise ValueError(f'Unknown method: {tea_solver}')
        else:
            raise ValueError(f'Unknown method: {plug_solver}')
    else:
        raise ValueError(f'Unknown method: {schedule_type}')
    
    dist.print0("tea_NFE_gt: ", tea_NFE_gt)

    # Get the time schedule
    solver_kwargs['sigma_min'] = net.sigma_min
    solver_kwargs['sigma_max'] = net.sigma_max
    if t_steps is None:
        t_steps = solver_utils.get_schedule(tea_num_steps, solver_kwargs['sigma_min'], solver_kwargs['sigma_max'], device=device, \
                                            schedule_type=schedule_type, schedule_rho=solver_kwargs["schedule_rho"], net=net)
    solver_kwargs['t_steps'] = t_steps

    
    # Generating ground truth trajectories, 3 solvers are provided
    if tea_solver == 'euler':
        sampler_fn = tea_solvers.euler_sampler
    elif tea_solver == 'heun':
        sampler_fn = tea_solvers.heun_sampler
    elif tea_solver == 'dpm':
        sampler_fn = tea_solvers.dpm_2_sampler  
    else:
        raise ValueError(f'Unknown method: {tea_solver}')
    

    xt_list_gt_batch, d_first_list, class_labels_list, cond_list, uncond_cond_list = [], [], [], [], []
    dist.print0(f'Generating {len(seeds)} teacher trajectory...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = c = uc = None
        if net.label_dim:
            if solver_kwargs['model_source'] == 'adm':
                class_labels = rnd.randint(net.label_dim, size=(batch_size,), device=device)
            elif solver_kwargs['model_source'] == 'ldm' and dataset_name == 'ms_coco':
                if solver_kwargs['prompt'] is None:
                    prompts = sample_captions[batch_seeds[0]:batch_seeds[-1]+1]
                else:
                    prompts = [solver_kwargs['prompt'] for i in range(batch_size)]
                if solver_kwargs['guidance_rate'] != 1.0:
                    uc = net.model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = net.model.get_learned_conditioning(prompts)
            else:
                class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]

        # Generate images.
        with torch.no_grad():
            if solver_kwargs['model_source'] == 'ldm':
                with autocast("cuda"):
                    with net.model.ema_scope():
                        xt_list, d_first, tea_t_steps = sampler_fn(net, latents, condition=c, unconditional_condition=uc, **solver_kwargs)
            else:
                xt_list, d_first, tea_t_steps = sampler_fn(net, latents, class_labels=class_labels, **solver_kwargs)

        xt_list_gt = [xt_list[i] for i in stu_in_tea_index]
        stu_t_steps_list = [tea_t_steps[i] for i in stu_in_tea_index]

        for index_i in range(len(xt_list_gt)):  # Put the traj on the CPU to prevent out of memory
            xt_list_gt[index_i] = xt_list_gt[index_i].to('cpu')

        xt_list_gt_batch.append(xt_list_gt)  # Save the ground truth trajectory of each batch
        d_first_list.append(d_first)  # Save the first diffusion step of each batch, used to adjust the direction of the basis vector
        class_labels_list.append(class_labels)  # Save the class labels of each batch
        cond_list.append(c)  # Save the conditional labels of each batch
        uncond_cond_list.append(uc)  # Save the unconditional conditional labels of each batch
    
    if not os.path.exists(tea_traj_dir):
        os.makedirs(tea_traj_dir)
        
    # Save the ground truth trajectory of each batch
    with open(f'{tea_traj_dir}/xt_list_gt_batch.pkl', 'wb') as f:
        pickle.dump(xt_list_gt_batch, f)

    # Save the first diffusion step of each batch, used to adjust the direction of the basis vector
    with open(f'{tea_traj_dir}/d_first_list.pkl', 'wb') as f:
        pickle.dump(d_first_list, f)

    # Save the class labels of each batch
    with open(f'{tea_traj_dir}/class_labels_list.pkl', 'wb') as f:
        pickle.dump(class_labels_list, f)

    # Save the conditional labels of each batch
    with open(f'{tea_traj_dir}/cond_list.pkl', 'wb') as f:
        pickle.dump(cond_list, f)

    # Save the unconditional conditional labels of each batch
    with open(f'{tea_traj_dir}/uncond_cond_list.pkl', 'wb') as f:
        pickle.dump(uncond_cond_list, f)

    # Save the stu_t_steps_list
    with open(f'{tea_traj_dir}/stu_t_steps_list.pkl', 'wb') as f:
        pickle.dump(stu_t_steps_list, f)

    # Done.
    torch.distributed.barrier()
    dist.print0('tea_traj done.\n')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
