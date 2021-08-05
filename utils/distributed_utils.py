import os
import torch
import random
import multiprocessing
import numpy as np
import torch.distributed as t_dist


def dist_init(port=2333):
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    rank = int(os.environ['SLURM_PROCID'])
    world_size = os.environ['SLURM_NTASKS']
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)
    
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    
    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = world_size
    os.environ['RANK'] = str(rank)
    
    t_dist.init_process_group(backend='nccl')
    
    return rank, int(world_size), gpu_id


def init_device(args):
    # Random seed setting
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            print('Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = int(os.environ['LOCAL_RANK'])

    print(f"device: {args.device}")
    torch.cuda.set_device(args.device)
    t_dist.init_process_group(backend='nccl', init_method='env://')

    rank = t_dist.get_rank()
    size = t_dist.get_world_size()
    device = torch.device(f'cuda:{args.device}')
    print(f"args.num_gpu: {args.num_gpu}, rank: {rank}, world_size: {size}")

    print(f'=> Device: running distributed training with world size:{size}')
    # PyTorch version record
    print(f'=> PyTorch Version: {torch.__version__}\n')
    t_dist.barrier()
    return rank, size, device