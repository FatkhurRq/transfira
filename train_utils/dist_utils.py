import os
import torch.distributed as dist
import torch


def init_distributed(dist_url="env://"):
    """
    Initializes the distributed backend that will take care
    of synchronizing nodes/GPUs

    This only works with torch.distributed.launch or torch.run
    """

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend="nccl", init_method=dist_url, world_size=world_size, rank=rank
    )

    # This will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # Synchronize all threads to reach this point
    dist.barrier()

    return


def is_dist_avail_and_initialized():
    if not (dist.is_available()):
        return False
    if not (dist.is_initialized()):
        return False
    return True


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
    return


def get_rank():
    if not (is_dist_avail_and_initialized()):
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0
