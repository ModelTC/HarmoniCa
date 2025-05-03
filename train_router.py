# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models.router_models import DiT_models, STE
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
import random

# from torch.nn.functional import sigmoid


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def get_lambda(model, z, seed, T, device, diffusion, model_kwargs):
    model.lambda_gen = True
    dynamic_lambda = dict()
    model_kwargs["fix_reuse_feature"] = False
    model_kwargs["activate_router"] = True
    model.mask = [torch.tensor([1.0] * model.depth * 2) for _ in range(T)]
    model.reset()
    seed_everything(seed)
    pre = diffusion.ddim_sample_loop(
        model.forward,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    for i in range(T - 2, -1, -1):
        model.mask[i] = model.routers[i]()
        model.reset()
        seed_everything(seed)
        cur = diffusion.ddim_sample_loop(
            model.forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        loss = mean_flat((cur - pre) ** 2).mean()
        dynamic_lambda[i] = loss
        pre = cur
    del model.lambda_gen, model.mask
    return dynamic_lambda


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def format_image_to_wandb(num_router, router_size, router_scores):
    image = np.zeros((num_router, router_size, 3), dtype=np.float32)
    ones = np.ones((3), dtype=np.float32)
    for idx, score in enumerate(router_scores):
        mask = score.cpu().detach()
        for pos in range(router_size):
            image[idx, pos] = ones * mask[pos].item()
    return image


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert (
        args.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(
            args.results_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace(
            "/", "-"
        )  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{args.l1}-{args.lambda_c}"  # Create an experiment folder
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert (
        args.image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size, num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    msg = model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        logger.info(f"Loaded model from {ckpt_path} with msg: {msg}")
    model.eval()  # important!

    diffusion = create_diffusion(str(args.num_sampling_steps))
    model.add_router(args.num_sampling_steps, diffusion.timestep_map)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opts = torch.optim.AdamW(
        [param for name, param in model.named_parameters() if "routers" in name],
        lr=args.lr,
        weight_decay=0,
    )

    if args.wandb and rank == 0:
        import wandb

        wandb.init(
            # Set the project where this run will be logged
            project="DiT-Router-Dist",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{experiment_index:03d}-{model_string_name}-dynamic-lambda-{args.l1}-{args.lambda_c}",
            # Track hyperparameters and run metadata
            config=args.__dict__,
        )
        wandb.define_metric("step")
        wandb.define_metric("loss", step_metric="step")

    model.eval()

    logger.info(f"Training for {args.epochs} epochs...")

    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_data_loss, running_l1_loss = 0, 0
    start_time = time()
    assert (
        args.log_every % args.num_sampling_steps == 0
    ), "Log every must be divisible by num_sampling_steps"
    dynamic_lambda = None
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        bs = args.global_batch_size // dist.get_world_size()
        y = torch.randint(low=0, high=1000, size=(bs // 2,), device=device)
        y_null = torch.tensor([1000] * (bs // 2), device=device)
        y = torch.cat([y, y_null], 0)
        x_t = torch.randn(bs // 2, 4, latent_size, latent_size, device=device)
        x_t = torch.cat([x_t, x_t], 0)
        model_kwargs = dict(
            y=y,
            cfg_scale=args.cfg_scale,
            thres=args.ste_threshold,
            label_for_dropout=None,
        )
        assert args.epochs * args.num_sampling_steps % args.lambda_c == 0
        if (epoch * args.num_sampling_steps) % args.lambda_c == 0:
            dynamic_lambda = get_lambda(
                model.module,
                x_t,
                seed,
                args.num_sampling_steps,
                device,
                diffusion,
                model_kwargs,
            )
            if rank == 0:
                logger.info(f"Dynamic Lambda: {dynamic_lambda}")
        indices = list(range(diffusion.num_timesteps))[::-1]
        for i in indices:
            ori_t = torch.tensor([i] * bs, device=device)
            if i == indices[0]:
                with torch.no_grad():
                    x_t = diffusion.ddim_sample(
                        model,
                        x_t,
                        ori_t,
                        clip_denoised=False,
                        denoised_fn=None,
                        cond_fn=None,
                        model_kwargs=model_kwargs,
                        eta=0.0,
                    )["sample"]
                log_steps += 1
                train_steps += 1
                continue

            map_tensor = torch.tensor(
                diffusion.timestep_map, device=ori_t.device, dtype=ori_t.dtype
            )
            t = map_tensor[ori_t]

            model_kwargs["fix_reuse_feature"] = True
            model_kwargs["activate_router"] = False
            ori_model_output = model(x_t, t, **model_kwargs)

            model_kwargs["fix_reuse_feature"] = False
            model_kwargs["activate_router"] = True
            pred_model_output, l1_loss = model(x_t, t, **model_kwargs)
            data_loss = mean_flat((ori_model_output - pred_model_output) ** 2).mean()
            l1_loss = l1_loss.mean()
            loss = data_loss * dynamic_lambda[i] + args.l1 * l1_loss
            opts.zero_grad()
            model.zero_grad()
            loss.backward()
            opts.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "routers" in name:
                        param.clamp_(-5, 5)

            running_loss += loss.item()
            running_data_loss += data_loss.item()
            running_l1_loss += args.l1 * l1_loss.item()

            log_steps += 1
            train_steps += 1
            out = diffusion.distillation_post_process(
                pred_model_output,
                x_t,
                ori_t,
                clip_denoised=False,
                denoised_fn=None,
                cond_fn=None,
                eta=0.0,
            )
            x_t = out["sample"].detach()

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                for name, loss in [
                    ("loss", running_loss),
                    ("data_loss", running_data_loss),
                    ("l1_loss", running_l1_loss),
                ]:
                    loss = torch.tensor(loss / log_steps, device=device)
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss.item() / dist.get_world_size()
                    logger.info(
                        f"(step={train_steps:07d}, epoch={epoch:07d}) Train {name} Loss: {loss:.7f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )

                scores = [
                    model.module.routers[idx]()
                    for idx in range(0, args.num_sampling_steps)
                ]

                if args.wandb and rank == 0:
                    # print(scores)
                    mask = format_image_to_wandb(
                        args.num_sampling_steps, model.module.depth * 2, scores
                    )
                    mask = wandb.Image(
                        mask,
                    )
                    final_score = [sum(score) for score in scores]
                    wandb.log(
                        {
                            "step": train_steps,
                            "loss": running_loss / log_steps,
                            "data_loss": running_data_loss / log_steps,
                            "l1_loss": running_l1_loss / log_steps,
                            "non_zero": sum(final_score),
                            "router": mask,
                        }
                    )

                running_loss = 0
                running_data_loss, running_l1_loss = 0, 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "routers": model.module.routers.state_dict(),
                        "opt": opts.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps > args.max_steps:
                print("Reach Maximum Step")
                break

    model.eval()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse"], default="ema"
    )  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--wandb", action="store_true")

    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--l1", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--ste-threshold", type=float, default=None)
    parser.add_argument("--lambda-c", type=int, default=1000)

    args = parser.parse_args()
    main(args)
