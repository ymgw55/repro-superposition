import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import einops
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Run pretraining LM.")
    parser.add_argument(
        "--n_features",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--pack",
        type=int,
        default=256,
        help="Number of models trained simultaneously on each GPU",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=16,
        help="Number of random repeats for each (i, j)",
    )

    args = parser.parse_args()
    return args


args = get_args()
n_features = args.n_features
steps = args.steps
pack = args.pack
repeats = args.repeats


ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    # use of DDP atm demands CUDA,
    # we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"

    # 1) First, read rank information
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    # 2) Fix the device first (0..N-1 corresponding to CUDA_VISIBLE_DEVICES order)
    torch.cuda.set_device(ddp_local_rank)
    DEVICE = f"cuda:{ddp_local_rank}"

    # 3) Then initialize the process group (explicitly specify device_id)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=ddp_rank,
        world_size=ddp_world_size,
        device_id=ddp_local_rank,
    )
    master_process = ddp_rank == 0

else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

device_type = "cuda" if DEVICE.startswith("cuda") else "cpu"


@dataclass
class Config:
    n_features: int
    n_hidden: int

    # We optimize n_instances models in a single training loop
    # to let us sweep over sparsity or importance curves
    # efficiently.

    # We could potentially use torch.vmap instead.
    n_instances: int


class Model(nn.Module):
    def __init__(
        self,
        config,
        feature_probability: Optional[torch.Tensor] = None,
        importance: Optional[torch.Tensor] = None,
        device="cuda",
    ):
        super().__init__()
        self.config = config
        self.W = nn.Parameter(
            torch.empty(
                (config.n_instances, config.n_features, config.n_hidden), device=device
            )
        )
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(
            torch.zeros((config.n_instances, config.n_features), device=device)
        )

        if feature_probability is None:
            feature_probability = torch.ones(())
        self.feature_probability = feature_probability.to(device)
        if importance is None:
            importance = torch.ones(())
        self.importance = importance.to(device)

    def forward(self, features):
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = torch.einsum("...if,ifh->...ih", features, self.W)
        out = torch.einsum("...ih,ifh->...if", hidden, self.W)
        out = out + self.b_final
        out = F.relu(out)
        return out

    def generate_batch(self, n_batch):
        feat = torch.rand(
            (n_batch, self.config.n_instances, self.config.n_features),
            device=self.W.device,
        )
        batch = torch.where(
            torch.rand(
                (n_batch, self.config.n_instances, self.config.n_features),
                device=self.W.device,
            )
            <= self.feature_probability,
            feat,
            torch.zeros((), device=self.W.device),
        )
        return batch


def linear_lr(step, steps):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


def optimize(
    model,
    render=False,
    n_batch=1024,
    steps=10_000,
    lr=1e-3,
    lr_scale=cosine_decay_lr,
    hooks=[],
):
    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

    for step in range(steps):
        step_lr = lr * lr_scale(step, steps)
        for group in opt.param_groups:
            group["lr"] = step_lr
        opt.zero_grad(set_to_none=True)
        batch = model.generate_batch(n_batch)
        out = model(batch)
        error = model.importance * (batch.abs() - out) ** 2
        loss = einops.reduce(error, "b i f -> i", "mean").sum()
        loss.backward()
        opt.step()

        if hooks:
            hook_data = dict(
                model=model, step=step, opt=opt, error=error, loss=loss, lr=step_lr
            )
            for h in hooks:
                h(hook_data)


log_weight_start = -1
log_weight_end = 1
log_sparsity_start = 0
log_sparsity_end = -2


def interpolate(a, b, n, steps):
    return a + (b - a) * n / (steps - 1)


log_sparsities = np.zeros((steps, steps))
log_weights = np.zeros((steps, steps))
data = []
if master_process:
    iterator = tqdm(
        list(range(steps)),
        desc=f"Running {n_features} features, {steps} steps",
        dynamic_ncols=True,
    )
else:
    iterator = range(steps)

for i in iterator:
    if i % ddp_world_size != ddp_rank:
        continue

    log_sparsity = interpolate(log_sparsity_start, log_sparsity_end, i, steps)
    sparsity = 10**log_sparsity

    j0 = 0
    while j0 < steps:
        this_pack = min(pack, steps - j0)
        log_weights_tile = [
            interpolate(log_weight_start, log_weight_end, (j0 + p), steps)
            for p in range(this_pack)
        ]
        # shape: (pack,)
        rel_weights_tile = torch.tensor(
            [10**lw for lw in log_weights_tile], device=DEVICE
        )

        # importance
        imp = torch.ones(
            (this_pack, n_features), device=DEVICE
        )  # All ones at first, [P, F]
        imp[:, -1] = rel_weights_tile  # Set the weight for the last feature
        imp = imp.repeat_interleave(repeats, dim=0)  # [P * R, F]

        # feature probability
        fp = torch.full(
            (this_pack * repeats, n_features), sparsity, device=DEVICE
        )  # [P * R, F]

        config = Config(
            n_features=n_features,
            n_hidden=n_features - 1,
            n_instances=this_pack * repeats,
        )
        model = Model(
            config=config, device=DEVICE, importance=imp, feature_probability=fp
        )
        optimize(model)

        W = model.W.detach().reshape(
            this_pack, repeats, n_features, config.n_hidden
        )  # [P, R, F, H]
        W_norm = W / (
            1e-5 + torch.linalg.norm(W, 2, dim=-1, keepdim=True)
        )  # Same shape

        # Interference matrix (each instance independent): [P, R, F, F]
        interference = torch.einsum("prfh,prgh->prfg", W_norm, W)
        f_idx = torch.arange(n_features, device=W.device)
        interference[:, :, f_idx, f_idx] = 0

        # Take the mean of polysemanticity and norm
        # Then extract the "last feature"
        poly = torch.linalg.norm(interference, dim=-1)  # [P, R, F]
        last_poly = poly.mean(dim=1)[..., -1]  # [P]
        norms = torch.linalg.norm(W, 2, dim=-1)  # [P, R, F]
        last_norm = norms.mean(dim=1)[..., -1]  # [P]

        # Append to data for each (i, j)
        for p in range(this_pack):
            j = j0 + p
            data.append(
                dict(
                    log_sparsity_index=i,
                    log_weight_index=j,
                    log_sparsity=log_sparsity,
                    log_weight=log_weights_tile[p],
                    last_polysemanticity=last_poly[p].item(),
                    last_norm=last_norm[p].item(),
                )
            )
        j0 += this_pack

if device_type == "cuda":
    # wait for the GPU to finish work
    torch.cuda.synchronize()

# ===== Aggregation =====
if ddp:
    dist.barrier(device_ids=[ddp_local_rank])
    if master_process:
        gathered = [None] * ddp_world_size
        dist.gather_object(data, gathered, dst=0)
        data = [row for part in gathered for row in part]
    else:
        dist.gather_object(data, None, dst=0)

# ===== Save (master only) =====
if master_process:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{n_features}features-{steps}steps.csv"
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

# ===== DDP cleanup =====
if ddp:
    dist.destroy_process_group()
