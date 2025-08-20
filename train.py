import argparse
import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_constant_schedule_with_warmup

from config import SaeConfig, TrainConfig, UsrConfig, return_save_dir
from dataset import CustomWikiDataset
from model import SimpleHook, SparseAutoEncoder, normalize_activation

try:
    import wandb
    _WANDB = True
except:
    _WANDB = False

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        MODEL_DEVICE = torch.device("cuda:0")
        SAE_DEVICE = torch.device("cuda:1")
    else:
        MODEL_DEVICE = torch.device("cuda")
        SAE_DEVICE = torch.device("cuda")
else:
    MODEL_DEVICE = torch.device("cpu")
    SAE_DEVICE = torch.device("cpu")


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5) -> Tensor:
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        distances = torch.norm(points - guess, dim=1)
        # Avoid division by zero
        weights = 1 / (distances + 1e-8)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


def train(
    dl_train, dl_val, train_cfg, model_dir, layers, n_d, k, nl, ckpt, lr, save_dir
):
    # load the language model to extract activations
    model = AutoModelForCausalLM.from_pretrained(
        # os.path.join(model_dir, f"iter_{str(ckpt).zfill(7)}"),
        model_dir,
        torch_dtype=torch.bfloat16,
    ).to(MODEL_DEVICE)
    model.eval()

    # initialize the sparse autoencoders
    saes = {}
    optims = {}
    hooks = {}
    for layer in layers:
        # SAE
        cfg = SaeConfig(expansion_factor=n_d, k=k)
        sae = SparseAutoEncoder(cfg).to(SAE_DEVICE)
        saes[layer] = sae
        optims[layer] = torch.optim.Adam(sae.parameters(), lr=lr, eps=6.25e-10)

        # Hook
        target = model.model.embed_tokens if layer == 0 else model.model.layers[layer-1]
        hooks[layer] = SimpleHook(target)

    schedulers = {
        layer: get_constant_schedule_with_warmup(optims[layer], num_warmup_steps=train_cfg.lr_warmup_steps)
        for layer in layers
    }

    global_step = {layer: 0 for layer in layers}
    loss_sum = {layer: 0.0 for layer in layers}

    if _WANDB:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", save_dir.replace("/", "-")),
            name=f"multi_layers_nd{n_d}_k{k}_ckpt{ckpt}_lr{lr}",
            config={
                "layers": layers,
                "expansion_factor": n_d,
                "k": k,
                "nl": nl,
                "ckpt": ckpt,
                "lr": lr,
                "batch_size": train_cfg.batch_size,
                "inf_bs_expansion": train_cfg.inf_bs_expansion,
                "lr_warmup_steps": train_cfg.lr_warmup_steps,
            },
        )
    for layer in layers:
        wandb.watch(saes[layer], log="all", log_freq=train_cfg.logging_step)

    for batch in tqdm(dl_train, desc="Training"):
        with torch.inference_mode():
            _ = model(batch.to(MODEL_DEVICE), use_cache=False)
        
        for layer in layers:
            sae = saes[layer]
            optim = optims[layer]
            scheduler = schedulers[layer]
            hook = hooks[layer]
            out = hook.output
            out = out[0] if isinstance(out, tuple) else out
            activation = normalize_activation(out[:, 1:, :].flatten(0, 1), nl)
            # split the activations into chunks
            for chunk in torch.chunk(activation, train_cfg.inf_bs_expansion, dim=0):
                # Initialize decoder bias with the geometric median of the chunk
                if global_step[layer] == 0:
                    median = geometric_median(chunk.to(SAE_DEVICE))
                    sae.b_dec.data = median.to(sae.dtype)
                # make sure the decoder weights are unit norm
                sae.set_decoder_norm_to_unit_norm()
                optim.zero_grad()
                out = sae(chunk.to(SAE_DEVICE))
                loss = out.loss
                loss.backward()
                loss_sum[layer] += loss.item()
                optim.step()
                scheduler.step()

                if global_step[layer] % train_cfg.logging_step == 0 and global_step[layer] > 0:
                    avg_loss = loss_sum[layer] / train_cfg.logging_step
                    print(f"layer{layer} Step: {global_step[layer]}, Loss: {avg_loss}")
                    if _WANDB:
                        wandb.log({f"layer{layer}/train_loss": avg_loss}, step=global_step[layer])
                    loss_sum[layer] = 0.0

                    # evaluation
                    # del dl_train, optimizer, lr_scheduler
                    sae.eval()
                    loss_eval = 0
                    total_eval_steps = 0

                    with torch.no_grad():
                        for i, val_batch in enumerate(dl_val):
                            if i >= train_cfg.val_check_batches:
                                break
                            with torch.inference_mode():
                                _ = model(val_batch.to(MODEL_DEVICE), use_cache=False)
                            activation = hook.output[0] if isinstance(hook.output, tuple) else hook.output
                            activation = activation[:, 1:, :]
                            activation = activation.flatten(0, 1)
                            activation = normalize_activation(activation, nl)
                            for val_chunk in torch.chunk(activation, train_cfg.inf_bs_expansion, dim=0):
                                out = sae(val_chunk.to(SAE_DEVICE))
                                loss_eval += out.loss.item()
                                total_eval_steps += 1
                        val_avg = loss_eval / max(total_eval_steps, 1)
                        print(f"Step {global_step[layer]} L{layer}   val/loss: {val_avg:.6f}")
                        if _WANDB:
                            wandb.log({f"layer{layer}/val_loss": val_avg}, step=global_step[layer])
                global_step[layer] += 1

    if _WANDB:
        wandb.finish()

    # save the trained SAE
    for layer, sae in saes.items():
        torch.save(sae.state_dict(), os.path.join(save_dir, f"sae_layer{layer}.pth"))

    print("Training finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=int, default=988240, help="Checkpoint")
    parser.add_argument(
        "--layers", type=int, nargs="+", default=12, help="Layer indices to extract activations"
    )
    parser.add_argument("--n_d", type=int, default=16, help="Expansion ratio (n/d)")
    parser.add_argument(
        "--k", type=int, default=32, help="K parameter for SAE (sparsity)"
    )
    parser.add_argument(
        "--nl",
        type=str,
        default="Scalar",
        help="normalization method: Standardization, Scalar, None",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--label", type=str, default=None, help="Data label")
    parser.add_argument(
        "--model_name_or_dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    usr_cfg = UsrConfig()
    train_cfg = TrainConfig()
    train_file_name = "train_data.pt"
    val_file_name = "val_data.pt"
    if args.label:
        train_file_name = args.label + train_file_name
        val_file_name = args.label + val_file_name
    train_data_pth = os.path.join(usr_cfg.tokenized_data_dir, train_file_name)
    val_data_pth = os.path.join(usr_cfg.tokenized_data_dir, val_file_name)

    # Use manifest-based datasets if available, fallback to old format
    # Try to find manifest files with dataset names
    import glob
    manifest_pattern = os.path.join(usr_cfg.tokenized_data_dir, f"{args.label or ''}*_train_manifest.json")
    train_manifests = glob.glob(manifest_pattern)
    train_manifest = train_manifests[0] if train_manifests else None

    manifest_pattern = os.path.join(usr_cfg.tokenized_data_dir, f"{args.label or ''}*_val_manifest.json")
    val_manifests = glob.glob(manifest_pattern)
    val_manifest = val_manifests[0] if val_manifests else None

    if train_manifest and os.path.exists(train_manifest):
        from dataset import StreamingShardedDataset
        dataset_train = StreamingShardedDataset(train_manifest, cache_size=4)
        dataset_val = StreamingShardedDataset(val_manifest, cache_size=2)
    else:
        dataset_train = CustomWikiDataset(train_data_pth)
        dataset_val = CustomWikiDataset(val_data_pth)

    dl_train = DataLoader(
        dataset_train,
        batch_size=train_cfg.batch_size * train_cfg.inf_bs_expansion,
        shuffle=True,
    )
    k = train_cfg.val_check_batches * (train_cfg.batch_size * train_cfg.inf_bs_expansion)
    indices = torch.randperm(len(dataset_val))[:k].tolist()
    sampler = SubsetRandomSampler(indices)
    dl_val = DataLoader(
        dataset_val,
        batch_size=train_cfg.batch_size * train_cfg.inf_bs_expansion,
        sampler=sampler,
    )

    model_name_or_dir = usr_cfg.model_name_or_dir
    if args.model_name_or_dir:
        model_name_or_dir = args.model_name_or_dir

    print(
        f"Layers: {args.layers}, n/d: {args.n_d}, k: {args.k}, nl: {args.nl}, ckpt: {args.ckpt}, lr: {args.lr}"
    )
    sae_root_dir = usr_cfg.sae_save_dir
    if args.label:
        sae_root_dir = args.label + sae_root_dir
    save_dir = return_save_dir(
        sae_root_dir,
        args.n_d,
        args.k,
        args.nl,
        args.ckpt,
        args.lr,
    )
    layers = []
    for layer in args.layers:
        if os.path.exists(os.path.join(save_dir, f"sae_layer{layer}.pth")):
            print(f"Already exists at: {save_dir}")
            if input("Overwrite? (y/n): ").lower() != "y":
                continue
        layers.append(layer)
    os.makedirs(save_dir, exist_ok=True)

    train(
        dl_train,
        dl_val,
        train_cfg,
        model_name_or_dir,
        layers,
        args.n_d,
        args.k,
        args.nl,
        args.ckpt,
        args.lr,
        save_dir,
    )

if __name__ == "__main__":
    main()
