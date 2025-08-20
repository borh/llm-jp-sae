import argparse
import json
import math
import os

import torch
from config import EvalConfig, SaeConfig, TrainConfig, UsrConfig, return_save_dir
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import ActivationRecord, CustomWikiDataset, FeatureRecord
from model import SimpleHook, SparseAutoEncoder, normalize_activation


def _select_devices(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cpu")
    elif torch.cuda.device_count() > 1:
        return torch.device("cuda:0"), torch.device("cuda:1")
    else:
        return torch.device("cuda"), torch.device("cuda")


def collect_feature_pattern(
    dl_test,
    model_dir,
    layer,
    n_d,
    k,
    nl,
    ckpt,
    lr,
    save_dir,
    num_examples,
    act_threshold_p,
    force_cpu=False,
    chunk_size=8,
):
    MODEL_DEVICE, SAE_DEVICE = _select_devices(force_cpu)
    # Support both a HF model id (e.g. "llm-jp/llm-jp-3-1.8b") and a local dir
    # that contains subfolders named iter_{ckpt}. Prefer local iter_* if present.
    ckpt_dir = os.path.join(model_dir, f"iter_{str(ckpt).zfill(7)}")
    model_path = ckpt_dir if os.path.isdir(ckpt_dir) else model_dir
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(MODEL_DEVICE)
    model.eval()
    # Load SAE weights first to detect correct d_in
    sae_state_dict = torch.load(
        os.path.join(save_dir, f"sae_layer{layer}.pth"), map_location="cpu"
    )
    if "encoder.weight" in sae_state_dict:
        detected_d_in = sae_state_dict["encoder.weight"].shape[
            1
        ]  # in_features of encoder
    else:
        raise Exception(f"Could not automatically detect d_in: {sae_state_dict}")

    # https://github.com/pytorch/pytorch/issues/153195
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

    # Construct SAE config with correct d_in
    sae_config = SaeConfig(d_in=detected_d_in, expansion_factor=n_d, k=k)
    sae = SparseAutoEncoder(sae_config).to(SAE_DEVICE)
    sae.eval()
    sae.load_state_dict(sae_state_dict)

    # sanity print to confirm shapes / sizes
    try:
        print(
            f"[SAE] layer={layer} d_in={sae.d_in} expansion_factor={sae.cfg.expansion_factor} "
            f"num_latents={sae.num_latents} device={SAE_DEVICE}"
        )
    except Exception:
        # non-fatal: don't crash on debug print failure
        pass

    sae = torch.compile(sae)

    # Load tokenizer from the provided model_dir / HF id
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    features_dir = os.path.join(save_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    hook_layer = (
        model.model.embed_tokens if layer == 0 else model.model.layers[layer - 1]
    )
    hook = SimpleHook(hook_layer)
    hook_sae = SimpleHook(sae.encoder)

    # authoritative number of latent features coming from the SAE instance
    num_features = int(sae.num_latents)

    with torch.no_grad():
        # Find the activation threshold (act_threshold) for each feature
        max_act_values = torch.zeros(num_features, dtype=torch.bfloat16).to(SAE_DEVICE)
        for step, batch in tqdm(
            enumerate(dl_test), desc="SAE activation threshold extraction"
        ):
            # Ensure batch is 2D even if batch_size=1
            if batch.dim() == 1:
                batch = batch.unsqueeze(0)

            # Process in chunks
            for i in range(0, batch.shape[0], chunk_size):
                batch_chunk = batch[i : i + chunk_size].to(MODEL_DEVICE)

                # with torch.autocast(device_type="cuda"):
                _ = model(batch_chunk, use_cache=False)
                activation = hook.output if layer == 0 else hook.output[0]
                # Ensure activation is 3D (batch, seq, hidden)
                if activation.dim() == 2:
                    activation = activation.unsqueeze(0)
                activation = activation[:, 1:, :]
                shape = activation.shape  # bs, seq, d
                activation = activation.flatten(0, 1)
                activation = normalize_activation(activation, nl).to(SAE_DEVICE)
                _ = sae(activation)
                sae_activation = hook_sae.output.view(
                    shape[0], shape[1], -1
                )  # bs, seq, d_in * n_d
                max_act_values = torch.max(
                    max_act_values, sae_activation.flatten(0, 1).max(dim=0)[0]
                )

                # Clear intermediate tensors
                del activation, sae_activation, batch_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        act_thresholds = max_act_values * act_threshold_p

        cnt_full = 0
        feature_records = [FeatureRecord(feature_id=i) for i in range(num_features)]
        feature_idxs = torch.arange(num_features).to(SAE_DEVICE)
        feature_notfull = torch.ones(num_features, dtype=torch.bool).to(SAE_DEVICE)
        feature_cnt = torch.zeros(num_features, dtype=torch.int32).to(SAE_DEVICE)

        with tqdm(dl_test, desc="SAE feature extraction") as pbar:
            for step, batch in enumerate(pbar):
                # Ensure batch is 2D even if batch_size=1
                if batch.dim() == 1:
                    batch = batch.unsqueeze(0)

                pbar.set_postfix(cnt_full=cnt_full)

                # Process in chunks
                for chunk_start in range(0, batch.shape[0], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, batch.shape[0])
                    batch_chunk = batch[chunk_start:chunk_end].to(MODEL_DEVICE)

                    try:
                        _ = model(batch_chunk, use_cache=False)
                        activation = hook.output if layer == 0 else hook.output[0]
                        # Ensure activation is 3D (batch, seq, hidden)
                        if activation.dim() == 2:
                            activation = activation.unsqueeze(0)
                        activation = activation[:, 1:, :]
                        shape = activation.shape
                        activation = activation.flatten(0, 1)
                        activation = normalize_activation(activation, nl).to(SAE_DEVICE)
                        out = sae(activation)
                    except Exception as e:
                        # Debug info dump
                        print("\n[DEBUG] SAE forward crashed:", e)
                        print("[DEBUG] SAE checkpoint parameters:")
                        for key, val in sae_state_dict.items():
                            print(f"    {key}: {tuple(val.shape)}")
                        try:
                            enc_out = sae.encoder(activation)
                            print(
                                f"[DEBUG] Encoder output shape: {tuple(enc_out.shape)}"
                            )
                            print(f"[DEBUG] Expected num_latents: {sae.num_latents}")
                            print(
                                f"[DEBUG] Activation shape entering SAE: {tuple(activation.shape)}"
                            )
                        except Exception as inner_e:
                            print("[DEBUG] Failed to run encoder separately:", inner_e)
                        raise  # re‑raise to keep current error handling
                    latent_indices = out.latent_indices.view(shape[0], shape[1], k)
                    latent_acts = out.latent_acts.view(shape[0], shape[1], k)
                    sae_activation = hook_sae.output.view(shape[0], shape[1], -1)

                    # bounds check BEFORE any indexing to avoid GPU OOB
                    try:
                        li_min = int(latent_indices.min().item())
                        li_max = int(latent_indices.max().item())
                    except Exception:
                        raise RuntimeError(
                            f"Failed to inspect latent_indices for bounds check; shape={latent_indices.shape}"
                        )
                    if li_min < 0 or li_max >= num_features:
                        raise RuntimeError(
                            f"latent_indices out of bounds for SAE: min={li_min}, max={li_max}, "
                            f"num_features={num_features}, layer={layer}, save_dir={save_dir}"
                        )

                    # safe to index now
                    exceed_position = act_thresholds[latent_indices] < latent_acts

                    for batch_idx in tqdm(range(shape[0]), leave=False):
                        # Adjust the actual batch index for the full batch
                        actual_batch_idx = chunk_start + batch_idx
                        feature_idxs_notfull = feature_idxs[feature_notfull]
                        indices = latent_indices[batch_idx][
                            exceed_position[batch_idx]
                        ].unique()
                        tokens = [
                            w.replace("▁", " ")
                            for w in tokenizer.convert_ids_to_tokens(
                                batch[actual_batch_idx][1:]
                            )
                        ]
                        for idx in indices:
                            if idx in feature_idxs_notfull:
                                act_values = sae_activation[batch_idx, :, idx].tolist()
                                feature_records[idx].act_patterns.append(
                                    ActivationRecord(
                                        tokens=tokens, act_values=act_values
                                    )
                                )
                                feature_cnt[idx] += 1
                                if feature_cnt[idx] == num_examples:
                                    feature_notfull[idx] = False
                                    save_token_act(feature_records[idx], features_dir)
                                    feature_records[idx] = None
                                    cnt_full += 1
                                elif feature_cnt[idx] > num_examples:
                                    raise ValueError(
                                        "Feature count exceeds num_examples"
                                    )
    for feature_record in tqdm(feature_records, desc="save remaining"):
        if feature_record is not None and len(feature_record.act_patterns) > 0:
            save_token_act(feature_record, features_dir)


def _format_activation_record(activation_record, max_act):
    tokens = activation_record.tokens
    acts = activation_record.act_values
    token_act_list = []

    for token, act in zip(tokens, acts):
        if act < 0:
            act = 0
        else:
            act = math.ceil(act * 10 / max_act)
        token_act_list.append([token, act])

    return token_act_list


def format_example(activation_records, max_act):
    token_act_list = []
    for idx_sample, activation_record in enumerate(activation_records):
        token_act = _format_activation_record(activation_record, max_act)
        token_act_list.append(token_act)

    return token_act_list


def save_token_act(
    feature_record,
    features_dir,
):
    len_data = len(feature_record.act_patterns)
    max_act = max(
        [max(feature_record.act_patterns[i].act_values) for i in range(len_data)]
    )

    token_act_list = format_example(
        feature_record.act_patterns,
        max_act,
    )

    with open(
        os.path.join(features_dir, f"{feature_record.feature_id}.json"), "w"
    ) as f:
        json.dump(
            {
                "token_act": token_act_list,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=int, default=988240, help="Checkpoint")
    parser.add_argument(
        "--layer", type=int, default=12, help="Layer index to extract activations"
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
    parser.add_argument("--chunk-size", type=int, default=8, help="Chunk size")
    args = parser.parse_args()

    usr_cfg = UsrConfig()
    eval_cfg = EvalConfig()
    train_cfg = TrainConfig()

    # Try manifest first, fallback to old format
    import glob

    manifest_pattern = os.path.join(usr_cfg.tokenized_data_dir, "*_test_manifest.json")
    test_manifests = glob.glob(manifest_pattern)
    test_manifest = test_manifests[0] if test_manifests else None

    if test_manifest and os.path.exists(test_manifest):
        from dataset import StreamingShardedDataset

        dataset_test = StreamingShardedDataset(test_manifest, cache_size=2)
    else:
        test_data_pth = os.path.join(usr_cfg.tokenized_data_dir, "test_data.pt")
        dataset_test = CustomWikiDataset(test_data_pth)
    dl_test = DataLoader(
        dataset_test,
        batch_size=1,  # train_cfg.batch_size * train_cfg.inf_bs_expansion,
        shuffle=False,
    )

    save_dir = return_save_dir(
        usr_cfg.sae_save_dir,
        args.layer,
        args.n_d,
        args.k,
        args.nl,
        args.ckpt,
        args.lr,
    )
    collect_feature_pattern(
        dl_test,
        usr_cfg.model_name_or_dir,
        args.layer,
        args.n_d,
        args.k,
        args.nl,
        args.ckpt,
        args.lr,
        save_dir,
        eval_cfg.num_examples,
        eval_cfg.act_threshold_p,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
