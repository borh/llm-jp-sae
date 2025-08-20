"""
Run collect_feature_pattern + evaluate.auto_language for multiple model:sae_root pairs
using manifest files from the new sharded dataset format.

Example:
  python run_eval_all.py \
    --models \
      "llm-jp/llm-jp-3-1.8b:sae/SAEs-LLM-jp-3-1.8B-dolma:data/tokenized/llmjp_dolma_test_manifest.json" \
      "allenai/OLMo-2-0425-1B:sae/SAEs-OLMo-2-0425-1B-dolma:data/tokenized/olmo2_dolma_warp_html_test_manifest.json" \
    --n_d 16 --k 32 --nl Scalar --ckpt 988240 --lr 0.001
"""

import argparse
import os
from pathlib import Path

from torch.utils.data import DataLoader
from config import return_save_dir, TrainConfig
from dataset import CustomWikiDataset, StreamingShardedDataset
from collect_feature_pattern import collect_feature_pattern
from evaluate import auto_language


def detect_layers(save_dir: Path):
    if not save_dir.exists():
        return []
    layers = []
    for fname in os.listdir(save_dir):
        if fname.startswith("sae_layer") and fname.endswith(".pth"):
            try:
                num = int(fname[len("sae_layer") : -4])
                layers.append(num)
            except Exception:
                continue
    return sorted(set(layers))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="model_id:sae_root:manifest_path triples (manifest_path is a test manifest .json file)",
    )
    p.add_argument("--n_d", type=int, default=16)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--nl", type=str, default="Scalar")
    p.add_argument("--ckpt", type=int, default=988240)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_examples", type=int, default=50)
    p.add_argument("--act_threshold_p", type=float, default=0.7)
    p.add_argument(
        "--batch-size", type=int, default=None, help="override DataLoader batch size"
    )
    p.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force model and SAE to run on CPU",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Process activations in chunks of this size to reduce memory",
    )
    args = p.parse_args()

    batch_size = args.batch_size or (
        TrainConfig().batch_size * TrainConfig().inf_bs_expansion
    )

    # parse model_id:sae_root:manifest_path triples
    pairs = []
    for token in args.models:
        parts = token.split(":")
        if len(parts) != 3:
            raise SystemExit(
                "Each --models entry must be model_id: sae_root:manifest_path"
            )
        model_id, sae_root, manifest_path = parts
        manifest_path = Path(manifest_path.strip())
        if not manifest_path.exists():
            raise SystemExit(f"Manifest file not found: {manifest_path}")
        pairs.append((model_id.strip(), sae_root.strip(), manifest_path))

    for model_id, sae_root, manifest_path in pairs:
        print(f"\n=== MODEL {model_id}    SAE root: {sae_root} ===")
        print(f"Using manifest: {manifest_path}")

        # Build DataLoader for this model using the manifest
        if manifest_path.suffix == ".json":
            dataset = StreamingShardedDataset(str(manifest_path), cache_size=2)
        else:
            # Fallback for old .pt files if needed
            dataset = CustomWikiDataset(str(manifest_path))
        dl_test = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        save_dir = Path(
            return_save_dir(sae_root, 0, args.n_d, args.k, args.nl, args.ckpt, args.lr)
        )
        layers = detect_layers(save_dir)
        if not layers:
            print(f"No sae_layer*.pth files found in {save_dir}; skipping.")
            continue
        print("Detected layers:", layers)

        for layer in layers:
            print(f"\n-- layer {layer} --")
            try:
                collect_feature_pattern(
                    dl_test=dl_test,
                    model_dir=model_id,
                    layer=layer,
                    n_d=args.n_d,
                    k=args.k,
                    nl=args.nl,
                    ckpt=args.ckpt,
                    lr=args.lr,
                    save_dir=str(save_dir),
                    num_examples=args.num_examples,
                    act_threshold_p=args.act_threshold_p,
                    force_cpu=args.force_cpu,
                    chunk_size=args.chunk_size,  # Add this line
                )
            except Exception as e:
                print(f"collect_feature_pattern failed for layer {layer}: {e}")
                continue

        features_dir = save_dir / "features"
        if features_dir.exists():
            print(f"Running auto_language on {features_dir}")
            try:
                auto_language(str(features_dir))
            except Exception as e:
                print(f"auto_language failed for {features_dir}: {e}")
        else:
            print(f"No features directory at {features_dir} - nothing to evaluate.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
