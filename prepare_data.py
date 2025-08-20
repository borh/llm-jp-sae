import argparse
import gzip
import json
import os
import random
import time
from pathlib import Path
from typing import Iterator, List

import requests
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from safetensors.torch import save_file

from config import DataConfig, UsrConfig

# -----------------------------------------------------------------------------
# GLOBALS ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

CHUNK_SIZE = 1 << 14  # 16 KiB per stream read
PROGRESS_EVERY = 100  # flush cadence (batches) when tokenising
SEED = 42  # reproducibility

# -----------------------------------------------------------------------------
# RESUMABLE I/O UTILS ---------------------------------------------------------
# -----------------------------------------------------------------------------


def _download_url(url: str, dest: Path, retry: int = 3) -> None:
    """Download *url* to *dest* streaming with progress.

    * Skip if *dest* already exists and non‑empty.
    * Retry up to *retry* times.
    * On SSL certificate failure, retry once with `verify=False`.
    * Write to temporary `*.part` then atomic rename.
    """
    # if dest.exists() and dest.stat().st_size > 0:
    #     return

    dest.parent.mkdir(parents=True, exist_ok=True)

    verify = False
    attempt = 0
    while attempt < retry:
        try:
            with requests.get(url, stream=True, timeout=30, verify=verify) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0) or 0)
                tmp = dest.with_suffix(".part")
                with (
                    open(tmp, "wb") as f,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"↯ {dest.name}",
                        leave=False,
                    ) as bar,
                ):
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            os.replace(tmp, dest)
            print(f"  ✓ downloaded {dest.name}")
            return
        except requests.exceptions.SSLError:
            if verify:
                import warnings

                import urllib3

                warnings.warn(
                    f"SSL verify failed for {url}. Retrying insecurely.", RuntimeWarning
                )
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                verify = False
                continue
            attempt += 1
        except Exception as e:
            attempt += 1
            print(f"  download error ({attempt}/{retry}) for {url}: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}")


def _iter_jsonl(path: Path) -> Iterator[str]:
    """Yield the *text* field for every JSON‑line in *path* (handles .gz)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
                yield obj.get("text") or obj.get("content", "")
            except Exception:
                continue  # skip malformed


# -----------------------------------------------------------------------------
# SAMPLING WITH CACHE ---------------------------------------------------------
# -----------------------------------------------------------------------------


def _load_or_sample(lines: List[str], rate: float, cache_path: Path) -> List[str]:
    """Sample *rate* proportion of *lines* (line‑level), caching result."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = [line.rstrip("\n") for line in f]
        print(
            f"  ✓ loaded cached sample ({len(cached)}/{len(lines)}) from {cache_path.name}"
        )
        return cached

    sample_size = max(1, int(len(lines) * rate))
    print(f"  Sampling {sample_size}/{len(lines)} lines …")
    random.seed(42)
    sampled = random.sample(lines, sample_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for t in sampled:
            f.write(t.replace("\n", " ") + "\n")
    print(f"  ✓ sample saved → {cache_path.name}")
    return sampled


# -----------------------------------------------------------------------------
# TOKENISATION HELPERS --------------------------------------------------------
# -----------------------------------------------------------------------------


def _token_file(name: str, save_dir: Path) -> Path:
    return save_dir / f"tokenized_{name}.pt"


# Core functional components
def stream_texts(txt_file_path: Path) -> Iterator[str]:
    """Pure function: Stream texts from file."""
    dataset = load_dataset("text", data_files=str(txt_file_path), streaming=True)
    return (example["text"] for example in dataset["train"])


def tokenize_text(tokenizer: AutoTokenizer, seq_len: int) -> callable:
    """Higher-order function: Returns a tokenization function."""

    def _tokenize(text: str) -> torch.Tensor:
        result = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=seq_len + 1,
            return_tensors="pt",
        )
        return result["input_ids"].squeeze(0)

    return _tokenize


def is_valid_sequence(pad_token_id: int) -> callable:
    """Higher-order function: Returns a validation predicate."""

    def _is_valid(token_ids: torch.Tensor) -> bool:
        return not token_ids.eq(pad_token_id).all()

    return _is_valid


def process_text_stream(
    texts: Iterator[str], tokenize_fn: callable, is_valid_fn: callable
) -> Iterator[torch.Tensor]:
    """Pure function: Transform and filter text stream."""
    for text in texts:
        tokens = tokenize_fn(text)
        if is_valid_fn(tokens):
            yield tokens


def save_tokens_incrementally(
    token_stream: Iterator[torch.Tensor], output_path: Path, batch_size: int = 10000
) -> int:
    """Save tokens in batches to final .pt file."""

    def batch_generator():
        """Convert token stream to tensor batches."""
        batch = []
        for tokens in token_stream:
            batch.append(tokens)
            if len(batch) >= batch_size:
                yield torch.stack(batch)
                batch = []
        if batch:
            yield torch.stack(batch)

    # Save batches to temporary files, then concatenate
    temp_files = []
    total_count = 0

    for i, batch_tensor in enumerate(
        tqdm(batch_generator(), desc="Processing batches")
    ):
        temp_file = output_path.with_suffix(f".temp{i}.pt")
        torch.save(batch_tensor, temp_file)
        temp_files.append(temp_file)
        total_count += batch_tensor.size(0)
        del batch_tensor

    # Concatenate all temp files into final output
    if temp_files:
        print(f"  Concatenating {len(temp_files)} temporary files...")
        all_tensors = []
        for temp_file in temp_files:
            tensor = torch.load(temp_file)
            all_tensors.append(tensor)
            temp_file.unlink()  # Delete immediately after loading

        final_tensor = torch.cat(all_tensors, dim=0)
        torch.save(final_tensor, output_path)
        del all_tensors, final_tensor

    return total_count


def tokenize_corpus_streaming(
    name: str,
    tokenizer: AutoTokenizer,
    txt_file_path: Path,
    seq_len: int,
    save_dir: Path,
    batch_size: int = 10000,
    num_proc: int | None = None,
) -> Path:
    """Tokenize corpus and save as safetensors shards for memory-efficient loading."""
    
    output_dir = save_dir / f"tokenized_{name}"
    manifest_path = output_dir / "manifest.json"
    
    if manifest_path.exists():
        print(f"  ✓ tokens exist for {name} — skipping tokenisation")
        return output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if num_proc is None:
        import multiprocessing
        num_proc = min(multiprocessing.cpu_count(), 4)
    
    print(f"  Processing {name} with {num_proc} processes...")
    
    # Load dataset
    dataset = Dataset.from_text(str(txt_file_path))
    
    def tokenize_and_filter_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=seq_len + 1,
            return_tensors="np",
        )
        
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        input_ids = tokenized["input_ids"]
        valid_mask = (input_ids != pad_token_id).any(axis=1)
        
        return {"input_ids": input_ids[valid_mask].tolist()}
    
    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_and_filter_function,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=["text"],
        desc=f"Tokenizing {name}",
    )
    
    # Save as sharded safetensors
    shard_size = 50000  # rows per shard
    total_rows = len(tokenized_dataset)
    num_shards = (total_rows + shard_size - 1) // shard_size
    
    shard_info = []
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, total_rows)
        
        shard_data = tokenized_dataset.select(range(start_idx, end_idx))
        input_ids = torch.tensor(shard_data["input_ids"], dtype=torch.long)
        
        shard_path = output_dir / f"shard_{shard_idx:04d}.safetensors"
        save_file({"input_ids": input_ids}, shard_path)
        
        shard_info.append({
            "shard_idx": shard_idx,
            "start": start_idx,
            "end": end_idx,
            "rows": end_idx - start_idx,
            "file": f"shard_{shard_idx:04d}.safetensors"
        })
        
        print(f"  Saved shard {shard_idx + 1}/{num_shards}")
    
    # Save manifest
    manifest = {
        "name": name,
        "total_rows": total_rows,
        "seq_len": seq_len + 1,
        "num_shards": num_shards,
        "shards": shard_info
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  ✓ saved {total_rows} rows in {num_shards} shards → {output_dir}")
    return output_dir


# -----------------------------------------------------------------------------
# DATA PREPARATION ------------------------------------------------------------
# -----------------------------------------------------------------------------


def prepare_dolma(tmp_dir: Path, rate: float, label: str) -> List[str]:
    sample_path = "dolma_sample.txt"
    if label:
        sample_path = label + sample_path
    cache = tmp_dir / sample_path
    if cache.exists() and cache.stat().st_size > 0:
        print("  ✓ dolma_sample.txt found — skipping dolma shards")
        with open(cache, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    url_list = (
        "https://huggingface.co/datasets/allenai/dolma/raw/main/urls/v1_6-sample.txt"
    )
    url_file = tmp_dir / "dolma_urls.txt"
    _download_url(url_list, url_file)

    with open(url_file, "r", encoding="utf-8") as f:
        shard_urls = [u.strip() for u in f if u.strip()]

    shard_dir = tmp_dir / "dolma_shards"
    texts: List[str] = []
    for url in tqdm(shard_urls, desc="Processing dolma shards", unit="file"):
        dest = shard_dir / Path(url).name
        _download_url(url, dest)
        texts.extend(t for t in _iter_jsonl(dest) if t)
    print(f"  collected {len(texts)} dolma lines")
    return _load_or_sample(texts, rate, cache)


def prepare_ja_warp_html(tmp_dir: Path, rate: float, label: str) -> List[str]:
    sample_path = "warp_sample.txt"
    if label:
        sample_path = label + sample_path
    cache = tmp_dir / sample_path
    if cache.exists() and cache.stat().st_size > 0:
        print("  ✓ warp_sample.txt found — skipping warp_html download")
        with open(cache, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    base = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_warp_html/level0"
    shard_names = ["html_nii_01-06.jsonl.gz", "html_nii_07-12.jsonl.gz"]

    texts: List[str] = []
    for name in shard_names:
        url = f"{base}/{name}"
        dest = tmp_dir / name
        _download_url(url, dest)
        texts.extend(t for t in _iter_jsonl(dest) if t)
    print(f"  collected {len(texts)} warp_html lines")
    return _load_or_sample(texts, rate, cache)


# -----------------------------------------------------------------------------
# MAIN ------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Download texts and tokenize them.")
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="label",
    )
    parser.add_argument(
        "--model_name_or_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dolma_sample_rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--warp_sample_rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for parallel tokenization (default: auto-detect, max 8)",
    )
    args = parser.parse_args()

    # RNG reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)

    usr_cfg = UsrConfig()
    data_cfg = DataConfig()
    if args.label is not None:
        data_cfg.label = args.label
    if args.model_name_or_dir is not None:
        usr_cfg.model_name_or_dir = args.model_name_or_dir
    if args.dolma_sample_rate is not None:
        data_cfg.dolma_sample_rate = args.dolma_sample_rate
    if args.warp_sample_rate is not None:
        data_cfg.warp_sample_rate = args.warp_sample_rate
    usr_cfg.model_name_or_dir

    tmp_dir = Path(usr_cfg.raw_data_dir) / "tmp_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(usr_cfg.tokenized_data_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) PREPARE TEXT CORPORA ------------------------------------------------
    if data_cfg.dolma_sample_rate:
        print("▶ Preparing dolma sample …")
        dolma_texts = prepare_dolma(tmp_dir, data_cfg.dolma_sample_rate, data_cfg.label)
        print(f"  dolma sample: {len(dolma_texts)} lines")

    if data_cfg.warp_sample_rate:
        print("▶ Preparing ja_warp_html sample …")
        warp_texts = prepare_ja_warp_html(
            tmp_dir, data_cfg.warp_sample_rate, data_cfg.label
        )
        print(f"  warp_html sample: {len(warp_texts)} lines")

    # 2) TOKENISER -----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(usr_cfg.model_name_or_dir)
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad token.")

    # 3) TOKENISATION with streaming pipeline ---------------------------------
    dolma_name = "dolma"
    warp_name = "warp_html"
    if data_cfg.label:
        dolma_name = data_cfg.label + dolma_name
        warp_name = data_cfg.label + warp_name

    # Try-catch wrapper for tokenization
    try:
        if data_cfg.dolma_sample_rate:
            dolma_txt_path = tmp_dir / f"{dolma_name.replace('dolma', 'dolma_sample')}.txt"
            dolma_tok = tokenize_corpus_streaming(
                dolma_name,
                tokenizer,
                dolma_txt_path,
                data_cfg.seq_len,
                save_dir,
                num_proc=args.num_proc,  # Use command-line specified number
            )

        if data_cfg.warp_sample_rate:
            warp_txt_path = tmp_dir / f"{warp_name.replace('warp_html', 'warp_sample')}.txt"
            warp_tok = tokenize_corpus_streaming(
                warp_name,
                tokenizer,
                warp_txt_path,
                data_cfg.seq_len,
                save_dir,
                num_proc=args.num_proc,  # Use command-line specified number
            )
    except Exception as e:
        print(f"\n✗ Tokenization failed: {e}")
        print("Cleaning up any partial files...")
        # Clean up any partial/empty files
        for pattern in [f"tokenized_{dolma_name}.pt", f"tokenized_{warp_name}.pt"]:
            for partial_file in save_dir.glob(pattern):
                if partial_file.exists() and partial_file.stat().st_size < 1000:  # < 1KB = likely empty
                    partial_file.unlink()
                    print(f"  Removed partial file: {partial_file}")
        raise SystemExit(f"Preprocessing failed: {e}")

    # 4) COMBINE & SPLIT using memory-efficient sharding
    def create_split_manifests(source_dirs: list, save_dir: Path, ratios: list, label: str = ""):
        """Create train/val/test split manifests without loading data."""
        import json
        import random

        # Collect all shards from all sources
        all_shards = []
        dataset_names = []
        for source_dir in source_dirs:
            manifest_path = source_dir / "manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Extract clean dataset name from directory (e.g., "tokenized_olmo2_dolma" -> "olmo2_dolma")
            dir_name = source_dir.name
            if dir_name.startswith("tokenized_"):
                clean_name = dir_name[len("tokenized_"):]
            else:
                clean_name = manifest["name"]
            
            if clean_name not in dataset_names:
                dataset_names.append(clean_name)
            
            for shard in manifest["shards"]:
                all_shards.append({
                    **shard,
                    "source_dir": str(source_dir),
                    "source_name": clean_name
                })

        # Shuffle shard order for randomization
        random.seed(42)
        random.shuffle(all_shards)

        # Calculate split points
        total_rows = sum(s["rows"] for s in all_shards)
        n_train = int(total_rows * ratios[0])
        n_val = int(total_rows * ratios[1])

        # Assign shards to splits
        current_rows = 0
        train_shards, val_shards, test_shards = [], [], []

        for shard in all_shards:
            if current_rows < n_train:
                train_shards.append(shard)
            elif current_rows < n_train + n_val:
                val_shards.append(shard)
            else:
                test_shards.append(shard)
            current_rows += shard["rows"]

        # Save split manifests
        # Use clean dataset names without duplication
        combined_name = "_".join(sorted(dataset_names))
        for split_name, shards in [("train", train_shards), ("val", val_shards), ("test", test_shards)]:
            fname = f"{combined_name}_{split_name}_manifest.json"
            manifest = {
                "split": split_name,
                "total_rows": sum(s["rows"] for s in shards),
                "shards": shards
            }
            with open(save_dir / fname, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"  ✓ saved {split_name} manifest ({len(shards)} shards, {manifest['total_rows']} rows)")

    tokenized_dirs = []
    if data_cfg.dolma_sample_rate:
        tokenized_dirs.append(Path(save_dir) / f"tokenized_{dolma_name}")
    if data_cfg.warp_sample_rate:
        tokenized_dirs.append(Path(save_dir) / f"tokenized_{warp_name}")

    create_split_manifests(tokenized_dirs, save_dir, data_cfg.train_val_test_ratio, data_cfg.label or "")

    print("✔ All preprocessing complete.")


if __name__ == "__main__":
    main()
