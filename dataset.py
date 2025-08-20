import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import torch
from torch.utils.data import Dataset
from safetensors.torch import load_file

class CustomWikiDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=True)
        self.all_len = self.data.shape[0]

    def __len__(self):
        return self.all_len

    def __getitem__(self, idx):
        return self.data[idx]

class StreamingShardedDataset(Dataset):
    """Memory-efficient dataset that loads shards on-demand."""

    def __init__(self, manifest_path: str, cache_size: int = 2):
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.shards = self.manifest["shards"]
        self.cache_size = cache_size
        self.shard_cache = {}
        self.row_to_shard = []

        # Build row index
        for shard_idx, shard in enumerate(self.shards):
            for _ in range(shard["rows"]):
                self.row_to_shard.append(shard_idx)

    def __len__(self):
        return self.manifest["total_rows"]

    def _load_shard(self, shard_idx: int) -> torch.Tensor:
        if shard_idx in self.shard_cache:
            return self.shard_cache[shard_idx]

        # Evict oldest cached shard if needed
        if len(self.shard_cache) >= self.cache_size:
            oldest = min(self.shard_cache.keys())
            del self.shard_cache[oldest]

        # Load shard
        shard_info = self.shards[shard_idx]
        shard_path = Path(shard_info["source_dir"]) / shard_info["file"]
        data = load_file(shard_path)
        self.shard_cache[shard_idx] = data["input_ids"]

        return data["input_ids"]

    def __getitem__(self, idx):
        shard_idx = self.row_to_shard[idx]
        shard_data = self._load_shard(shard_idx)

        # Calculate local index within shard
        shard_start = sum(self.shards[i]["rows"] for i in range(shard_idx))
        local_idx = idx - shard_start

        return shard_data[local_idx]

@dataclass
class ActivationRecord:
    tokens: List[str]
    act_values: List[float]


@dataclass
class FeatureRecord:
    feature_id: int
    act_patterns: List[ActivationRecord] = field(default_factory=list)
