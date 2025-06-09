# Utilities for performing imputation with trained PyTorch DeepImpute models

from typing import Dict
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .datareader import DataReader
from .model import DeepImpute


class TestDataset(Dataset):
    """Dataset used for inference. It simply one-hot encodes samples."""

    def __init__(self, data: np.ndarray, depth: int):
        self.data = torch.as_tensor(data, dtype=torch.long)
        self.depth = depth

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.nn.functional.one_hot(sample, num_classes=self.depth).float()


def load_model(path: str, device: torch.device) -> DeepImpute:
    """Load a saved DeepImpute model."""
    model = torch.load(path, map_location=device)
    model.eval()
    return model


def impute_the_target(args: Dict) -> None:
    """Impute missing genotypes for the provided target file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args["batch_size_per_gpu"] * (torch.cuda.device_count() or 1)
    if args.get("target") is None:
        raise ValueError("Target file missing for imputation. use --target to specify a target file.")

    # load training configuration if available
    config_path = os.path.join(args["save_dir"], "commandline_args.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            training_args = json.load(f)
        args["sites_per_model"] = training_args.get("sites_per_model", args.get("sites_per_model"))
        args["tihp"] = training_args.get("tihp", args.get("tihp"))
        args["cs"] = training_args.get("cs", args.get("cs"))
        args["co"] = training_args.get("co", args.get("co"))

    dr = DataReader()
    dr.assign_training_set(
        file_path=args["ref"],
        target_is_gonna_be_phased_or_haps=args["tihp"],
        variants_as_columns=args.get("ref_vac", False),
        delimiter=args.get("ref_sep"),
        file_format=args.get("ref_file_format", "infer"),
        first_column_is_index=args.get("ref_fcai", True),
        comments=args.get("ref_comment", "##"),
    )
    dr.assign_test_set(
        file_path=args["target"],
        variants_as_columns=args.get("target_vac", False),
        delimiter=args.get("target_sep"),
        file_format=args.get("target_file_format", "infer"),
        first_column_is_index=args.get("target_fcai", True),
        comments=args.get("target_comment", "##"),
    )

    all_preds = []
    break_points = list(np.arange(0, dr.VARIANT_COUNT, args["sites_per_model"])) + [dr.VARIANT_COUNT]
    for w in range(len(break_points) - 1):
        print(f"Imputing chunk {w + 1}/{len(break_points) - 1}")
        final_start_pos = max(0, break_points[w] - 2 * args["co"])
        final_end_pos = min(dr.VARIANT_COUNT, break_points[w + 1] + 2 * args["co"])
        test_np = dr.get_target_set(final_start_pos, final_end_pos).astype(np.int64)
        dataset = TestDataset(test_np, dr.SEQ_DEPTH)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model_path = os.path.join(args["save_dir"], "models", f"w_{w}.pt")
        model = load_model(model_path, device)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                preds.append(out.cpu())
        all_preds.append(torch.cat(preds, dim=0).numpy())

    all_preds = np.hstack(all_preds)
    dest = dr.write_ligated_results_to_file(
        dr.preds_to_genotypes(all_preds),
        os.path.join(args["save_dir"], "out", "ligated_results"),
        compress=args.get("compress_results", True),
    )
    print(f"Done! Please find the file at {dest}")
