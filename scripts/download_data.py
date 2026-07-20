"""
Download the Stack Exchange Q&A corpus and save it locally as Parquet.

Run once, from the project root:
    python scripts/download_data.py
    python scripts/download_data.py --configs datascience ai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

# Running "python scripts/download_data.py" puts scripts/ on sys.path,
# not the project root -- so "from config.settings import settings"
# would fail. Adding the root explicitly fixes it. (Module 9 replaces
# this with a proper editable install; this is the pragmatic fix now.)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402


def list_available_configs() -> list[str]:
    """Ask the Hub which per-site configs this dataset actually exposes."""
    print(f"Inspecting dataset: {settings.dataset_name}")
    configs = get_dataset_config_names(settings.dataset_name)
    print(f"Found {len(configs)} configs.")
    return configs


def download_config(config: str) -> pd.DataFrame:
    """Download a single Stack Exchange site and return it as a DataFrame."""
    print(f"\n--- {config} ---")
    dataset = load_dataset(settings.dataset_name, config, split="train")

    df = dataset.to_pandas()
    # Provenance: once we concatenate sites, we need to know the origin
    # of every row. This becomes retrieval metadata later.
    df["source_site"] = config

    print(f"  rows    : {len(df):,}")
    print(f"  columns : {list(df.columns)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Stack Exchange FAQ data.")
    parser.add_argument(
        "--configs",
        nargs="+",                       # accept one or more values
        default=settings.dataset_configs,
        help="Stack Exchange sites to download.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",             # a flag: present = True
        help="Print all available configs and exit.",
    )
    args = parser.parse_args()

    if args.list_configs:
        for name in list_available_configs():
            print(f"  {name}")
        return

    available = list_available_configs()
    requested = args.configs

    missing = [c for c in requested if c not in available]
    if missing:
        print(f"\nERROR: configs not found: {missing}")
        print("Run with --list-configs to see valid names.")
        sys.exit(1)

    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for config in requested:
        df = download_config(config)
        # Save each site separately so you can re-mix later without
        # re-downloading anything.
        df.to_parquet(settings.raw_data_dir / f"{config}.parquet", index=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output = settings.raw_data_dir / "stackexchange_raw.parquet"
    combined.to_parquet(output, index=False)

    print("\n" + "=" * 60)
    print(f"Total rows : {len(combined):,}")
    print(f"Saved to   : {output.relative_to(settings.project_root)}")
    print("\nPer-site counts:")
    print(combined["source_site"].value_counts().to_string())
    print("\nFirst record:")
    for column in combined.columns:
        value = str(combined.iloc[0][column])
        preview = value[:200] + ("..." if len(value) > 200 else "")
        print(f"  {column}: {preview}")
    print("=" * 60)


if __name__ == "__main__":
    main()