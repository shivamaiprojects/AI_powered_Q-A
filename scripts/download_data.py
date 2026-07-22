from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Xet is an accelerated transfer backend backed by a native DLL. Windows
# Application Control policies commonly block it; standard HTTPS transfer
# is used instead. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402

PARQUET_REVISION = "refs/convert/parquet"


def list_repo_parquet_files() -> list[str]:
    """Return every Parquet file path on the auto-converted branch."""
    api = HfApi()
    files = api.list_repo_files(
        repo_id=settings.dataset_name,
        repo_type="dataset",
        revision=PARQUET_REVISION,
    )
    return [f for f in files if f.endswith(".parquet")]


def available_configs(parquet_files: list[str]) -> list[str]:
    """Derive config names from the top-level directory of each file path."""
    return sorted({f.split("/")[0] for f in parquet_files})


def download_config(config: str, parquet_files: list[str]) -> pd.DataFrame:
    """Download and concatenate every Parquet shard belonging to one config."""
    shards = sorted(f for f in parquet_files if f.startswith(f"{config}/"))
    if not shards:
        raise ValueError(f"No Parquet files found for config '{config}'.")

    print(f"\n--- {config} ({len(shards)} shard(s)) ---")

    frames = []
    for shard in shards:
        local_path = hf_hub_download(
            repo_id=settings.dataset_name,
            filename=shard,
            repo_type="dataset",
            revision=PARQUET_REVISION,
        )
        frames.append(pd.read_parquet(local_path))

    df = pd.concat(frames, ignore_index=True)
    df["source_site"] = config

    print(f"  rows    : {len(df):,}")
    print(f"  columns : {list(df.columns)}")
    return df


def summarise(df: pd.DataFrame, output: Path) -> None:
    """Print a schema and content summary of the combined corpus."""
    print("\n" + "=" * 60)
    print(f"Total rows : {len(df):,}")
    print(f"Saved to   : {output.relative_to(settings.project_root)}")

    print("\nPer-site counts:")
    print(df["source_site"].value_counts().to_string())

    print("\nCharacter lengths:")
    for column in df.select_dtypes(include=["object", "str"]).columns:
        if column == "source_site":
            continue
        lengths = df[column].astype(str).str.len()
        print(
            f"  {column:<16} min={lengths.min():>6,}  "
            f"median={int(lengths.median()):>6,}  max={lengths.max():>7,}"
        )

    print("\nFirst record:")
    for column in df.columns:
        value = str(df.iloc[0][column])
        preview = value[:300] + ("..." if len(value) > 300 else "")
        print(f"\n  [{column}]\n  {preview}")
    print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Stack Exchange FAQ data.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=settings.dataset_configs,
        help="Stack Exchange sites to download.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print all available configs and exit.",
    )
    args = parser.parse_args()

    print(f"Inspecting dataset: {settings.dataset_name}")
    parquet_files = list_repo_parquet_files()
    configs = available_configs(parquet_files)
    print(f"Found {len(configs)} configs.")

    if args.list_configs:
        for name in configs:
            print(f"  {name}")
        return

    missing = [c for c in args.configs if c not in configs]
    if missing:
        print(f"\nERROR: configs not found: {missing}")
        print("Run with --list-configs to see valid names.")
        sys.exit(1)

    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for config in args.configs:
        df = download_config(config, parquet_files)
        df.to_parquet(settings.raw_data_dir / f"{config}.parquet", index=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output = settings.raw_data_dir / "stackexchange_raw.parquet"
    combined.to_parquet(output, index=False)

    summarise(combined, output)


if __name__ == "__main__":
    main()