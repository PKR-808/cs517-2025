#!/usr/bin/env python3
"""
–  Data‑prep utility for the playlist‑SMT project
────────────────────────────────────────────────────────────────
Reads a Spotify‑style CSV, applies user‑defined filters, and writes
a `.jsonl` file (one JSON object per track) ready for the solver.

Example
-------
python prep_tracks.py tracks.csv filtered.jsonl \
    --duration-min 120 --duration-max 300 \
    --pop-min 40 \
    --allowed-genres acoustic,folk \
    --verbose
"""

import argparse
import json
import pathlib
import sys
from typing import List

import pandas as pd


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def comma_split(arg: str) -> List[str]:
    """Split a comma‑separated CLI string into a list; empty string → []."""
    return [s.strip() for s in arg.split(",") if s.strip()]


def load_and_filter(
    csv_path: pathlib.Path,
    dur_min: int,
    dur_max: int,
    pop_min: int,
    allowed_genres: List[str],
    verbose: bool,
) -> pd.DataFrame:
    """Load CSV, apply filters, and add derived fields."""
    df = pd.read_csv(csv_path)

    # Duration filter (ms → within [dur_min, dur_max] minutes)
    dur_min_ms, dur_max_ms = dur_min * 1000, dur_max * 1000
    df = df.query("@dur_min_ms <= duration_ms <= @dur_max_ms")

    # Popularity floor
    df = df.query("popularity >= @pop_min")

    # Allowed genres (optional)
    if allowed_genres:
        df = df[df["genre"].isin(allowed_genres)]

    # Drop rows lacking required numeric fields
    req_cols = ["valence", "energy", "duration_ms"]
    df = df.dropna(subset=req_cols)

    # Derived columns
    df["duration_sec"] = (df["duration_ms"] // 1000).astype(int)
    df["val_bin"] = (df["valence"] >= 0.5).astype(int)

    # Keep only the columns we care about downstream
    keep = [
        "track_id",
        "track_name",
        "artist_name",
        "genre",
        "duration_sec",
        "valence",
        "val_bin",
        "energy",
        "popularity",
        "key",
        "tempo",
    ]
    df = df[keep]

    if verbose:
        print(
            f"After filtering: {len(df)} tracks "
            f"(dur: {dur_min}-{dur_max} s, pop ≥ {pop_min}, "
            f"genres: {'ANY' if not allowed_genres else allowed_genres})"
        )

    return df


def export_jsonl(df: pd.DataFrame, dst: pathlib.Path, verbose: bool):
    """Write DataFrame rows as JSONL."""
    with dst.open("w", encoding="utf‑8") as f:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), f)
            f.write("\n")

    if verbose:
        print(f"Wrote {len(df)} lines to {dst}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter and convert tracks CSV to .jsonl for SMT solver"
    )
    p.add_argument("src_csv", type=pathlib.Path, help="Input CSV file")
    p.add_argument("dst_jsonl", type=pathlib.Path, help="Output .jsonl file")

    # Duration in *seconds* (more intuitive than ms).
    p.add_argument("--duration-min", type=int, default=120, help="min track length (s)")
    p.add_argument("--duration-max", type=int, default=300, help="max track length (s)")

    p.add_argument("--pop-min", type=int, default=50, help="min popularity (0‑100)")

    p.add_argument(
        "--allowed-genres",
        type=comma_split,
        default="",
        help="comma‑separated whitelist (empty = allow all)",
    )

    p.add_argument(
        "-v", "--verbose", action="store_true", help="print filter summary"
    )

    return p.parse_args(argv)


def main(argv: List[str] | None = None):
    args = parse_args(argv or sys.argv[1:])

    df = load_and_filter(
        csv_path=args.src_csv,
        dur_min=args.duration_min,
        dur_max=args.duration_max,
        pop_min=args.pop_min,
        allowed_genres=args.allowed_genres,
        verbose=args.verbose,
    )

    export_jsonl(df, args.dst_jsonl, verbose=args.verbose)


if __name__ == "__main__":
    main()
