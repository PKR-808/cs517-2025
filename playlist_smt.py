#!/usr/bin/env python3
"""
playlist_smt.py  –  SAT/SMT optimisation model for emotion‑smooth playlists
────────────────────────────────────────────────────────────────────────────
Requires `z3-solver` 4.8+ and the JSONL produced by prep_tracks.py

Example
-------
python playlist_smt.py tracks_filtered.jsonl \
    --total 1800 \
    --val-delta 0.10 \
    --energy-delta 0.15 \
    --pop-avg 50 \
    --genre-run 3 \
    --max-slots 20 \
    --tolerance 5 \
    --seed 123 \
    --out solution.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from typing import Dict, List
from time import time

from z3 import (
    Bool,
    If,
    Optimize,
    Real,
    Sum,
    sat,
    set_param,
)

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def comma_split(arg: str) -> List[str]:
    return [s.strip() for s in arg.split(",") if s.strip()]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build & solve the playlist optimisation SMT model"
    )
    p.add_argument("jsonl", type=pathlib.Path, help="input track list (.jsonl)")
    p.add_argument("--total", type=int, default=1800, help="target length (s)")
    p.add_argument("--tolerance", type=int, default=5, help="± seconds tolerance")
    p.add_argument(
        "--val-delta", type=float, default=0.10, help="max jump in valence (0‑1)"
    )
    p.add_argument(
        "--energy-delta",
        type=float,
        default=0.15,
        help="max jump in energy (0‑1)",
    )
    p.add_argument("--pop-avg", type=float, default=0.0, help="avg popularity ≥ value")
    p.add_argument(
        "--genre-run",
        type=int,
        default=0,
        help="max consecutive tracks of same genre (0 = off)",
    )
    p.add_argument(
        "--max-slots",
        type=int,
        default=20,
        help="upper bound on playlist length (<= 25 keeps model tractable)",
    )
    p.add_argument(
        "--sample-n",
        type=int,
        default=0,
        help="randomly sample at most N tracks before building model (0 = keep all)",
    )
    p.add_argument("--seed", type=int, default=None, help="rng seed")
    p.add_argument("--out", type=pathlib.Path, default="solution.json")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────
def load_tracks(path: pathlib.Path) -> List[Dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


# ──────────────────────────────────────────────────────────────
# Z3 helpers
# ──────────────────────────────────────────────────────────────
def exactly_one(bools):
    """Return Z3 constraint enforcing exactly one of the given Bool vars."""
    return Sum([If(b, 1, 0) for b in bools]) == 1


def at_most_one(bools):
    """Pairwise at‑most‑one (sufficient for our sizes)."""
    c = []
    for i in range(len(bools)):
        for j in range(i + 1, len(bools)):
            c.append(If(bools[i] & bools[j], 0, 1) == 1)
    return c


# ──────────────────────────────────────────────────────────────
# Main solver construction
# ──────────────────────────────────────────────────────────────
def build_and_solve(
    tracks: List[Dict],
    total_s: int,
    tol_s: int,
    val_delta: float,
    energy_delta: float,
    pop_avg: float,
    genre_run: int,
    max_slots: int,
    verbose: bool,
):
    n = len(tracks)
    if verbose:
        print(f"Building model with {n} tracks, {max_slots} slots…")

    opt = Optimize()
    opt.set("timeout", 60_000)          # 60 s safety net


    set_param("parallel.enable", True)

    # Decision vars: x[i][j] == True  ⇔  track i is placed at slot j
    x = [[Bool(f"x_{i}_{j}") for j in range(max_slots)] for i in range(n)]

    # ------------------------------------------------------------------
    # 1.  Exactly one track per slot
    # ------------------------------------------------------------------
    for j in range(max_slots):
        opt.add(exactly_one([x[i][j] for i in range(n)]))

    # ------------------------------------------------------------------
    # 2.  Each track used at most once
    # ------------------------------------------------------------------
    for i in range(n):
        opt.add(Sum([If(x[i][j], 1, 0) for j in range(max_slots)]) <= 1)

    # ------------------------------------------------------------------
    # 3.  Total duration within tolerance
    # ------------------------------------------------------------------
    total_duration = Sum(
        [
            tracks[i]["duration_sec"] * If(x[i][j], 1, 0)
            for i in range(n)
            for j in range(max_slots)
        ]
    )
    opt.add(total_duration >= total_s - tol_s)
    opt.add(total_duration <= total_s + tol_s)

    # ------------------------------------------------------------------
    # 4.  Adjacent valence & energy jump constraints
    # ------------------------------------------------------------------
    valence_j = [
        Real(f"val_{j}") for j in range(max_slots)
    ]  # helper Reals for linearity
    energy_j = [Real(f"eng_{j}") for j in range(max_slots)]

    for j in range(max_slots):
        # val_j == Σ_i val_i * x_ij
        opt.add(
            valence_j[j]
            == Sum([tracks[i]["valence"] * If(x[i][j], 1, 0) for i in range(n)])
        )
        opt.add(
            energy_j[j]
            == Sum([tracks[i]["energy"] * If(x[i][j], 1, 0) for i in range(n)])
        )

    # Absolute jump vars
    zv = [Real(f"zv_{j}") for j in range(max_slots - 1)]
    ze = [Real(f"ze_{j}") for j in range(max_slots - 1)]

    for j in range(max_slots - 1):
        # |val_{j+1} - val_j| ≤ δ
        opt.add(zv[j] >= valence_j[j + 1] - valence_j[j])
        opt.add(zv[j] >= valence_j[j] - valence_j[j + 1])
        opt.add(zv[j] <= val_delta)

        opt.add(ze[j] >= energy_j[j + 1] - energy_j[j])
        opt.add(ze[j] >= energy_j[j] - energy_j[j + 1])
        opt.add(ze[j] <= energy_delta)

    # ------------------------------------------------------------------
    # 5.  Average popularity (optional)
    # ------------------------------------------------------------------
    if pop_avg > 0:
        avg_popularity = (
            Sum(
                [
                    tracks[i]["popularity"] * If(x[i][j], 1, 0)
                    for i in range(n)
                    for j in range(max_slots)
                ]
            )
            / max_slots
        )
        opt.add(avg_popularity >= pop_avg)

    # ------------------------------------------------------------------
    # 6.  Genre run‑length constraint (optional)
    # ------------------------------------------------------------------
    if genre_run > 0:
        genres = list({t["genre"] for t in tracks})
        for g in genres:
            ids_g = [i for i, t in enumerate(tracks) if t["genre"] == g]
            if not ids_g:
                continue
            for j in range(max_slots - genre_run):
                window = [
                    x[i][j + k] for i in ids_g for k in range(genre_run + 1)
                ]
                opt.add(Sum([If(b, 1, 0) for b in window]) <= genre_run)

    # ------------------------------------------------------------------
    # 7.  Objective: minimise sum |Δval| + |Δenergy|
    # ------------------------------------------------------------------
    roughness = Sum(zv + ze)
    # (Optional) soft‑maximize popularity: uncomment to combine:
    #   avg_pop_var = Sum([ tracks[i]["popularity"] * If(x[i][j],1,0)
    #                       for i in range(n)
    #                       for j in range(max_slots)])  / max_slots
    #   opt.minimize(roughness - 0.01 * avg_pop_var)
    opt.minimize(roughness)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
        # Solve  (heartbeat + timeout)
    # ------------------------------------------------------------------
    # print("⏳  calling Z3 …", flush=True)
    # start = time.perf_counter()

    # check = opt.check()
    # elapsed = time.perf_counter() - start

    # print(f"✔  Z3 returned {check} in {elapsed:,.2f} s")
    if opt.check() != sat:
        return None  # unsat

    m = opt.model()
    playlist = []
    cur_time = 0
    for j in range(max_slots):
        trk_idx = next(i for i in range(n) if m[x[i][j]])
        t = tracks[trk_idx]
        playlist.append(
            {
                "slot": j + 1,
                "track_id": t["track_id"],
                "track_name": t["track_name"],
                "artist_name": t["artist_name"],
                "genre": t["genre"],
                "start_sec": cur_time,
                "duration_sec": t["duration_sec"],
                "valence": t["valence"],
                "energy": t["energy"],
                "popularity": t["popularity"],
            }
        )
        cur_time += t["duration_sec"]

    result = {
        "playlist": playlist,
        "total_duration_sec": cur_time,
        "objective_val": m.evaluate(roughness).as_decimal(6),
    }
    return result


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
def main(argv=None):
    args = parse_args(argv)

    tracks = load_tracks(args.jsonl)

    if args.sample_n and len(tracks) > args.sample_n:
        rng = random.Random(args.seed)
        tracks = rng.sample(tracks, args.sample_n)

    res = build_and_solve(
        tracks=tracks,
        total_s=args.total,
        tol_s=args.tolerance,
        val_delta=args.val_delta,
        energy_delta=args.energy_delta,
        pop_avg=args.pop_avg,
        genre_run=args.genre_run,
        max_slots=args.max_slots,
        verbose=args.verbose,
    )

    if res is None:
        print("❌  No feasible playlist found under the given parameters.")
        sys.exit(1)

    with args.out.open("w") as f:
        json.dump(res, f, indent=2)

    print(f"✅  Playlist written to {args.out} (obj={res['objective_val']})")


if __name__ == "__main__":
    main()