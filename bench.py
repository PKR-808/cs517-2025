import csv, itertools, pathlib, random, subprocess, time

JSONL   = "tracks_filtered.jsonl"
OUT_DIR = pathlib.Path("bench_out")
OUT_DIR.mkdir(exist_ok=True)

# Parameter grid
N_TRACKS   = [40, 60, 80, 100]
VAL_DELTA  = [0.05, 0.10, 0.15]
MAX_SLOTS  = [12, 16]
REPETITIONS = 2              # median of repeats

rows = []
for n,k,d in itertools.product(N_TRACKS, MAX_SLOTS, VAL_DELTA):
    for r in range(REPETITIONS):
        cmd = [
            "python", "playlist_smt.py", JSONL,
            "--sample-n", str(n),
            "--max-slots", str(k),
            "--val-delta", str(d),
            "--energy-delta", "0.15",
            "--pop-avg", "40",
            "--tolerance", "5",
            "--out", OUT_DIR / "tmp.json",
        ]
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        t1 = time.perf_counter()

        rows.append({
            "n": n,
            "slots": k,
            "val_delta": d,
            "returncode": res.returncode,
            "time_sec": round(t1 - t0, 3),
        })
        print(rows[-1])

# Save CSV
with open("bench.csv", "w", newline="") as f:
    csv.DictWriter(f, fieldnames=rows[0].keys()).writeheader()
    csv.DictWriter(f, fieldnames=rows[0].keys()).writerows(rows)
print("✔ wrote bench.csv")
