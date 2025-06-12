import csv, itertools, pathlib, random, subprocess, time

JSONL   = "tracks_filtered.jsonl"
OUT_DIR = pathlib.Path("bench_out")
OUT_DIR.mkdir(exist_ok=True)

# Parameter grid
TOTAL_TIME = [600, 1000, 1400, 2000, 3000]
N_TRACKS   = [60, 80, 100, 500, 1000]
VAL_DELTA  = [0.05, 0.10, 0.20, 0.30, 0.40]
MAX_SLOTS  = [4, 8, 12, 16, 20]
REPETITIONS = 2              # median of repeats

rows = []
for t,n,k,d in itertools.product(TOTAL_TIME, N_TRACKS, MAX_SLOTS, VAL_DELTA):
    for r in range(REPETITIONS):
        cmd = [
            "python", "playlist_smt.py", JSONL,
            "--total", str(t),
            "--sample-n", str(n),
            "--max-slots", str(k),
            "--val-delta", str(d),
            "--energy-delta", "0.30",
            "--pop-avg", "50",
            "--tolerance", "100",
            "--out", OUT_DIR / f"tmp_t{t}_n{n}_k{k}_d{d}.json",
        ]
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        t1 = time.perf_counter()

        rows.append({
            "time":t,
            "n": n,
            "slots": k,
            "val_delta": d,
            "returncode": res.returncode,
            "time_sec": round(t1 - t0, 3),
        })
        print(rows[-1])

# Save CSV
with open("bench_1.csv", "w", newline="") as f:
    csv.DictWriter(f, fieldnames=rows[0].keys()).writeheader()
    csv.DictWriter(f, fieldnames=rows[0].keys()).writerows(rows)
print("✔ wrote bench_1.csv")
