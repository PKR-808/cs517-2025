import csv, itertools, pathlib, random, subprocess, time

JSONL   = "tracks_filtered.jsonl"
OUT_DIR = pathlib.Path("bench_out_gridall")
OUT_DIR.mkdir(exist_ok=True)

# Parameter grid
# TOTAL_TIME = [600, 1000, 1400, 1800, 2000, 3000]
# N_TRACKS   = [60, 100, 1000, 5000, 10000, 1000000]
# MAX_SLOTS  = [4, 8, 10, 12, 16, 20]
# VAL_DELTA  = [0.05, 0.10, 0.20,  0.25, 0.30, 0.40]
# ENG_DELTA  = [0.05, 0.10, 0.20,  0.25, 0.30, 0.40]
# POP_AVG    = [40, 45, 50, 60, 70, 80]
# TOLERANCE  = [30, 60, 100, 200, 500]
TOTAL_TIME = [1000]
N_TRACKS   = [1000]
MAX_SLOTS  = [5]
VAL_DELTA  = [0.20]
ENG_DELTA  = [0.20]
POP_AVG    = [50]
TOLERANCE  = [100]
RAND_SEEDS = [1, 42, 101, 307, 523, 659]
REPETITIONS = 1             # median of repeats

rows = []
for t,n,k,d,e,p,l,s in itertools.product(TOTAL_TIME, N_TRACKS, MAX_SLOTS, VAL_DELTA, ENG_DELTA, POP_AVG, TOLERANCE, RAND_SEEDS):
    for r in range(REPETITIONS):
        cmd = [
            "python", "playlist_smt.py", JSONL,
            "--total", str(t),
            "--sample-n", str(n),
            "--max-slots", str(k),
            "--val-delta", str(d),
            "--energy-delta", str(e),
            "--pop-avg", str(p),
            "--tolerance", str(l),
            "--seed", str(s),
            "--out", OUT_DIR / f"tmp_t{t}_n{n}_k{k}_d{d}_e{d}_p{p}_l{l}_s{s}.json",
        ]
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        t1 = time.perf_counter()

        rows.append({
            "time":t,
            "n": n,
            "slots": k,
            "val_delta": d,
            "eng_delta":e,
            "pop_avg": p,
            "tolerance":l,
            "seed":s,
            "returncode": res.returncode,
            "time_sec": round(t1 - t0, 3),
        })
        print(rows[-1])

# Save CSV
with open("bench_seedvssetup_final.csv", "w", newline="") as f:
    csv.DictWriter(f, fieldnames=rows[0].keys()).writeheader()
    csv.DictWriter(f, fieldnames=rows[0].keys()).writerows(rows)
print("✔ wrote bench_seedvssetup_final.csv")
