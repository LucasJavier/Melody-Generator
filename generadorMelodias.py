"""
Batch runner for melody_ea.py -> evolve (with best-score logging)

Generates multiple melodies across models, modes, bar lengths, population sizes,
generation counts, and (w_rules, w_mlp) weight pairs, saving all MIDIs into ./out
AND appending the best fitness found to a CSV log.

Usage (from the same folder where melody_ea.py and the models live):
    python run_batch_evolve.py

Notes:
- Assumes by default:
  * 6 model files in ./models :
      mlp_easy_1N.pkl, mlp_easy_2N.pkl, mlp_easy_3N.pkl,
      mlp_hard_1N.pkl, mlp_hard_2N.pkl, mlp_hard_3N.pkl
  * Output folder is ./out (created if missing)
  * For 8 bars: pop=60, gens=90  
    For 16 bars: pop=100, gens=150
  * Runs 6 weight mixes per configuration: (1.0,0.0), (0.8,0.2), ..., (0.0,1.0)
  * Logs best fitness per run to CSV at ./out/evolve_scores.csv
  * Optionally writes full stdout per run to ./out/logs/<same_name>.log

- Total runs: 6 models × 4 mode/bar configs × 6 weight pairs = 144 MIDIs

Switches:
- VERBOSE: append --verbose to each evolve call.
- OVERWRITE: when False, skip runs whose MIDI already exists.
- DRY_RUN: print commands only, do not execute.
- SAVE_STDOUT_LOGS: when True, save per-run logs to ./out/logs/.
- GEN_8 can be set to 100 if you prefer 100 generations for 8 bars.
"""
from __future__ import annotations
import sys
import csv
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import List, Tuple, Optional

# ----------------------- User-tunable knobs -----------------------
MELODY_EA_PATH = Path("melody_ea.py")   # path to melody_ea.py
MODELS_DIR     = Path("models")         # folder containing the .pkl files
OUT_DIR        = Path("out")            # output folder for .mid
LOG_DIR        = OUT_DIR / "logs"        # per-run stdout logs (optional)
SCORE_CSV_PATH = OUT_DIR / "evolve_scores.csv"

MODEL_FILES = [
    "mlp_easy_1N.pkl",
    "mlp_easy_2N.pkl",
    "mlp_easy_3N.pkl",
    "mlp_hard_1N.pkl",
    "mlp_hard_2N.pkl",
    "mlp_hard_3N.pkl",
]

# Weight pairs (w_rules, w_mlp)
WEIGHT_SETS: List[Tuple[float, float]] = [
    (1.0, 0.0),
    (0.8, 0.2),
    (0.6, 0.4),
    (0.4, 0.6),
    (0.2, 0.8),
    (0.0, 1.0),
]

# Configs by (mode, bars) -> (pop_size, generations)
GEN_8  = 90
GEN_16 = 150
CONFIGS = {
    ("minor", 8):  (60, GEN_8),
    ("minor", 16): (100, GEN_16),
    ("major", 8):  (60, GEN_8),
    ("major", 16): (100, GEN_16),
}

# Execution flags
SEED      = 42        # or None to use default from melody_ea
VERBOSE   = True      # add --verbose to each run
OVERWRITE = False     # skip existing output files if False
DRY_RUN   = False     # only print commands without executing
SAVE_STDOUT_LOGS = True  # write ./out/logs/<midi_name>.log with the full stdout

# -----------------------------------------------------------------

def sanitize_float(f: float) -> str:
    """Make a filename-safe short string for floats, e.g., 0.8 -> '0p8'."""
    s = f"{f:.1f}"
    return s.replace("-", "m").replace(".", "p")


def build_out_name(model: Path, mode: str, bars: int, pop: int, gens: int, wr: float, wm: float) -> Path:
    stem = model.stem  # e.g., mlp_easy_1N
    wr_s = sanitize_float(wr)
    wm_s = sanitize_float(wm)
    fname = f"{stem}__{mode}_{bars}bars__pop{pop}_gen{gens}__wr{wr_s}_wm{wm_s}.mid"
    return OUT_DIR / fname


def build_cmd(model_path: Path, out_path: Path, mode: str, bars: int, pop: int, gens: int, wr: float, wm: float) -> List[str]:
    cmd = [
        sys.executable, "-u", str(MELODY_EA_PATH), "evolve",
        "--out-midi", str(out_path),
        "--mode", mode,
        "--bars", str(bars),
        "--pop-size", str(pop),
        "--generations", str(gens),
        "--model", str(model_path),
        "--w-rules", str(wr),
        "--w-mlp", str(wm),
    ]
    if SEED is not None:
        cmd += ["--seed", str(SEED)]
    if VERBOSE:
        cmd += ["--verbose"]
    return cmd


def extract_best_fitness_from_text(text: str) -> Optional[float]:
    """
    Parse the evolve() output.
    Primary: look for a line like: "[EA] Best fitness = 1.0326"
    Fallback: use the last occurrence of lines like: "[GEN 30] best = 1.0326 | gen_best = ..."
    Return None if nothing matches.
    """
    best_val: Optional[float] = None

    # Primary pattern
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[EA] Best fitness ="):
            try:
                best_val = float(line.split("=")[-1].strip())
            except Exception:
                pass

    if best_val is not None:
        return best_val

    # Fallback: seek last [GEN ...] best = X
    last_best: Optional[float] = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[GEN ") and " best = " in line:
            try:
                # e.g. "[GEN 30] best = 1.0326 | gen_best = 1.0326 | mean = 0.9034"
                after = line.split(" best = ", 1)[1]
                num = after.split()[0]
                last_best = float(num)
            except Exception:
                pass
    return last_best


def append_score_csv(csv_path: Path,
                     when: datetime,
                     model_path: Path,
                     mode: str,
                     bars: int,
                     pop: int,
                     gens: int,
                     wr: float,
                     wm: float,
                     seed: Optional[int],
                     out_midi: Path,
                     best_fitness: Optional[float],
                     exit_code: int) -> None:
    """Append a row to the CSV, creating header if missing."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow([
                "timestamp", "model", "mode", "bars", "pop", "generations",
                "w_rules", "w_mlp", "seed", "out_midi", "best_fitness", "exit_code"
            ])
        w.writerow([
            when.isoformat(timespec="seconds"),
            model_path.name,
            mode,
            bars,
            pop,
            gens,
            f"{wr:.3f}",
            f"{wm:.3f}",
            seed if seed is not None else "",
            str(out_midi),
            f"{best_fitness:.6f}" if best_fitness is not None else "",
            exit_code,
        ])


def run_evolve(model_path: Path, mode: str, bars: int, pop: int, gens: int, wr: float, wm: float):
    out_path = build_out_name(model_path, mode, bars, pop, gens, wr, wm)

    if not OVERWRITE and out_path.exists():
        print(f"[skip] exists: {out_path}")
        # Still log a row marking skipped? Not necessary; skip silently here.
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_STDOUT_LOGS:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(model_path, out_path, mode, bars, pop, gens, wr, wm)

    print("\n[run] ", " ".join(cmd))
    if DRY_RUN:
        return

    when = datetime.now()

    # Capture stdout/stderr while also echoing to our console
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    captured_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured_lines.append(line)
        print(line, end="")  # echo live
    proc.wait()

    full_text = "".join(captured_lines)
    best = extract_best_fitness_from_text(full_text)

    # Save per-run stdout log (optional)
    if SAVE_STDOUT_LOGS:
        log_name = out_path.with_suffix(".log").name
        with (LOG_DIR / log_name).open("w", encoding="utf-8") as flog:
            flog.write(full_text)

    # Append to CSV
    append_score_csv(
        SCORE_CSV_PATH, when, model_path, mode, bars, pop, gens, wr, wm, SEED, out_path, best, proc.returncode
    )

    # If the subprocess failed, raise after logging so the batch continues gracefully outside
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main():
    if not MELODY_EA_PATH.exists():
        print(f"[error] melody_ea.py not found at: {MELODY_EA_PATH}")
        sys.exit(1)

    # Resolve model paths
    model_paths = []
    for mf in MODEL_FILES:
        mp = (MODELS_DIR / mf).resolve()
        if not mp.exists():
            print(f"[warn] model not found: {mp}")
        model_paths.append(mp)

    # Cartesian product over: models × [(mode,bars)] × weights
    for model_path in model_paths:
        if not model_path.exists():
            print(f"[skip-model] missing: {model_path}")
            continue

        for (mode, bars), (pop, gens) in CONFIGS.items():
            for wr, wm in WEIGHT_SETS:
                try:
                    run_evolve(model_path, mode, bars, pop, gens, wr, wm)
                except subprocess.CalledProcessError as e:
                    print(f"[fail] {model_path.name} {mode} {bars}bars wr={wr} wm={wm}: exit={e.returncode}")
                except Exception as ex:
                    print(f"[fail] {model_path.name} {mode} {bars}bars wr={wr} wm={wm}: {ex}")


if __name__ == "__main__":
    main()
