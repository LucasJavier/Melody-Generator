"""
Ejecutamos por lotes para melody_ea.py -> evolve (con registro de la mejor puntuación)

Genera múltiples melodías en diferentes modelos, modos, longitudes de compás, tamaños de población,
recuentos de generación y pares de pesos (w_rules, w_mlp), guardando todos los MIDI en ./out
Y añadiendo la mejor aptitud encontrada a un registro CSV.

Notas:
- Asume por defecto:
  * 6 archivos de modelo en ./models :
      mlp_easy_1N.pkl, mlp_easy_2N.pkl, mlp_easy_3N.pkl,
      mlp_hard_1N.pkl, mlp_hard_2N.pkl, mlp_hard_3N.pkl
  * La carpeta de salida es ./out (creada si no existe)
  * Para 8 barras: pop=60, gens=90  
    Para 16 barras: pop=100, gens=150
  * Ejecuta 6 combinaciones de pesos por configuración: (1,0,0,0), (0,8,0,2), ..., (0,0,1,0)
  * Registra la mejor aptitud por ejecución en CSV en ./out/evolve_scores.csv
  * Opcionalmente, escribe la salida estándar completa por ejecución en ./out/logs/<mismo_nombre>.log

- Ejecuciones totales: 6 modelos × 4 configuraciones de modo/barra × 6 pares de pesos = 144 MIDI

Interruptores:
- VERBOSE: añade --verbose a cada llamada evolve.
- OVERWRITE: cuando es False, omite las ejecuciones cuyo MIDI ya existe.
- DRY_RUN: solo imprime comandos, no los ejecuta.
- SAVE_STDOUT_LOGS: cuando es True, guarda los registros por ejecución en ./out/logs/.
- GEN_8 se puede establecer en 100 si prefieres 100 generaciones para 8 compases.
"""

from __future__ import annotations
import sys
import csv
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import List, Tuple, Optional

# ----------------------- Knobs ajustables -----------------------
MELODY_EA_PATH = Path("melody_ea.py")   # ruta melody_ea.py
MODELS_DIR     = Path("models")         # carpeta que contiene los archivos .pkl 
OUT_DIR        = Path("out")            # carpeta de salida para los .mid
LOG_DIR        = OUT_DIR / "logs"        # registros stdout por ejecucion (optional)
SCORE_CSV_PATH = OUT_DIR / "evolve_scores.csv"

MODEL_FILES = [
    "mlp_easy_1N.pkl",
    "mlp_easy_2N.pkl",
    "mlp_easy_3N.pkl",
    "mlp_hard_1N.pkl",
    "mlp_hard_2N.pkl",
    "mlp_hard_3N.pkl",
]

# Pares de pesos (w_rules, w_mlp)
WEIGHT_SETS: List[Tuple[float, float]] = [
    (1.0, 0.0),
    (0.8, 0.2),
    (0.6, 0.4),
    (0.4, 0.6),
    (0.2, 0.8),
    (0.0, 1.0),
]

# Configuraciones (mode, bars) -> (pop_size, generations)
GEN_8  = 90
GEN_16 = 150
CONFIGS = {
    ("minor", 8):  (60, GEN_8),
    ("minor", 16): (100, GEN_16),
    ("major", 8):  (60, GEN_8),
    ("major", 16): (100, GEN_16),
}

# Flags
SEED      = 42        # Seed para usar o None
VERBOSE   = True      # Para agregar logs
OVERWRITE = False     # Omitir archivos si existen
DRY_RUN   = False     # Imprimir comando ejecutando o no
SAVE_STDOUT_LOGS = True  # Escribe ./out/logs/<midi_name>.log con la salida completa
# -----------------------------------------------------------------

def sanitize_float(f: float) -> str:
    """Crea una cadena corta para nombres de archivo, para números flotantes, por ejemplo, 0,8 -> '0p8'"""
    s = f"{f:.1f}"
    return s.replace("-", "m").replace(".", "p")


def build_out_name(model: Path, mode: str, bars: int, pop: int, gens: int, wr: float, wm: float) -> Path:
    stem = model.stem  # ej, mlp_easy_1N
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
    Analiza la salida de evolve().
    Primario: buscar una línea como: «[EA] Best fitness = 0.9871»
    Alternativa: utilizar la última aparición de líneas como: «[GEN 30] best = 0.8971 | gen_best = ...»
    Devuelve None si no hay coincidencias.
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

    # Fallback: ultimo [GEN ...] best = X
    last_best: Optional[float] = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[GEN ") and " best = " in line:
            try:
                # ej: "[GEN 30] best = 1.0326 | gen_best = 1.0326 | mean = 0.9034"
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
    """Añade una fila al CSV, creando el encabezado si falta."""
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
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_STDOUT_LOGS:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(model_path, out_path, mode, bars, pop, gens, wr, wm)

    print("\n[run] ", " ".join(cmd))
    if DRY_RUN:
        return

    when = datetime.now()

    # Captura stdout/stderr mientras se repite en la consola.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    captured_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured_lines.append(line)
        print(line, end="") 
    proc.wait()

    full_text = "".join(captured_lines)
    best = extract_best_fitness_from_text(full_text)

    # Guarda los logs per-run (optional)
    if SAVE_STDOUT_LOGS:
        log_name = out_path.with_suffix(".log").name
        with (LOG_DIR / log_name).open("w", encoding="utf-8") as flog:
            flog.write(full_text)

    append_score_csv(
        SCORE_CSV_PATH, when, model_path, mode, bars, pop, gens, wr, wm, SEED, out_path, best, proc.returncode
    )

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main():
    if not MELODY_EA_PATH.exists():
        print(f"[error] melody_ea.py not found at: {MELODY_EA_PATH}")
        sys.exit(1)

    # Rutas del modelo
    model_paths = []
    for mf in MODEL_FILES:
        mp = (MODELS_DIR / mf).resolve()
        if not mp.exists():
            print(f"[warn] model not found: {mp}")
        model_paths.append(mp)

    # Producto cartesiano sobre: modelos × [(modo, barras)] × pesos
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
