"""
melody_ea.py
------------
Base para proyecto de generación de melodías con Algoritmo Evolutivo + evaluación por reglas y MLP.

Requisitos (instalar con pip):
    pip install music21 numpy scikit-learn joblib

Qué contiene:
A) Representación del individuo (genotipo por grados diatónicos + octava, en Do mayor/menor) y fenotipado a eventos/MIDI.
B) Extracción de *features* musicales (las mismas que venías usando) y pipeline para entrenar una MLP con dataset de MIDIs.
   Incluye funciones para crear versiones "con ruido" (negativas) a partir de melodías buenas.

Uso CLI (ejemplos):
-------------------
1) Entrenar MLP con una carpeta de MIDIs buenos y generar negativos por data augmentation:
    python melody_ea.py --train --good-dir ./midis_buenos --out-model mlp.pkl --negatives-per-good 3

2) Analizar un MIDI y calcular fitness por reglas y (opcional) score MLP:
    python melody_ea.py --analyze ./ejemplo.mid --model mlp.pkl

3) Generar un individuo aleatorio de 8 compases y guardarlo a MIDI:
    python melody_ea.py --gen-random --out-midi random.mid --mode major --qpm 96

"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter, defaultdict
import argparse
import json
import math
import random
import pathlib

import numpy as np

# music21
from music21 import converter, stream, note, chord, meter, key as m21key, tempo, interval, tie

# opcionales (para entrenar)
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GroupShuffleSplit
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================
#  A) REPRESENTACIÓN GENÓMICA
# ============================

# Tablas discretas de duración (en quarterLength) y velocidad
DUR_TABLE = [0.25, 0.5, 1.0, 2.0]      # semi-corchea, corchea, negra, blanca
VEL_TABLE = [40, 64, 85, 105]    # p, mp, mf, f

# Pitch-classes diatónicos para Do mayor / menor natural
PC_MAJOR = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B
PC_MINOR = [0, 2, 3, 5, 7, 8, 10]  # C D Eb F G Ab Bb (natural)

# Que escala usar (Mayor o menor)
def pcs_for_mode(mode: str) -> List[int]:
    return PC_MINOR if str(mode).lower().startswith('min') else PC_MAJOR

@dataclass
class Gene:
    """Un evento genético mínimo (monofónico)."""
    degree: int           # 0..6 (grado diatónico en C mayor/menor) 
    octShift: int         # -2..+2 (ajuste de octava relativo a una base) (ver cuanto para tesitura)
    durIdx: int           # índice a DUR_TABLE
    isRest: bool          # silencio
    velIdx: Optional[int] = None  # índice a VEL_TABLE (None -> default 64)
    # Campos expresivos (se habilitan cuando allow_expressive=True a partir de un umbral)
    tieFlag: Optional[str] = None            # 'start'|'stop'|'continue'|None
    articulation: Optional[str] = None       # 'staccato'|'tenuto'|'accent'|None

@dataclass
class Genome:
    """Individuo genético (genotipo)."""
    genes: List[Gene]
    mode: str = 'major'          # 'major'|'minor'
    time_signature: str = '4/4'
    qpm: int = 90 # bpm
    base_octave: int = 4         # C4 como centro
    tessitura_low: int = 60      # C4
    tessitura_high: int = 72     # C5
    allow_expressive: bool = False  # si True, respeta tie/articulations que traiga el gen

# ---------------- Fenotipado: genotipo -> notas/REST con compases, beat y ties correctos ---------------

@dataclass
class Event:
    pitch: Optional[int]    # MIDI 0..127, o None si es silencio
    duration_qL: float
    velocity: Optional[int]
    beat: float
    measure: int
    is_rest: bool
    articulations: List[str]
    tie: Optional[str]

@dataclass
class Individual:
    metadata: Dict[str, Any]
    events: List[Event]

# Convierte una escala en valor/es absoluto en MIDI
def degree_to_midi(degree: int, octShift: int, mode: str, base_oct: int = 4) -> int:
    pcs = pcs_for_mode(mode)
    pc = pcs[int(degree) % 7] # [0,6]
    midi = pc + 12 * (base_oct + int(octShift)) # Coloca en la octava correspondiente
    return int(midi)

# Clampea al rango de tesitura min/max
def clamp_tessitura(midi: int, lo: int, hi: int) -> int:
    if midi is None: 
        return midi
    while midi < lo: midi += 12
    while midi > hi: midi -= 12
    return midi

# De un genoma (genotipo: Genome) a un fenoma (fenotipo: Individual)
def phenotype_from_genome(g: Genome) -> Individual:
    """Convierte el genoma en una secuencia de eventos con compases válidos en la métrica especificada.
       Si un evento cruza el borde del compás, se corta y se agregan ties automáticamente."""
    ts = meter.TimeSignature(g.time_signature)
    bar_len = float(ts.barDuration.quarterLength)  # p.ej. 4.0 en 4/4
    events: List[Event] = []
    cur_in_bar = 0.0
    measure_num = 1

    for gene in g.genes:
        # Duración solicitada
        dur_total = float(DUR_TABLE[gene.durIdx])
        remaining = dur_total

        # Pitch + dinámica
        if not gene.isRest:
            midi = degree_to_midi(gene.degree, gene.octShift, g.mode, g.base_octave)
            midi = clamp_tessitura(midi, g.tessitura_low, g.tessitura_high)
        else:
            midi = None

        vel = VEL_TABLE[gene.velIdx] if (gene.velIdx is not None and 0 <= gene.velIdx < len(VEL_TABLE)) else 64

        first_chunk = True
        while remaining > 1e-9:
            space = bar_len - cur_in_bar
            use = min(remaining, space)
            beat_pos = 1.0 + cur_in_bar

            # tie automático si el evento se corta en compás siguiente
            tie_flag = None
            if not gene.isRest and use < remaining:
                tie_flag = 'start' if first_chunk else 'continue'
            elif not gene.isRest and not first_chunk and use >= remaining:
                tie_flag = 'stop'  # cierra la ligadura

            # si allow_expressive, respetar flags del gen (prioridad al tie automático para cortes)
            final_tie = tie_flag if tie_flag is not None else (gene.tieFlag if g.allow_expressive else None)
            arts = [gene.articulation] if (g.allow_expressive and gene.articulation) else []

            events.append(Event(
                pitch=None if gene.isRest else midi,
                duration_qL=use,
                velocity=None if gene.isRest else vel,
                beat=beat_pos,
                measure=measure_num,
                is_rest=gene.isRest,
                articulations=arts,
                tie=final_tie
            ))

            cur_in_bar += use
            remaining -= use
            first_chunk = False

            if abs(cur_in_bar - bar_len) < 1e-9:
                cur_in_bar = 0.0
                measure_num += 1

    meta = {
        "normalizedKey": {"tonic": "C", "mode": g.mode},
        "timeSignature": g.time_signature,
        "qpm": g.qpm
    }
    return Individual(metadata=meta, events=events)

# ---------------------- Construir stream/MIDI desde Individual -----------------------

# De un fenotipo (Individual) a un steam MIDI
def individual_to_stream(ind: Individual) -> stream.Part:
    p = stream.Part()
    ts = ind.metadata.get("timeSignature", "4/4")
    p.insert(0, meter.TimeSignature(ts))
    if ind.metadata.get("qpm"):
        p.insert(0, tempo.MetronomeMark(number=int(ind.metadata["qpm"])))
    off = 0.0
    for e in ind.events:
        el = note.Rest() if (e.is_rest or e.pitch is None) else note.Note(e.pitch)
        el.duration.quarterLength = e.duration_qL
        if not e.is_rest and e.velocity is not None:
            el.volume.velocity = int(e.velocity)
        if e.tie in ('start', 'stop', 'continue'):
            try:
                el.tie = tie.Tie(e.tie)
            except Exception:
                pass
        # articulaciones simples
        for a in e.articulations or []:
            try:
                from music21 import articulations as arts
                if hasattr(arts, a):
                    el.articulations.append(getattr(arts, a)())
            except Exception:
                pass
        p.insert(off, el)
        off += e.duration_qL
    # normalizar notación y compases (mejora cálculo de beatStrength luego)
    try:
        p.makeMeasures(inPlace=True)
        p.makeNotation(inPlace=True)
    except Exception:
        pass
    return p

# Convierte, con funcionx externa, el fenoma en MIDI y lo guarda en la direccion pasada
def save_individual_as_midi(ind: Individual, out_path: str | pathlib.Path):
    part = individual_to_stream(ind)
    s = stream.Score()
    nk = ind.metadata.get("normalizedKey")
    if nk and nk.get("tonic") and nk.get("mode"):
        s.insert(0, m21key.Key(nk["tonic"], nk["mode"]))
    if ind.metadata.get("timeSignature"):
        s.insert(0, meter.TimeSignature(ind.metadata["timeSignature"]))
    if ind.metadata.get("qpm"):
        s.insert(0, tempo.MetronomeMark(number=int(ind.metadata["qpm"])))
    s.insert(0, part)
    s.write("midi", fp=str(out_path))


# ============================
#  B) EXTRACCIÓN DE FEATURES
# ============================

# Metricas

def _hist_norm(values: List[Any]) -> Dict[Any, float]:
    if not values:
        return {}
    c = Counter(values)
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()}

def _tessitura(events: List[Event]) -> Tuple[Optional[int], Optional[int]]:
    pitches = [e.pitch for e in events if e.pitch is not None]
    if not pitches:
        return None, None
    return int(min(pitches)), int(max(pitches))

def _scale_fit(events: List[Event], key_obj: m21key.Key) -> float:
    pcs = {p.pitchClass for p in key_obj.getPitches()}
    notes = [e for e in events if not e.is_rest and e.pitch is not None]
    if not notes:
        return 0.0
    in_scale = sum(((e.pitch % 12) in pcs) for e in notes)
    return in_scale / len(notes)

def _intervals_semitones(events: List[Event]) -> List[int]:
    notes = [e.pitch for e in events if e.pitch is not None]
    return [abs(notes[i+1] - notes[i]) for i in range(len(notes)-1)]

def _step_leap_ratio(intervals: List[int], thr: int = 5) -> float:
    if not intervals:
        return 0.0
    steps = sum(1 for d in intervals if d <= thr)
    return steps / len(intervals)

def _entropy_from_hist(h: Dict[Any, float]) -> float:
    if not h:
        return 0.0
    return -sum(p * math.log(p + 1e-12) for p in h.values())

def _duration_bins(events: List[Event]) -> Dict[float, float]:
    return _hist_norm([e.duration_qL for e in events if e.duration_qL is not None])

def _pitch_class_hist(events: List[Event]) -> Dict[int, float]:
    pcs = [e.pitch % 12 for e in events if e.pitch is not None]
    return _hist_norm(pcs)

def _rest_ratio(events: List[Event]) -> float:
    if not events:
        return 0.0
    rests = sum(1 for e in events if e.is_rest)
    return rests / len(events)

def _meter_alignment_score(events: List[Event], part: stream.Part) -> float:
    strengths = []
    weights = []
    flat = part.flatten()
    # índice aproximado (measure, beat) -> beatStrength
    strength_index = defaultdict(dict)
    for n in flat.notesAndRests:
        mnum = getattr(n, 'measureNumber', 0) or 0
        strength_index[mnum][getattr(n, 'beat', 1.0)] = getattr(n, 'beatStrength', 0.0)
    for e in events:
        s = strength_index.get(e.measure, {}).get(e.beat, 0.0)
        w = (e.velocity if (e.velocity is not None) else 64) if not e.is_rest else 0
        strengths.append(s * w)
        weights.append(w)
    if sum(weights) == 0:
        return 0.0
    return float(sum(strengths) / sum(weights))

def _ngram_repetition_score(sequence: List[int|float], n: int = 3) -> float:
    if len(sequence) < n + 1:
        return 0.0
    grams = [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    c = Counter(grams)
    total = len(grams)
    repeats = sum(freq for _, freq in c.items() if freq > 1)
    return repeats / total

# Obtenemos todas las metricas
def compute_features(ind: Individual, key_obj: m21key.Key) -> Dict[str, Any]:
    ev = ind.events
    intervals = _intervals_semitones(ev)
    d_hist = _duration_bins(ev)
    H_dur = _entropy_from_hist(d_hist)
    H_dur_max = math.log(max(len(d_hist), 1))
    ent_norm = (H_dur / H_dur_max) if H_dur_max > 0 else 0.0
    pc_hist = _pitch_class_hist(ev)
    f_scale = _scale_fit(ev, key_obj)
    step_ratio = _step_leap_ratio(intervals)
    mean_interval = float(np.mean(intervals)) if intervals else 0.0
    rest_r = _rest_ratio(ev)
    low, high = _tessitura(ev)
    range_semi = (high - low) if (low is not None and high is not None) else 0

    int_sign = [np.sign(ind.events[i+1].pitch - ind.events[i].pitch)
                for i in range(len(ind.events)-1)
                if (ind.events[i].pitch is not None and ind.events[i+1].pitch is not None)]
    rep_int3 = _ngram_repetition_score([e for e in intervals if e is not None], n=3)
    rep_contour3 = _ngram_repetition_score([int(x) for x in int_sign], n=3)
    dur_seq = [e.duration_qL for e in ev if not e.is_rest]
    rep_dur3 = _ngram_repetition_score(dur_seq, n=3)

    # reconstruir Part para alignment
    p = stream.Part()
    ts_str = ind.metadata.get('timeSignature') or '4/4'
    p.insert(0, meter.TimeSignature(ts_str))
    off = 0.0
    for e in ev:
        el = note.Rest() if e.is_rest else note.Note(e.pitch)
        el.duration.quarterLength = e.duration_qL
        p.insert(off, el)
        off += e.duration_qL
    try:
        p.makeMeasures(inPlace=True)
        p.makeNotation(inPlace=True)
    except Exception:
        pass
    align = _meter_alignment_score(ev, p)

    return {
        "scale_fit": f_scale,
        "mean_interval_semitones": mean_interval,
        "step_ratio_leq5st": step_ratio,
        "duration_entropy_norm": ent_norm,
        "rest_ratio": rest_r,
        "range_semitones": range_semi,
        "pitch_class_hist": pc_hist,
        "duration_hist": d_hist,
        "interval_hist": _hist_norm(intervals),
        "rep_ngram3_intervals": rep_int3,
        "rep_ngram3_contour": rep_contour3,
        "rep_ngram3_durations": rep_dur3,
        "meter_alignment": align
    }

FEATURE_ORDER = [
    "scale_fit",
    "mean_interval_semitones",
    "step_ratio_leq5st",
    "duration_entropy_norm",
    "rest_ratio",
    "range_semitones",
    "rep_ngram3_intervals",
    "rep_ngram3_contour",
    "rep_ngram3_durations",
    "meter_alignment"
]

def features_to_vector(feat: Dict[str, Any]) -> np.ndarray:
    """Convierte el dict de features en un vector numérico fijo.
       (Histograms se omiten en el vector base; podés añadir divergencias luego)."""
    return np.array([float(feat[k]) for k in FEATURE_ORDER], dtype=np.float32)

# Devuelve el valor de fitness pesado
def rule_based_fitness(feat: Dict[str, Any]) -> float:
    """Escalar sencillo combinando objetivos razonables (ajustá a gusto)."""
    scale_fit = feat["scale_fit"]
    step_ratio = feat["step_ratio_leq5st"]
    align = feat["meter_alignment"]
    ent = feat["duration_entropy_norm"]
    mean_int = feat["mean_interval_semitones"]
    rep_dur = feat["rep_ngram3_durations"]
    rep_cont = feat["rep_ngram3_contour"]
    rest_r = feat["rest_ratio"]
    range_s = feat["range_semitones"]

    score = (
        0.25*scale_fit
      + 0.15*step_ratio
      + 0.15*align
      + 0.15*(1 - abs(ent - 0.6))
      + 0.10*(1 - min(mean_int/6.0, 1.0))
      + 0.10*(1 - abs(rep_dur - 0.6))
      + 0.10*(1 - abs(rep_cont - 0.7))
      - 0.05*rest_r
      - 0.10*max(0.0, (range_s - 12) / 12.0)  # penaliza exceder 1 octava
    )
    return float(score)


# ======================================
#  C) PARSEAR MIDI -> INDIVIDUAL/FEATURES
# ======================================

def _quantize(value: float, grid: float) -> float:
    if grid is None or grid <= 0:
        return value
    return round(value / grid) * grid

def _to_monophonic(part: stream.Part) -> stream.Part:
    flat = part.flatten().notesAndRests.stream()
    mono = stream.Part()
    for el in flat:
        if isinstance(el, chord.Chord):
            n = note.Note(max(el.pitches, key=lambda p: p.midi))
            n.duration = el.duration
            n.offset = el.offset
            if el.volume and el.volume.velocity is not None:
                n.volume.velocity = el.volume.velocity
            for a in el.articulations:
                n.articulations.append(a)
            mono.insert(el.offset, n)
        else:
            if hasattr(el, 'duration') and getattr(el.duration, 'isGrace', False):
                continue
            mono.insert(el.offset, el)
    # reconstruir compases/tempo/ts si existen
    for t in part.recurse().getElementsByClass(tempo.MetronomeMark):
        mono.insert(t.offset, t)
    if part.timeSignature is not None:
        mono.insert(0, part.timeSignature)
    if part.keySignature is not None:
        mono.insert(0, part.keySignature)
    try:
        mono.makeMeasures(inPlace=True)
        mono.makeNotation(inPlace=True)
    except Exception:
        pass
    return mono

def _analyze_key(s: stream.Score | stream.Part) -> Tuple[str, str, m21key.Key]:
    k = s.analyze('key')
    return k.tonic.name, k.mode, k

def _transpose_to_C(part: stream.Part) -> Tuple[stream.Part, m21key.Key]:
    tonic, mode, k = _analyze_key(part)
    target_mode = 'minor' if 'minor' in mode.lower() else 'major'
    target_key = m21key.Key('C', target_mode)
    i = interval.Interval(k.tonic, target_key.tonic)
    transposed = part.transpose(i, inPlace=False)
    return transposed, target_key

def parse_midi_to_individual(midi_path: str | pathlib.Path, quant_grid: float = 0.25) -> Tuple[Individual, m21key.Key, Dict[str, Any]]:
    score = converter.parse(str(midi_path))
    parts = score.parts.stream() if hasattr(score, 'parts') else []
    if len(parts) == 0:
        part = score.flat if hasattr(score, 'flat') else score
    else:
        part = parts[0]
    part = _to_monophonic(part)
    transposed, key_obj = _transpose_to_C(part)

    # recolectar eventos (con beat/measure)
    events: List[Event] = []
    flat = transposed.flatten()
    for el in flat.notesAndRests:
        if hasattr(el, 'duration') and getattr(el.duration, 'isGrace', False):
            continue
        mnum = int(getattr(el, 'measureNumber', 0) or 0)
        b = float(getattr(el, 'beat', 1.0))
        d = _quantize(el.quarterLength, quant_grid)
        if isinstance(el, note.Note):
            vel = el.volume.velocity
            tflag = el.tie.type if el.tie is not None else None
            events.append(Event(
                pitch=el.pitch.midi, duration_qL=d, velocity=vel, beat=b,
                measure=mnum, is_rest=False, articulations=[a.classes[0] for a in getattr(el, 'articulations', [])],
                tie=tflag
            ))
        elif isinstance(el, note.Rest):
            events.append(Event(
                pitch=None, duration_qL=d, velocity=None, beat=b, measure=mnum,
                is_rest=True, articulations=[], tie=None
            ))
    events = [e for e in events if e.duration_qL > 0]
    ts = None
    ts_el = transposed.recurse().getElementsByClass(meter.TimeSignature).first()
    if ts_el:
        ts = ts_el.ratioString
    qpm = None
    mm = transposed.recurse().getElementsByClass(tempo.MetronomeMark).first()
    if mm and mm.number:
        qpm = mm.number

    low, high = _tessitura(events)
    ind = Individual(
        metadata={
            "originalKey": {},  # omitimos para simplicidad
            "normalizedKey": {"tonic": 'C', "mode": key_obj.mode},
            "timeSignature": ts or '4/4',
            "qpm": qpm or 90,
            "tessitura_midi": {"low": low, "high": high},
            "quant_grid_qL": quant_grid,
        },
        events=events
    )
    feats = compute_features(ind, key_obj)
    return ind, key_obj, feats


# ======================================
#  D) DATASET: BUENAS vs CORRUPTAS (0/1)
# ======================================

# Corromper en tono - pitch -
def corrupt_events_offscale(ev: List[Event], mode: str, prob: float = 0.15) -> List[Event]:
    pcs = set(pcs_for_mode(mode))
    out: List[Event] = []
    for e in ev:
        if e.is_rest or e.pitch is None or random.random() > prob: # Vemos, bajo probabilidad, si dejar la nota o modif.
            out.append(e)
            continue
        # empujar a un pc "malo" (no diatónico)
        pc = e.pitch % 12
        candidates = [x for x in range(12) if x not in pcs]
        bad_pc = random.choice(candidates)
        new_pitch = e.pitch - pc + bad_pc
        out.append(Event(new_pitch, e.duration_qL, e.velocity, e.beat, e.measure, False, e.articulations, e.tie))
    return out

# Corromper intervalos
def corrupt_events_large_leaps(ev: List[Event], prob: float = 0.12, max_jump: int = 12) -> List[Event]:
    out: List[Event] = ev.copy()
    note_idxs = [i for i, e in enumerate(ev) if (not e.is_rest and e.pitch is not None)]
    for i in note_idxs:
        if random.random() <= prob: # Bajo probabilidad, para modif. intervalo entre nota
            delta = random.choice([-1, 1]) * random.randint(7, max_jump)  # >= quinta justa
            e = ev[i]
            out[i] = Event(e.pitch + delta, e.duration_qL, e.velocity, e.beat, e.measure, False, e.articulations, e.tie)
    return out

#  Corromper ritmo - duracion -
def corrupt_events_rhythm(ev: List[Event], mode: str = "flat_or_chaos", p_flat: float = 0.5) -> List[Event]:
    # o bien planchar ritmos (todas negras) o hacerlos caóticos
    out: List[Event] = []
    do_flat = (random.random() < p_flat)
    for e in ev:
        if e.is_rest:
            out.append(e)
            continue
        if mode == "flat_or_chaos" and do_flat:
            new_d = 1.0
        else:
            new_d = random.choice(DUR_TABLE)
        out.append(Event(e.pitch, new_d, e.velocity, e.beat, e.measure, False, e.articulations, e.tie))
    return out

# Corrompe silencios
def corrupt_events_add_rests(ev: List[Event], prob: float = 0.10) -> List[Event]:
    out: List[Event] = []
    for e in ev:
        if not e.is_rest and random.random() < prob:
            out.append(Event(None, e.duration_qL, None, e.beat, e.measure, True, [], None))
        else:
            out.append(e)
    return out

# Genera el nuevo fenoma (individual) con ruido
def individual_from_events_like(ind: Individual, new_events: List[Event]) -> Individual:
    meta = dict(ind.metadata)
    return Individual(metadata=meta, events=new_events)

def generate_negative_variants(ind: Individual, key_obj: m21key.Key, n: int = 2) -> List[Individual]:
    """Crea n variantes con 'ruido' (malas) a partir de una melodía buena."""
    negs: List[Individual] = []
    mode = key_obj.mode
    for _ in range(n):
        ev = ind.events
        v = corrupt_events_offscale(ev, mode, prob=0.15)
        v = corrupt_events_large_leaps(v, prob=0.12, max_jump=12)
        v = corrupt_events_rhythm(v, mode="flat_or_chaos", p_flat=0.5)
        v = corrupt_events_add_rests(v, prob=0.10)
        negs.append(individual_from_events_like(ind, v))
    return negs

# Carga los MIDIs con melodias "buenas", las computa, etiqueta con 1, crea nuevas con ruido, las computa, y las etiqueta con 0
def load_dataset_from_dir(good_dir: str | pathlib.Path, negatives_per_good: int = 2,
                        limit: Optional[int] = None, show_progress: bool = False
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Lee todos los .mid/.midi de una carpeta como positivos (label=1) y genera negativos sintéticos (label=0)."""
    X: List[np.ndarray] = []
    y: List[int] = []
    groups: List[str] = []

    p = pathlib.Path(good_dir)
    files = sorted([f for f in p.glob("**/*") if f.suffix.lower() in (".mid", ".midi")])
    if limit is not None:
        files = files[:int(limit)]
    total = len(files)
    if total == 0:
        raise RuntimeError("No se encontraron MIDIs válidos en la carpeta.")
    
    for i, f in enumerate(files):
        if show_progress:
            print(f"[{i}/{total}] Procesando: {f.name}", flush=True)
        try:
            ind, key_obj, feats = parse_midi_to_individual(f)
            X.append(features_to_vector(feats)); y.append(1)
            groups.append(f.stem)
            if show_progress:
                print("   + positivo OK", flush=True)
            # negativos
            neg_inds = generate_negative_variants(ind, key_obj, n=negatives_per_good)
            for nin in neg_inds:
                nfeats = compute_features(nin, key_obj)
                X.append(features_to_vector(nfeats)); y.append(0)
                groups.append(f.stem)
            if show_progress:
                print(f"   + {negatives_per_good} negativos OK  (acum: {len(y)} muestras)", flush=True)
        except Exception as ex:
            print(f"[WARN] Saltando {f}: {ex}")
    if not X:
        raise RuntimeError("No se encontraron MIDIs válidos en la carpeta.")
    return np.vstack(X), np.array(y, dtype=np.int32), groups


# ======================================
#  E) ENTRENAMIENTO DEL MLP (features)
# ======================================

def train_mlp_on_folder(good_dir: str | pathlib.Path, out_model: str | pathlib.Path, 
            negatives_per_good: int = 2, limit: Optional[int] = None, show_progress: bool = False):
    
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn/joblib no están disponibles. Instalá: pip install scikit-learn joblib")
   
    print(f"[INFO] Leyendo dataset desde: {good_dir}", flush=True)
    X, y, groups = load_dataset_from_dir(good_dir, 
                                negatives_per_good=negatives_per_good,
                                limit=limit,
                                show_progress=show_progress)
    print(f"[INFO] Dataset listo. X={X.shape}, y={y.shape} (positivos={int(y.sum())}, negativos={int((y==0).sum())})", flush=True)
  
    # Shuffle
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (tr_idx, te_idx) = next(gss.split(X, y, groups=groups))
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    print(f"[INFO] Split agrupado por archivo: train={len(ytr)} test={len(yte)} (grupos no mezclados)", flush=True)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=400, random_state=42))
    ])

    print("[INFO] Entrenando MLP...", flush=True)
    pipe.fit(Xtr, ytr)
    print("[INFO] Evaluando...", flush=True)
    yhat = pipe.predict(Xte)
    print(classification_report(yte, yhat, digits=3))

    out_model = pathlib.Path(out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_model)
    print(f"[OK] Modelo guardado en: {out_model}")

def score_with_model(feats: Dict[str, Any], model_path: str | pathlib.Path) -> float:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn/joblib no están disponibles. Instalá: pip install scikit-learn joblib")
    clf = joblib.load(model_path)
    x = features_to_vector(feats).reshape(1, -1)
    # Probabilidad de clase 1 (melodía buena)
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(x)[0, 1]
        return float(p)
    # fallback
    pred = clf.predict(x)[0]
    return float(pred)


# ======================================
#  F) UTILIDADES EA: aleator y gating expresivo
# ======================================

# Mutacion ? a nivel de Gen
def random_gene(mode: str) -> Gene:
    return Gene(
        degree=random.randint(0, 6),
        octShift=random.randint(-1, 1),
        durIdx=random.randint(0, len(DUR_TABLE)-1),
        isRest=(random.random() < 0.10),
        velIdx=random.randint(0, len(VEL_TABLE)-1),
        tieFlag=None,
        articulation=None
    )

def random_genome(n_compases: int = 8, mode: str = 'major', qpm: int = 96) -> Genome:
    """Genera un genoma aleatorio que *aproxima* duración total n_compases * 4 negras."""
    target_qL = n_compases * 4.0
    genes: List[Gene] = []
    acc = 0.0
    while acc < target_qL - 1e-9:
        g = random_gene(mode)
        genes.append(g)
        acc += DUR_TABLE[g.durIdx]
    return Genome(
        genes=genes, mode=mode, qpm=qpm,
        tessitura_low=60, tessitura_high=72,
        allow_expressive=False
    )

def enable_expressive_if_threshold(genome: Genome, fitness_value: float, thr: float = 0.65) -> Genome:
    """'Gate' expresivo: si la aptitud supera umbral, habilitá tie/artic en el fenotipado.
       (En EA real, también empezarías a mutar esos campos)."""
    genome.allow_expressive = (fitness_value >= thr)
    return genome

# ======================================
#  G) EVOLUTIONARY ALGORITHM (EA)
# ======================================

@dataclass
class EAConfig:
    pop_size: int = 40
    generations: int = 60
    tournament_k: int = 3
    elitism: int = 2
    crossover_rate: float = 0.9
    mutation_rate: float = 0.25
    # mut probs (se evalúan por gen y/o por genoma)
    p_mut_degree: float = 0.15
    p_mut_oct: float = 0.10
    p_mut_dur: float = 0.12
    p_mut_vel: float = 0.10
    p_flip_rest: float = 0.05
    p_ins_gene: float = 0.08
    p_del_gene: float = 0.06
    # longitud objetivo (en negras) = bars*4.0
    bars: int = 8
    mode: str = "major"
    qpm: int = 96
    expressive_thr: float = 0.65  # si fitness ≥ thr, habilita ties/articulations en fenotipado
    seed: Optional[int] = 42
    # mezcla fitness
    w_rules: float = 0.6
    w_mlp: float = 0.4
    model_path: Optional[str] = None
    # límites de longitud del genoma (para evitar explosión/colapso)
    min_genes: int = 8
    max_genes: int = 256

def _target_quarter_length(bars: int, ts: str = "4/4") -> float:
    ts_obj = meter.TimeSignature(ts)
    return float(ts_obj.barDuration.quarterLength) * float(bars)

def make_key_for_mode(mode: str) -> m21key.Key:
    return m21key.Key('C', 'minor' if 'minor' in mode.lower() else 'major')

def genome_total_qL(g: Genome) -> float:
    return sum(DUR_TABLE[gene.durIdx] for gene in g.genes)

def clip_genome_length(g: Genome, conf: EAConfig) -> None:
    # Clipa por cantidad de genes (seguridad)
    if len(g.genes) > conf.max_genes:
        g.genes = g.genes[:conf.max_genes]
    if len(g.genes) < conf.min_genes:
        while len(g.genes) < conf.min_genes:
            g.genes.append(random_gene(g.mode))

def random_population(conf: EAConfig) -> List[Genome]:
    pop = []
    for _ in range(conf.pop_size):
        g = random_genome(n_compases=conf.bars, mode=conf.mode, qpm=conf.qpm)
        clip_genome_length(g, conf)
        pop.append(g)
    return pop

def mutate_gene(gene: Gene, conf: EAConfig, mode: str) -> Gene:
    # copia “ligera” (Gene es inmutable-ish por dataclass simple)
    d = Gene(**gene.__dict__)
    if random.random() < conf.p_mut_degree:
        d.degree = (d.degree + random.choice([-2,-1,1,2])) % 7
    if random.random() < conf.p_mut_oct:
        d.octShift = int(np.clip(d.octShift + random.choice([-1,1]), -2, 2))
    if random.random() < conf.p_mut_dur:
        d.durIdx = int(np.clip(d.durIdx + random.choice([-1,1]), 0, len(DUR_TABLE)-1))
    if random.random() < conf.p_mut_vel:
        d.velIdx = random.randint(0, len(VEL_TABLE)-1)
    if random.random() < conf.p_flip_rest:
        d.isRest = not d.isRest
    # Si más adelante habilitás articulaciones/ties en el genotipo:
    # (Acá solo se mutan si el individuo supera el gating durante fenotipado,
    # pero podés habilitar también mutaciones aquí si querés)
    return d

def mutate_genome(g: Genome, conf: EAConfig) -> Genome:
    out = Genome(**{**g.__dict__})
    # por-gen
    out.genes = [mutate_gene(gg, conf, g.mode) if random.random() < conf.mutation_rate else gg for gg in g.genes]

    # inserción/eliminación
    if random.random() < conf.p_ins_gene:
        ins_pos = random.randint(0, len(out.genes))
        out.genes.insert(ins_pos, random_gene(out.mode))
    if len(out.genes) > conf.min_genes and random.random() < conf.p_del_gene:
        del_pos = random.randrange(len(out.genes))
        del out.genes[del_pos]

    clip_genome_length(out, conf)
    return out

def crossover_one_point(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if not a.genes or not b.genes:
        return a, b
    i = random.randint(1, min(len(a.genes), len(b.genes)) - 1)
    child1 = Genome(genes=a.genes[:i] + b.genes[i:], mode=a.mode, time_signature=a.time_signature,
                    qpm=a.qpm, base_octave=a.base_octave, tessitura_low=a.tessitura_low,
                    tessitura_high=a.tessitura_high, allow_expressive=False)
    child2 = Genome(genes=b.genes[:i] + a.genes[i:], mode=b.mode, time_signature=b.time_signature,
                    qpm=b.qpm, base_octave=b.base_octave, tessitura_low=b.tessitura_low,
                    tessitura_high=b.tessitura_high, allow_expressive=False)
    return child1, child2

def crossover_uniform(a: Genome, b: Genome, swap_prob: float = 0.5) -> Tuple[Genome, Genome]:
    L = min(len(a.genes), len(b.genes))
    g1 = []
    g2 = []
    for i in range(L):
        if random.random() < swap_prob:
            g1.append(b.genes[i])
            g2.append(a.genes[i])
        else:
            g1.append(a.genes[i])
            g2.append(b.genes[i])
    # si hay longitud desigual, arrastramos el resto del padre más largo (raramente pasa por random_genome)
    if len(a.genes) > L:
        g1 += a.genes[L:]
    if len(b.genes) > L:
        g2 += b.genes[L:]
    child1 = Genome(genes=g1, mode=a.mode, time_signature=a.time_signature,
                    qpm=a.qpm, base_octave=a.base_octave, tessitura_low=a.tessitura_low,
                    tessitura_high=a.tessitura_high, allow_expressive=False)
    child2 = Genome(genes=g2, mode=b.mode, time_signature=b.time_signature,
                    qpm=b.qpm, base_octave=b.base_octave, tessitura_low=b.tessitura_low,
                    tessitura_high=b.tessitura_high, allow_expressive=False)
    return child1, child2

def tournament_select(pop: List[Genome], fits: List[float], k: int) -> int:
    idxs = np.random.choice(len(pop), size=k, replace=False)
    best = max(idxs, key=lambda i: fits[i])
    return int(best)

def evaluate_genome(g: Genome, conf: EAConfig, clf=None) -> float:
    # target bars/timeSignature ya están en g
    # gating expresivo (ligaduras/articulaciones) según fitness previo no es posible aquí;
    # lo hacemos con heurística: primero fenotipar sin gating, evaluar; si supera thr, re-fenotipar con gating
    phen = phenotype_from_genome(g)
    feats = compute_features(phen, make_key_for_mode(g.mode))
    frule = rule_based_fitness(feats)

    mlp_score = 0.0
    if clf is not None:
        try:
            x = features_to_vector(feats).reshape(1, -1)
            if hasattr(clf, "predict_proba"):
                mlp_score = float(clf.predict_proba(x)[0, 1])
            else:
                mlp_score = float(clf.predict(x)[0])
        except Exception:
            mlp_score = 0.0

    # score combinado
    score = conf.w_rules * frule + conf.w_mlp * mlp_score

    # gating: re-fenotipar para permitir ties/articulations si supera umbral (no cambia score en esta pasada)
    if score >= conf.expressive_thr and not g.allow_expressive:
        g.allow_expressive = True  # efecto para la próxima generación si sobrevive
    return float(score)

def load_model_if_any(conf: EAConfig):
    if conf.model_path is None:
        return None
    try:
        import joblib
        return joblib.load(conf.model_path)
    except Exception:
        return None

def evolve(conf: EAConfig, verbose: bool = True) -> Tuple[Genome, float]:
    if conf.seed is not None:
        random.seed(conf.seed)
        np.random.seed(conf.seed)

    clf = load_model_if_any(conf)
    pop = random_population(conf)

    # eval inicial
    fits = [evaluate_genome(g, conf, clf) for g in pop]
    best_idx = int(np.argmax(fits))
    best_g, best_f = pop[best_idx], fits[best_idx]
    if verbose:
        print(f"[GEN 0] best = {best_f:.4f}")

    for gen in range(1, conf.generations + 1):
        new_pop: List[Genome] = []

        # elitismo
        elite_idx = np.argsort(fits)[-conf.elitism:][::-1]
        for i in elite_idx:
            # copiar “duro” para no arrastrar referencias
            g = Genome(**{**pop[i].__dict__})
            g.genes = [Gene(**gg.__dict__) for gg in pop[i].genes]
            new_pop.append(g)

        # reproducción
        while len(new_pop) < conf.pop_size:
            # selección
            p1 = pop[tournament_select(pop, fits, conf.tournament_k)]
            p2 = pop[tournament_select(pop, fits, conf.tournament_k)]
            c1, c2 = (Genome(**{**p1.__dict__}), Genome(**{**p2.__dict__}))
            c1.genes = [Gene(**gg.__dict__) for gg in p1.genes]
            c2.genes = [Gene(**gg.__dict__) for gg in p2.genes]

            # crossover
            if random.random() < conf.crossover_rate:
                if random.random() < 0.5:
                    c1, c2 = crossover_one_point(c1, c2)
                else:
                    c1, c2 = crossover_uniform(c1, c2, swap_prob=0.5)

            # mutation
            c1 = mutate_genome(c1, conf)
            c2 = mutate_genome(c2, conf)

            new_pop.append(c1)
            if len(new_pop) < conf.pop_size:
                new_pop.append(c2)

        pop = new_pop
        fits = [evaluate_genome(g, conf, clf) for g in pop]
        gen_best_idx = int(np.argmax(fits))
        gen_best_f = fits[gen_best_idx]
        if gen_best_f > best_f:
            best_f = gen_best_f
            best_g = Genome(**{**pop[gen_best_idx].__dict__})
            best_g.genes = [Gene(**gg.__dict__) for gg in pop[gen_best_idx].genes]

        if verbose and (gen % 5 == 0 or gen == conf.generations):
            avg_f = float(np.mean(fits))
            print(f"[GEN {gen}] best = {best_f:.4f} | gen_best = {gen_best_f:.4f} | mean = {avg_f:.4f}")

    return best_g, best_f


# ======================================
#  CLI
# ======================================

def main():
    ap = argparse.ArgumentParser(description="Melody EA base: genotipo por grados, features y entrenamiento MLP.")
    sub = ap.add_subparsers(dest="cmd")

    # Entrenamiento
    ap_train = sub.add_parser("train", help="Entrenar MLP con carpeta de MIDIs buenos (genera negativos sintéticos).")
    ap_train.add_argument("--good-dir", required=True, type=str, help="Carpeta con .mid/.midi (positivos)")
    ap_train.add_argument("--out-model", required=True, type=str, help="Ruta para guardar el modelo .pkl")
    ap_train.add_argument("--negatives-per-good", type=int, default=2, help="Negativos por cada positivo (default=2)")
    ap_train.add_argument("--limit", type=int, default=None, help="Procesar como máximo N MIDIs (debug/rápido)")
    ap_train.add_argument("--show-progress", action="store_true", help="Mostrar progreso por archivo")
   
    # Analizar un MIDI
    ap_an = sub.add_parser("analyze", help="Analizar un MIDI y calcular fitness por reglas y (opcional) score MLP.")
    ap_an.add_argument("midi", type=str, help="Ruta a archivo .mid/.midi")
    ap_an.add_argument("--model", type=str, default=None, help="Modelo MLP .pkl para score (opcional)")

    # Generar un individuo aleatorio y exportar MIDI
    ap_gen = sub.add_parser("gen-random", help="Generar individuo aleatorio y exportar MIDI.")
    ap_gen.add_argument("--out-midi", required=True, type=str, help="Dónde guardar el MIDI generado")
    ap_gen.add_argument("--mode", type=str, default="major", choices=["major", "minor"])
    ap_gen.add_argument("--qpm", type=int, default=96)
    ap_gen.add_argument("--bars", type=int, default=8)

    # Evolución
    ap_evo = sub.add_parser("evolve", help="Correr el Algoritmo Evolutivo para generar una melodía.")
    ap_evo.add_argument("--out-midi", required=True, type=str, help="MIDI de salida del mejor individuo")
    ap_evo.add_argument("--mode", type=str, default="major", choices=["major", "minor"])
    ap_evo.add_argument("--qpm", type=int, default=96)
    ap_evo.add_argument("--bars", type=int, default=8)
    ap_evo.add_argument("--pop-size", type=int, default=40)
    ap_evo.add_argument("--generations", type=int, default=60)
    ap_evo.add_argument("--elitism", type=int, default=2)
    ap_evo.add_argument("--tournament-k", type=int, default=3)
    ap_evo.add_argument("--crossover-rate", type=float, default=0.9)
    ap_evo.add_argument("--mutation-rate", type=float, default=0.25)
    ap_evo.add_argument("--model", type=str, default=None, help="Ruta a mlp.pkl para mezclar score")
    ap_evo.add_argument("--w-rules", type=float, default=0.6)
    ap_evo.add_argument("--w-mlp", type=float, default=0.4)
    ap_evo.add_argument("--seed", type=int, default=42)
    ap_evo.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if args.cmd == "train":
        train_mlp_on_folder(args.good_dir, args.out_model, 
                        negatives_per_good=args.negatives_per_good,
                        limit=args.limit, show_progress=args.show_progress)

    elif args.cmd == "analyze":
        ind, key_obj, feats = parse_midi_to_individual(args.midi)
        print(json.dumps(feats, indent=2, ensure_ascii=False))
        rb = rule_based_fitness(feats)
        print(f"\nRule-based fitness: {rb:.4f}")
        if args.model:
            try:
                p = score_with_model(feats, args.model)
                print(f"MLP score (prob buena): {p:.4f}")
                # gating expresivo como ejemplo
                g = random_genome(n_compases=4, mode=key_obj.mode, qpm=ind.metadata.get("qpm", 90))
                g = enable_expressive_if_threshold(g, fitness_value=max(rb, p), thr=0.65)
                phen = phenotype_from_genome(g)
                out_tmp = pathlib.Path(args.midi).with_suffix(".analysis_gen.mid")
                save_individual_as_midi(phen, out_tmp)
                print(f"[Info] Generé una melodía aleatoria con gating expresivo -> {out_tmp}")
            except Exception as ex:
                print(f"[WARN] No pude evaluar con el modelo: {ex}")

    elif args.cmd == "gen-random":
        g = random_genome(n_compases=args.bars, mode=args.mode, qpm=args.qpm)
        ind = phenotype_from_genome(g)
        save_individual_as_midi(ind, args.out_midi)
        print(f"[OK] MIDI generado en {args.out_midi}")

    elif args.cmd == "evolve":
        conf = EAConfig(
            pop_size=args.pop_size,
            generations=args.generations,
            tournament_k=args.tournament_k,
            elitism=args.elitism,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            bars=args.bars,
            mode=args.mode,
            qpm=args.qpm,
            model_path=args.model,
            w_rules=args.w_rules,
            w_mlp=args.w_mlp,
            seed=args.seed
        )
        best_g, best_f = evolve(conf, verbose=args.verbose)
        print(f"[EA] Best fitness = {best_f:.4f}")
        # Fenotipar y guardar
        best_g = enable_expressive_if_threshold(best_g, best_f, thr=conf.expressive_thr)
        ind = phenotype_from_genome(best_g)
        save_individual_as_midi(ind, args.out_midi)
        print(f"[EA] Guardado: {args.out_midi}")

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
