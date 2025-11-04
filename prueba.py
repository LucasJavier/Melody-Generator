from __future__ import annotations \
# Para evaluacion pospuesta, asi podemos "anotar" clases antes que  sean definidas.
from dataclasses import dataclass, asdict
# dataclass: decorador.
# asdict: produce una representacion de diccionario anidada de una instancia de dataclass.
from typing import List, Optional, Dict, Any, Tuple
# Primitivas usuales, como el contendor Lista, valores opcionales Optional, mapeos Dict, cualquier tipo
# Any, tuplas Tuple.
from collections import Counter, defaultdict
# Counter: una subclase de Dict que cuenta items hasheados.
# defaultdict: para proporcionar una factory para las claves que faltan
import json
import math
import argparse
# argparse: Es el analizador de argumentos CLI estándar. Crea interfaces de línea de comandos, gestiona el
# texto de ayuda, la conversión y validación de tipos, y sale con información de uso en caso de errores de
# análisis.
import pathlib
# pathlib: Proporciona la API de ruta de sistema de archivos orientado a objetos (Path), que es más segura
# y expresiva que las cadenas sin formato y os.path. Se prefiere Path para la manipulación de rutas, la 
# lectura/escritura de archivos y la compatibilidad entre plataformas.

import numpy as np
from music21 import converter, stream, note, chord, meter, key as m21key, tempo, interval, tie
'''
1) converter: para leer y escribir partituras en distintos formatos (MIDI, MusicXML, ABC, MuseScore, etc.).
La función más usada es converter.parse(...) que carga un archivo o una cadena y devuelve un objeto 
music21.stream.Score (u otro tipo de stream según el contenido). También ofrece métodos para convertir 
entre formatos, para cargar desde URLs, y para operaciones de inspección del archivo fuente.
2) steam: contiene las estructuras de contenedor básicas de music21: Score, Part, Measure, Voice y 
utilidades como Stream, flatten(), recurse(), insert() y makeMeasures(). Un Score representa una partitura
completa (varias partes); un Part contiene la secuencia de notas/compases de un instrumento/voz; y 
Measure agrupa eventos por compás. Streams manejan offsets (posición temporal), permiten iterar sobre 
elementos musicales, agrupar por clases (getElementsByClass) y reconstruir medidas. 
3) note: define las clases que representan eventos simples: Note, Rest, Pitch, Tie y objetos relacionados.
note.Note contiene información de altura (pitch con propiedades como midi, nameWithOctave), duración 
(duration.quarterLength), dinámica/velocidad (volume.velocity), articulaciones, ligaduras (tie) y más. 
note.Rest modela silencios con duración.
4) chord: agrupa la representación de acordes; la clase principal es chord.Chord. Un Chord contiene varias
Pitch y propiedades útiles como pitches (lista de alturas), commonName, bass(), y root(). 
5) meter: contiene clases para la métrica: TimeSignature y utilidades relacionadas. 
meter.TimeSignature('4/4') crea una indicación métrica; los objetos Measure usan esa información para 
interpretar beats y offsets. meter también facilita el cálculo de beat y, junto con beatStrength 
(atributo que music21 asigna a notas según su posición dentro del compás), sirve para medir alineación 
rítmica con el pulso.
6) key: define Key y KeySignature y funciones para trabajar con tonalidad y transposición (usamos como m21key).
m21key.Key('C','major') representa una tonalidad; getPitches() devuelve las notas de la escala, y hay 
utilidades para obtener intervalos de transposición entre tonalidades (transposeIntervalFromKeyToKey).
Además, ofrece análisis de tonalidad (score.analyze('key')) que devuelve un Key o un objeto con tonic/mode.
Es esencial para normalizar a C mayor/menor o para extraer información tonal de una pieza.
7) tempo: contiene MetronomeMark y utilidades para marcar el tempo (p. ej. tempo.MetronomeMark(number=120)).
MetronomeMark.number suele contener los BPM (qpm) y puede insertarse en un Stream para preservar el tempo. 
8) interval: sirve para trabajar con intervalos musicales: representarlos, medirlos y usarlos para 
transponer notas/objetos musicales.
9) tie: provee la clase y utilidades para representar ligaduras de nota (ties) en la notación musica. Es 
la forma en que la partitura indica que una nota debe mantenerse (sostenerse) a través de la frontera entre 
dos notas escritas idénticas (misma altura), a menudo cuando una duración sobrepasa el fin de un compás.
'''


# ----------- Data structures -----------

@dataclass
class Event:
    # pitch: MIDI int 0..127, or None for rest
    pitch: Optional[int]
    duration_qL: float                # duration in quarterLength units (e.g., 1.0 = quarter note)
    velocity: Optional[int]           # 0..127 or None if not present
    beat: float                       # position within the measure (1-based in music21)
    measure: int                      # measure number (1-based)
    is_rest: bool
    articulations: List[str]
    tie: Optional[str]                # 'start' | 'stop' | 'continue' | None

@dataclass
class Individual:
    metadata: Dict[str, Any]          # key, mode, timeSignature, qpm, tessitura, etc.
    events: List[Event]               # monophonic event sequence (notes + rests)

# ----------- Utilities -----------

# Redondea un valor temporal a una grilla rítmica fija, ya que los MIDI reales suelen traer 
# “micro-desfasajes” (p.ej. 0.249 en vez de 0.25)
def _quantize(value: float, grid: float) -> float:
    if grid is None or grid <= 0:
        return value
    return round(value / grid) * grid

# Toma la nota mas alta del acorde y lo utiliza como parte de la secuencia melodica
def _pitch_of_chord(c: chord.Chord) -> int:
    # take highest note as melodic head (simple heuristic)
    p = max(c.pitches, key=lambda p: p.midi)
    return p.midi

# Utiliza '_pitch_of_chord' para transformar acordes por melodias, utilizando las notas mas altas de estos
def _to_monophonic(part: stream.Part) -> stream.Part:
    """
    Very simple melody extraction heuristic:
    - Flatten the part, replace Chord by its top note (highest pitch).
    - Keep rests as rests.
    """
    flat = part.flatten().notesAndRests.stream()
    mono = stream.Part()
    for el in flat:
        if isinstance(el, chord.Chord):
            n = note.Note(_pitch_of_chord(el))
            n.duration = el.duration
            n.offset = el.offset
            # copy velocity if present
            if el.volume and el.volume.velocity is not None:
                n.volume.velocity = el.volume.velocity
            # copy articulations if any
            for a in el.articulations:
                n.articulations.append(a)
            mono.insert(el.offset, n)
        else:
            mono.insert(el.offset, el)
    # Rebuild measures
    if part.timeSignature is not None:
        mono.insert(0, part.timeSignature)
    if part.keySignature is not None:
        mono.insert(0, part.keySignature)
    for t in part.recurse().getElementsByClass(tempo.MetronomeMark):
        mono.insert(t.offset, t)
    mono.makeMeasures(inPlace=True)
    return mono

# Analiza la tonalidad (clave y modo)
def _analyze_key(s: stream.Score | stream.Part) -> Tuple[str, str, m21key.Key]:
    # Let music21 guess key; returns tonic name and mode ('major'/'minor' etc.)
    k = s.analyze('key')
    tonic = k.tonic.name    # 'C', 'A-', etc.
    mode = k.mode           # 'major', 'minor', 'dorian', ...
    return tonic, mode, k

# Transpone a Do (M o m)
def _transpose_to_C(part: stream.Part) -> Tuple[stream.Part, m21key.Key]:
    tonic, mode, k = _analyze_key(part)
    # Build target key: C major or C minor depending on detected mode
    target_tonic = 'C'
    target_mode = 'minor' if 'minor' in mode.lower() else 'major'
    target_key = m21key.Key(target_tonic, target_mode)
    # Transponer por el intervalo entre la tónica detectada y C
    i = interval.Interval(k.tonic, target_key.tonic)
    transposed = part.transpose(i, inPlace=False)
    return transposed, target_key

# Para obtener todos los datos de un steam de notas
def _collect_events(part: stream.Part, q_grid: float | None = 0.25) -> List[Event]:
    events: List[Event] = []
    flat = part.flatten()

    for el in flat.notesAndRests:
        # saltear apoyaturas (grace notes)
        if hasattr(el, 'duration') and getattr(el.duration, 'isGrace', False):
            continue

        # número de compás y beat (si no existen, usa defaults)
        mnum = int(getattr(el, 'measureNumber', 0) or 0)
        b = float(getattr(el, 'beat', 1.0))

        d = _quantize(el.quarterLength, q_grid)

        if isinstance(el, note.Note):
            vel = el.volume.velocity
            arts = [a.classes[0] for a in getattr(el, 'articulations', [])]
            tie = el.tie.type if el.tie is not None else None
            events.append(Event(
                pitch=el.pitch.midi,
                duration_qL=d,
                velocity=vel,
                beat=b,
                measure=mnum,
                is_rest=False,
                articulations=arts,
                tie=tie
            ))
        elif isinstance(el, note.Rest):
            events.append(Event(
                pitch=None,
                duration_qL=d,
                velocity=None,
                beat=b,
                measure=mnum,
                is_rest=True,
                articulations=[],
                tie=None
            ))

    # limpiar duraciones cero tras cuantizar
    events = [e for e in events if e.duration_qL > 0]
    return events

# Devuelte el rango de la melodia
def _tessitura(events: List[Event]) -> Tuple[Optional[int], Optional[int]]:
    pitches = [e.pitch for e in events if (e.pitch is not None)]
    if not pitches:
        return None, None
    return int(min(pitches)), int(max(pitches))

# Obtenemos las notas de la escala en valor numericos
def _pitch_classes_in_key(k: m21key.Key) -> set[int]:
    return {p.pitchClass for p in k.getPitches()}

# Cuenta cuantas de las notas caen dentro de las notas "validas" de la clase
def _scale_fit(events: List[Event], key_obj: m21key.Key) -> float:
    pcs = _pitch_classes_in_key(key_obj)
    notes = [e for e in events if not e.is_rest and e.pitch is not None]
    if not notes:
        return 0.0
    in_scale = sum(( (e.pitch % 12) in pcs ) for e in notes)
    return in_scale / len(notes)

# Intervalo entre las notas (para ver que no haya saltos grandes) usando diferencia entre el tono anterior y el siguiente
def _intervals_semitones(events: List[Event]) -> List[int]:
    # consecutive note-to-note (skip rests)
    notes = [e.pitch for e in events if e.pitch is not None]
    return [abs(notes[i+1] - notes[i]) for i in range(len(notes)-1)]

# Cuenta cuanto intervalos (entre notas) satisfacen un umbral, frente al total de intervalos
def _step_leap_ratio(intervals: List[int], step_threshold: int = 5) -> float:
    if not intervals:
        return 0.0
    steps = sum(1 for d in intervals if d <= step_threshold)
    return steps / len(intervals)

# Arma un histograma normalizado
def _histogram_norm(values: List[Any]) -> Dict[Any, float]:
    if not values:
        return {}
    c = Counter(values) # Contamos cuantas veces aparece cada elemento de la lista, y guarda esas cuentas
    # como pares clave-valor, donde la clave es el elemento y el valor su frecuencia
    total = sum(c.values())
    return {k: v / total for k, v in c.items()} # Suma 1

# Evalua el ritmo, si es muy soso o muy caotico
def _entropy_from_hist(h: Dict[Any, float]) -> float:
    if not h:
        return 0.0
    return -sum(p * math.log(p + 1e-12) for p in h.values())

# Arma un histograma normalizado con la duracion de las notas
def _duration_bins(events: List[Event]) -> Dict[float, float]:
    return _histogram_norm([e.duration_qL for e in events if e.duration_qL is not None])

# Arma un histograma normalizado con los tonos de las notas
def _pitch_class_hist(events: List[Event]) -> Dict[int, float]:
    pcs = [e.pitch % 12 for e in events if e.pitch is not None]
    return _histogram_norm(pcs)

# Nos devuelve un valor de que tan cuadrada es la melodia con respecto a los acentos.
# Si el nivel que devuelve es alto, las notas caen en tiempos fuertes. Si es bajo, hay sincopas.
def _meter_alignment_score(events: List[Event], part: stream.Part) -> float:
    """Use music21's beatStrength (0..1). Average weighted by velocity if present."""
    strengths = []
    weights = []
    flat = part.flatten() # Aplana a una sola secuencia temporal.
    # Build index by (measure, beat) -> beatStrength (approx)
    strength_index = defaultdict(dict)
    '''
    Esto crea un defaultdict cuya fábrica predeterminada es el tipo dict integrado, es decir, una 
    asignación que crea y devuelve automáticamente un nuevo dict vacío cada vez que se accede a una 
    clave que falta utilizando la sintaxis de corchetes. Es útil para crear diccionarios anidados sin
    necesidad de comprobar primero la existencia de la clave.
    '''
    for n in flat.notesAndRests:
        m = n.measureNumber or 0
        strength_index[m][n.beat] = getattr(n, 'beatStrength', 0.0)
    for e in events:
        s = strength_index.get(e.measure, {}).get(e.beat, 0.0)
        w = (e.velocity if (e.velocity is not None) else 64) if not e.is_rest else 0
        strengths.append(s * w)
        weights.append(w)
    if sum(weights) == 0:
        return 0.0
    return float(sum(strengths) / sum(weights))

# Para ver la densidad de silencios
def _rest_ratio(events: List[Event]) -> float:
    if not events:
        return 0.0
    rests = sum(1 for e in events if e.is_rest)
    return rests / len(events)

# Nos devuelve el factor que representa la cantidad de repiticiones en una lista de secuencias.
# Cuenta qué proporción de n-gramas se repiten dentro de la secuencia que le pasamos. 
# Lo usamos para:
# i) intervalos en semitonos → repetición de patrones melódicos,
# ii) contorno (signo de +/−/0 de los intervalos) → repetición del “perfil” (sube, baja, igual).
def _ngram_repetition_score(sequence: List[int], n: int = 3) -> float:
    """Compute proportion of repeated n-grams in a discrete seq."""
    if len(sequence) < n + 1:
        return 0.0
    grams = [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    c = Counter(grams)
    total = len(grams)
    repeats = sum(freq for gram, freq in c.items() if freq > 1)
    return repeats / total

# Evaluacion de la melodia, usando todas las demas funciones y clases anteriores
def compute_features(ind: Individual, key_obj: m21key.Key) -> Dict[str, Any]:
    ev = ind.events
    # intervals
    intervals = _intervals_semitones(ev)
    # rhythm
    d_hist = _duration_bins(ev)
    H_dur = _entropy_from_hist(d_hist)
    H_dur_max = math.log(max(len(d_hist), 1))
    ent_norm = (H_dur / H_dur_max) if H_dur_max > 0 else 0.0
    # pitch-class
    pc_hist = _pitch_class_hist(ev)
    # scale fit
    f_scale = _scale_fit(ev, key_obj)
    # step/leap
    step_ratio = _step_leap_ratio(intervals)
    mean_interval = float(np.mean(intervals)) if intervals else 0.0
    # rests & density
    rest_r = _rest_ratio(ev)
    # tessitura
    low, high = _tessitura(ev)
    range_semi = (high - low) if (low is not None and high is not None) else 0
    # n-gram repetition (interval-based)
    # Lista con cada elemento la suma entre la nota anterior y siguiente
    int_sign = [np.sign(ev[i+1].pitch - ev[i].pitch) for i in range(len(ev)-1) if (ev[i].pitch is not None and ev[i+1].pitch is not None)]
    rep_int3 = _ngram_repetition_score([e for e in intervals if e is not None], n=3)
    rep_contour3 = _ngram_repetition_score([int(x) for x in int_sign], n=3)
    # n-gram repetition (duration-based)
    dur_seq = [e.duration_qL for e in ev if not e.is_rest]
    rep_dur3 = _ngram_repetition_score(dur_seq, n=3)
    # meter alignment
    # we need a Part with measure/beatStrength; reconstruct a minimal one:
    p = stream.Part()
    ts_str = ind.metadata.get('timeSignature') or '4/4'   # default si falta
    p.insert(0, meter.TimeSignature(ts_str))
    off = 0.0
    for e in ev:
        el = note.Rest() if e.is_rest else note.Note(e.pitch)
        el.duration.quarterLength = e.duration_qL
        p.insert(off, el)
        off += e.duration_qL

    # Crear compases y normalizar notación
    p.makeMeasures(inPlace=True)
    try:
        p.makeNotation(inPlace=True)  # opcional, ayuda a beatStrength
    except Exception:
        pass
    align = _meter_alignment_score(ev, p)

    return {
        # 1.0 → todas las notas están dentro de la escala (sin notas “extrañas”).
        # 0.0 → ninguna nota pertenece a la escala (completamente atonal).
        "Proporcion de notas dentro de la escala (scale_fit)": f_scale, 
        # Bajo (1–3 semitonos) → movimiento suave, paso por grado conjunto (do–re–mi).
        # Medio (4–7) → incluye algunos saltos moderados.
        # Alto (>8) → muchos saltos grandes, melodía más “angular” o saltarina.
        "Promedio de distancia entre notas consecutivas (mean_interval_semitones) ": mean_interval,  
        # Se puede sacar este parametro o hacerlo un hiperparametro de la AE
        # ¿Qué porcentaje de los saltos son razonablemente chicos?. # [0,1], higher ~ smoother
        # 1.0 → todos los intervalos son pequeños (ningún salto mayor a 4ª).
        # 0.5 → mitad suaves, mitad saltos grandes.
        # 0.0 → todos los saltos son grandes.
        "Proporcion de intervalos menores o iguales a 5 semitonos - 4ta justa - (step_ratio_leq5st)": step_ratio,       
        # [0,1], compare to target
        # ~0.0 → todas las notas duran igual → ritmo plano o mecánico.
        # ~1.0 → cada nota tiene una duración distinta → ritmo caótico.
        # Intermedio (0.4–0.7) → buena variedad rítmica.
        "Entropia/diversidad de duracion de notas - normalizada - (duration_entropy_norm)": ent_norm,  
        # [0,1]   
        # 0.0 → no hay pausas → frase melódica continua.
        # 0.1–0.3 → hay pausas ocasionales, respiración natural.
        # 0.5 → demasiadas pausas, fragmentada.   
        "Proporcion de cantidad de silencios (rest_ratio)": rest_r,                   
        # prefer within tessitura  
        # 5–12 → rango moderado (una octava o menos).
        # 12 → rango amplio (melodía que “salta” entre registros). 
        # < 5 → muy monótona.
        "Diferencia entre nota más grave y más aguda - en semitonos - (range_semitones)": range_semi,
        # dict {pc: prob}
        # Ej: 7 (G): 10%,  10 (Bb): 8%, etc
        "Distribución de clases de alturas - cuántas veces aparece cada nota dentro de la octava - en % (pitch_class_hist)": {k: 100*v for k, v in pc_hist.items()},  
        # dict {qL: prob}
        # 0.5 → corcheas (1/8)
        # 1.0 → negras (1/4)
        # 2.0 → blancas (1/2)          
        "Distribución de duraciones de notas - en unidad de negras: 1.0 - (duration_hist) en %": {k: 100*v for k, v in d_hist.items()},    
        # 0: notas repetidas
        # 1–2: movimiento conjunto
        # 3–5: pequeños saltos
        # 5: saltos grandes          
        # EJ: 0 → repetición de la misma nota (≈ 16.7%)
        # 1 → semitono (≈ 8.3%)
        # 2 → tono (≈ 47.2%) ⟵ casi la mitad
        # 3 → 3 semitonos (m3/M3 según contexto) (≈ 11.1%)
        # 4 → 4 semitonos (≈ 11.1%)
        # 5 → 4ª justa (≈ 5.6%)    
        "Frecuencia de intervalos - en semitonos - entre notas consecutivas en % (interval_hist)": {k: 100*v for k, v in _histogram_norm(intervals).items()},
        # [0,1], moderate is good
        # 0.0 → nada se repite → demasiado errático.
        # 1.0 → todo se repite igual → monótono.
        # Intermedio (~0.2–0.6) → equilibrio entre repetición y novedad.
        # Trigrama porque:
        # n=2 (bigramas) capta muy poco contexto.
        # n=3 suele capturar motivos cortos (típicos en melodías tonales) sin volverse frágil al ruido.
        # n>3 aumenta combinatoria y sparsity (menos repeticiones detectables en frases cortas).
        "Proporción de patrones melódicos repetidos - trigramas de intervalos - (rep_ngram3_intervals) ": rep_int3,          
        # [0,1]
        # 0 → completamente impredecible (subidas y bajadas aleatorias).
        # 1 → muy repetitivo en forma (siempre el mismo “gesto” melódico).  
        "Repetición de contornos melódicos - patrones de subir/bajar/igual - (rep_ngram3_contour)": rep_contour3,  
        # [0,1] repetición rítmica   
        # 0 → ritmo completamente libre/aleatorio.
        # 1 → patrón rítmico muy repetido (como un ostinato)."  
        "Repetición de patrones rítmicos - secuencias de 3 duraciones - (rep_ngram3_durations)": rep_dur3, 
        # [0,1], higher ~ more on strong beats
        # 1.0 → todo en pulsos fuertes, muy “cuadrado”.
        # 0.0 → sincopado o desalineado.
        "Qué tanto las notas caen en tiempos fuertes del compás (meter_alignment)": align                   
    }

# Crea un 'Individuo' a partir de un MIDI
# Con 'pick_part': Muchos MIDIs tienen varias pistas/partes (melodía, bajo, acordes, percusión…).
# pick_part te deja elegir qué parte procesar (0 = la primera). El uso típico: --part 0 para la línea
#  melódica; si agarró batería/acompañamiento, probar --part 1, --part 2, etc. Luego _to_monophonic
# reduce acordes a una sola línea (nota más aguda, heurística simple) para quedarte con melodía monofónica.
def parse_midi_to_individual(
    midi_path: str | pathlib.Path,
    quant_grid: float = 0.25,
    pick_part: int = 0,
    force_mode: Optional[str] = None  # 'major'|'minor'|None
) -> Tuple[Individual, m21key.Key, Dict[str, Any]]:
    """
    - Loads MIDI
    - Picks one part (monophonic extraction)
    - Analyzes key and transposes to C major/minor (unless force_mode provided)
    - Extracts events and metadata
    - Computes features
    """
    score = converter.parse(str(midi_path))
    parts = score.parts.stream()
    if len(parts) == 0:
        # treat whole score as a single part
        part = score.parts[0] if hasattr(score, 'parts') and len(score.parts) else score.flat
    else:
        part = parts[pick_part]

    part = _to_monophonic(part)

    # analyze key
    tonic, mode, detected_key = _analyze_key(part)
    mode_to_use = force_mode if force_mode in ('major', 'minor') else ('minor' if 'minor' in mode.lower() else 'major')

    # transpose to C (respecting mode)
    transposed, key_obj = _transpose_to_C(part)
    # If forcing mode, re-key to that mode:
    if force_mode is not None and key_obj.mode != force_mode:
        key_obj = m21key.Key('C', force_mode)

    # collect events
    events = _collect_events(transposed, q_grid=quant_grid)

    # metadata
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
            "originalKey": {"tonic": tonic, "mode": mode},
            "normalizedKey": {"tonic": key_obj.tonic.name, "mode": key_obj.mode},
            "timeSignature": ts,
            "qpm": qpm,
            "tessitura_midi": {"low": low, "high": high},
            "quant_grid_qL": quant_grid,
        },
        events=events
    )

    feats = compute_features(ind, key_obj)
    return ind, key_obj, feats

# Pasar de un 'Individuo' a un 'Steam' que es una secuencia de notas de Music21
def individual_to_stream(ind: Individual) -> stream.Part:
    p = stream.Part()
    if ind.metadata.get("timeSignature"):
        p.insert(0, meter.TimeSignature(ind.metadata["timeSignature"]))
    off = 0.0
    for e in ind.events:
        el = note.Rest() if e.is_rest or e.pitch is None else note.Note(e.pitch)
        el.duration.quarterLength = e.duration_qL
        if not e.is_rest and e.velocity is not None:
            el.volume.velocity = e.velocity
        for a in e.articulations or []:
            try:
                # naive articulation attach
                from music21 import articulations as arts
                if hasattr(arts, a):
                    el.articulations.append(getattr(arts, a)())
            except Exception:
                pass
        if e.tie in ('start', 'stop', 'continue'):
            try:
                el.tie = tie.Tie(e.tie)
            except Exception:
                pass
        p.insert(off, el)
        off += e.duration_qL
    return p

# Guardamos el 'Individuo' es un MIDI
def save_individual_as_midi(ind: Individual, out_path: str | pathlib.Path):
    p = individual_to_stream(ind)
    s = stream.Score()
    # add normalized key signature if present
    # '0' (offset) para indicar que es al inicio del stream, al inicio de la partitura
    nk = ind.metadata.get("normalizedKey")
    if nk and nk.get("tonic") and nk.get("mode"):
        s.insert(0, m21key.Key(nk["tonic"], nk["mode"]))
    if ind.metadata.get("timeSignature"):
        s.insert(0, meter.TimeSignature(ind.metadata["timeSignature"]))
    if ind.metadata.get("qpm"):
        s.insert(0, tempo.MetronomeMark(number=ind.metadata["qpm"]))
    s.insert(0, p)
    s.write("midi", fp=str(out_path))

def save_individual_json(ind: Individual, feats: Dict[str, Any], out_json: str | pathlib.Path):
    data = {
        "metadata": ind.metadata,
        "events": [asdict(e) for e in ind.events],
        "features": feats
    }
    pathlib.Path(out_json).write_text(json.dumps(data, ensure_ascii=False, indent=2))

# ----------- CLI -----------

def main():
    ap = argparse.ArgumentParser(description="Extract monophonic individual + features from MIDI.")
    ap.add_argument("midi", type=str, nargs='?', default=None, help="Path to .mid/.midi (optional if --demo-bach is used)")
    ap.add_argument("--grid", type=float, default=0.25, help="Quantization grid in quarterLength (default 0.25 = 16th)")
    ap.add_argument("--part", type=int, default=0, help="Index of part to extract (default 0)")
    ap.add_argument("--force-mode", type=str, default=None, choices=["major", "minor"], help="Force normalization to C major/minor")
    ap.add_argument("--out-json", type=str, default=None, help="Where to save JSON (individual + features)")
    ap.add_argument("--out-midi", type=str, default=None, help="Where to save normalized monophonic MIDI")
    ap.add_argument("--demo-bach", type=str, default=None, help="If set, generates a demo MIDI from corpus (e.g., --demo-bach bach.mid) and uses it")
    args = ap.parse_args()

    if args.demo_bach:
        from music21 import corpus
        demo_path = pathlib.Path(args.demo_bach)
        s = corpus.parse('bach/bwv66.6')  # coral de Bach
        s.write('midi', fp=str(demo_path))
        print("Demo MIDI generated at:", demo_path)
        args.midi = str(demo_path)

    if args.midi is None:
        ap.error("You must provide a MIDI path or use --demo-bach to generate one.")

    ind, kobj, feats = parse_midi_to_individual(args.midi, quant_grid=args.grid, pick_part=args.part, force_mode=args.force_mode)

    print("Detected original key:", ind.metadata["originalKey"])
    print("Normalized key:", ind.metadata["normalizedKey"])
    print("Events:", len(ind.events))
    print("Features (summary):")
    for k, v in feats.items():
        if isinstance(v, dict):
            print(f"  {k}: { {kk: round(vv,4) for kk,vv in v.items()} }")
        else:
            print(f"  {k}: {round(v,4) if isinstance(v,(int,float)) else v}")

    # if args.out_json:
    #     save_individual_json(ind, feats, args.out_json)
    #     print("Saved JSON to:", args.out_json)
    if args.out_midi:
        save_individual_as_midi(ind, args.out_midi)
        print("Saved MIDI to:", args.out_midi)

if __name__ == "__main__":
    main()
