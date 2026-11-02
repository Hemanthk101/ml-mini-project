"""Short phrase synthesis and safe waveform assembly; original Tacotron unchanged."""
import re
import numpy as np

INITIALISMS = {'GPU':'gee pee you', 'CPU':'see pee you', 'RTF':'ar tee eff',
               'TTS':'tee tee ess', 'ML':'em ell', 'AI':'ay eye', 'F1':'eff one'}


def expand_initialisms(text):
    return re.sub(r'\b(?:GPU|CPU|RTF|TTS|ML|AI|F1)\b',
                  lambda match: INITIALISMS[match.group(0).upper()], text, flags=re.IGNORECASE)


def split_phrases(normalized_text, max_chars=80):
    """Split cleaned English at sentences/clauses, then words; never discard text."""
    phrases = []
    for sentence in re.split(r'(?<=[.!?;:,])\s+', normalized_text.strip()):
        current = ''
        for word in sentence.split():
            if current and len(current)+len(word)+1 > max_chars:
                phrases.append(current)
                current = ''
            # Retain long single words rather than silently dropping or fragmenting them.
            current = (current+' '+word).strip()
        if current:
            phrases.append(current)
    return phrases


def trim_silence(wav, sample_rate, threshold_db=-45, pad_seconds=.12):
    """Absolute amplitude avoids negative-only samples being mistaken for silence."""
    wav = np.asarray(wav, dtype=np.float64)
    if not len(wav) or not np.isfinite(wav).all():
        raise RuntimeError('Invalid waveform returned by reconstruction.')
    peak = np.max(np.abs(wav))
    if peak <= 1e-8:
        raise RuntimeError('Reconstruction returned silent audio.')
    active = np.flatnonzero(np.abs(wav) > peak*10**(threshold_db/20))
    pad = int(sample_rate*pad_seconds)
    return wav[max(0, active[0]-pad):min(len(wav), active[-1]+pad+1)]


def phrase_pause(text):
    """A short continuation pause avoids breaking sentences at length boundaries."""
    return .22 if text.rstrip().endswith(('.', '!', '?')) else (
        .12 if text.rstrip().endswith((',', ';', ':')) else .04)


def attention_diagnostics(alignment):
    alignment = np.asarray(alignment)
    if alignment.ndim != 2 or not np.isfinite(alignment).all():
        raise RuntimeError('Invalid attention alignment.')
    positions = alignment.argmax(axis=0)
    return dict(decoder_steps=int(alignment.shape[1]),
                eos_attention=float(alignment[-1, -1]),
                character_coverage=float(np.mean(alignment.max(axis=1) >= .1)),
                backward_jumps=int(np.sum(np.diff(positions) < -3)),
                reached_eos=bool(np.all(alignment[-1, -8:] >= .9)))


def retry_phrases(text, depth):
    """Bounded fallback, preserving intentional repetitions and every word."""
    words=text.split()
    if depth >= 2 or len(words) < 2:
        raise RuntimeError('Tacotron did not finish this phrase reliably. Try shorter English text.')
    middle=len(words)//2
    return [' '.join(words[:middle]), ' '.join(words[middle:])]


def join_phrases(waves, sample_rate, pause_seconds=.18, texts=None):
    if not waves:
        raise ValueError('No speech segments to join.')
    silence = np.zeros(int(sample_rate*pause_seconds), dtype=np.float64)
    chunks = []
    if texts is not None and len(texts) != len(waves):
        raise ValueError('One text is required for every speech segment.')
    for index, wave in enumerate(waves):
        wave = np.asarray(wave, dtype=np.float64).copy()
        # Five-millisecond ramps prevent discontinuities at trimmed boundaries.
        if texts is not None:
            ramp = min(int(sample_rate*.005), len(wave)//2)
            if ramp:
                wave[:ramp] *= np.linspace(0, 1, ramp)
                wave[-ramp:] *= np.linspace(1, 0, ramp)
        peak = np.max(np.abs(wave))
        if peak > 1e-8:
            wave = wave*(.92/peak)
        if chunks:
            chunks.append(silence if texts is None else
                          np.zeros(int(sample_rate*phrase_pause(texts[index-1]))))
        chunks.append(wave)
    return np.concatenate(chunks)
