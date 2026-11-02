"""Measured TTS timing and explicitly defined transcript metrics (no ASR guesses)."""
import re
import unicodedata


def tokens(text):
    normalized = unicodedata.normalize('NFKC', text).casefold()
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", normalized, flags=re.UNICODE)


def alignment_counts(reference, observed):
    """Minimum edit alignment; ties prefer exact match, substitution, deletion, insertion."""
    rows = [[(0, 0, 0, 0)] * (len(observed) + 1) for _ in range(len(reference) + 1)]
    for i in range(1, len(reference) + 1):
        rows[i][0] = (0, 0, i, 0)
    for j in range(1, len(observed) + 1):
        rows[0][j] = (0, 0, 0, j)
    for i, expected in enumerate(reference, 1):
        for j, actual in enumerate(observed, 1):
            c, s, d, ins = rows[i-1][j-1]
            diagonal = (c+1, s, d, ins) if expected == actual else (c, s+1, d, ins)
            c, s, d, ins = rows[i-1][j]
            deletion = (c, s, d+1, ins)
            c, s, d, ins = rows[i][j-1]
            insertion = (c, s, d, ins+1)
            rows[i][j] = min([diagonal, deletion, insertion], key=lambda x: sum(x[1:]))
    return rows[-1][-1]


def transcript_metrics(reference, transcript):
    ref, hyp = tokens(reference), tokens(transcript)
    if not ref:
        raise ValueError('Reference must contain at least one word.')
    correct, substitutions, deletions, insertions = alignment_counts(ref, hyp)
    precision = correct / len(hyp) if hyp else 0.0
    recall = correct / len(ref)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ref_chars, hyp_chars = list(' '.join(ref)), list(' '.join(hyp))
    _, cs, cd, ci = alignment_counts(ref_chars, hyp_chars)
    return dict(precision=precision, recall=recall, f1=f1,
                wer=(substitutions+deletions+insertions)/len(ref),
                cer=(cs+cd+ci)/len(ref_chars), correct=correct,
                substitutions=substitutions, deletions=deletions, insertions=insertions,
                reference_words=len(ref), transcript_words=len(hyp),
                scope='Word-match precision/recall/F1 from supplied transcript; not a MOS score.',
                normalization='Unicode NFKC, case-insensitive, punctuation removed; word order retained.')


def timing_metrics(inference_seconds, synthesis_seconds, delivery_seconds, queue_seconds, duration):
    if duration <= 0 or synthesis_seconds <= 0:
        raise ValueError('Audio duration and processing time must be positive.')
    return dict(inference_seconds=inference_seconds, synthesis_seconds=synthesis_seconds,
                latency_seconds=delivery_seconds, queue_seconds=queue_seconds,
                audio_seconds=duration, rtf=synthesis_seconds/duration,
                rtfx=duration/synthesis_seconds, requests_per_second=1/synthesis_seconds)


def aggregate(records, failed, submitted):
    metrics = [r['metrics'] for r in records]
    processing = sum(m['synthesis_seconds'] for m in metrics)
    durations = sum(m['audio_seconds'] for m in metrics)
    latencies = sorted(m['latency_seconds'] for m in metrics)
    def percentile(q):
        if not latencies:
            return None
        index = (len(latencies)-1)*q
        lo = int(index)
        hi = min(lo+1, len(latencies)-1)
        return latencies[lo]+(latencies[hi]-latencies[lo])*(index-lo)
    terminal = len(records)+failed
    return dict(completed=len(records), failed=failed, submitted=submitted,
                error_rate=failed/terminal if terminal else None,
                mean_latency_seconds=sum(latencies)/len(latencies) if latencies else None,
                p50_latency_seconds=percentile(.5), p95_latency_seconds=percentile(.95),
                rtf=processing/durations if durations else None,
                rtfx=durations/processing if processing else None,
                requests_per_second=len(records)/processing if processing else None,
                scope='Current server session; throughput is completed jobs / synthesis service time, excludes idle and queue time.')
