# GPU Speech Lab app

Start from the project folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-app.ps1
```

Open http://127.0.0.1:8765. Keep the server process running. The app binds only to the local loopback interface and requires the existing CUDA runtime, GPU and checkpoint. It keeps the original Tacotron model loaded and serializes synthesis requests. No replacement architecture, vocoder or fabricated quality scores are used. The minimal browser interface uses local system fonts.

## Timing definitions

All durations use Python's monotonic `time.perf_counter()` clock. Audio duration is read from the generated PCM WAV file.

| Metric | Definition | Scope |
|---|---|---|
| Model execution | TensorFlow session call duration | Decoder, mel, Griffin-Lim; first job also includes software execution tracing |
| Synthesis service time | Worker start through WAV write | Text preparation, model execution, inverse preemphasis, endpoint trimming and file output |
| Delivery latency | Submission to result readiness | Queue, synthesis, diagnostic plot creation; excludes browser polling delay, HTTP transfer and model loading |
| Queue wait | Submission to worker start | Waiting for serialized GPU service |
| RTF | Synthesis service seconds / generated audio seconds | Lower is better; <1 indicates faster than real time |
| Audio throughput / RTFX | Audio seconds / synthesis service seconds | Reciprocal of RTF; higher is better |
| Request throughput | Completed jobs / sum of synthesis service seconds | Serial service capacity; excludes queue, idle, plotting, loading and failed job times; not concurrent endpoint load-test throughput |
| Error rate | Failed synthesis jobs / terminal jobs | Terminal = completed + failed; pending jobs excluded; N/A before a job terminates |
| Rejected requests | Count of invalid POST requests | Separate from model failures |
| P50 / P95 | Linear interpolation of sorted delivery latencies | Current process only; small sample statistics are descriptive |

Aggregate RTF is total synthesis time / total audio duration, not the unweighted average of per-job ratios. Cold model load is reported separately. The first request's execution trace verifies actual GPU MatMul/Conv2D operations and adds overhead; subsequent requests do not retrace. No streaming or time-to-first-audio claim is made. WAVs, spectrograms, GPU evidence and JSON reports are saved in `app_runs/`; this folder is excluded from Git. Session counters reset on restart, while files remain available locally.

## Precision, recall, F1 and transcript error

These measures require an **observed transcript**: words actually heard in the generated audio or obtained from independent speech recognition. The app does not run ASR and does not copy the input into a transcript automatically. Until a transcript is supplied, these scores are unmeasured. Any manual transcript's reliability is the user's responsibility.

Reference = original input text; hypothesis = supplied transcript. Normalize Unicode NFKC, casefold and strip punctuation, retaining internal apostrophes. Align normalized word sequences using minimum edit distance; equal-cost alignments prefer a diagonal operation, then deletion, then insertion. Let C = exact aligned words, S = substitutions, D = deletions, I = insertions.

- Precision = C / hypothesis word count; empty hypothesis gives zero.
- Recall = C / reference word count.
- F1 = harmonic mean of precision and recall; zero when both are zero.
- WER = (S + D + I) / reference word count. It may exceed 100% with many insertions.
- CER = character edit distance / normalized reference character count, including spaces.

These are explicitly **word-match** measures, not speaker classification accuracy, acoustic regression accuracy, naturalness, MOS, or an automatic speech-quality benchmark. Punctuation-only references cannot be evaluated. Numerals are not expanded by this evaluator, so transcript conventions should match the input (the original model's English cleaner does expand numbers). See [scikit-learn's precision, recall and F-measures documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measures) for the classification meaning of these metrics.

## Training error

The training evidence comes from `results/gpu-verified/experiment_results.csv` from the actual three-step GPU smoke test. Original loss = mean mel L1 + 0.5 full-spectrum linear L1 + 0.5 low-frequency linear L1 (below 3 kHz). Final recorded total loss is 0.9332861900. This is training loss on eight prepared samples, not a held-out test error. The app uses the pretrained `tacotron-20180906` checkpoint, not the three-step smoke-test checkpoint.

## Validation

Metric algorithm checks: `./run.ps1 -m unittest discover -s tests -p test_app_metrics.py -v`. Known substitution, insertion, deletion, silence, ordering, normalization, weighted timing and percentile fixtures test formulas; their numbers are test fixtures, never model performance. Actual live GPU app evidence is separately recorded in `results/app/verification.json` after synthesis and WAV/API checks.

## Simplified interface

The current interface shows only a text box and Generate button. A brief loading or error message appears when needed, and the audio player appears after generation. Timing metrics and diagnostic files continue to be recorded locally; they are not displayed in the interface. GET /api/status returns session metrics, GET /api/jobs/{id} returns request metrics, and POST /api/evaluate accepts an id and observed transcript for word-match evaluation.


## Quality-path timing scope

The updated app sums neural TensorFlow calls across short phrases for model execution. CPU librosa Griffin-Lim, phrase normalization, pauses and WAV writing are included in synthesis service time and RTF. See SPEECH_QUALITY.md for the actual recognition comparison.

