# Feasible fixes for continuity repetition evaluation and memory

The supplied 2018 CS229 report describes weak SVR/simple-network outputs and repeated words in its small sequence model. This project uses the original Tacotron implementation and its existing pretrained checkpoint; it does not retrain those student baselines or introduce Tacotron 2.

## Implemented inference changes

- The upstream inference helper compares predicted floating-point frames with an exact zero vector. In our saved examples this did not terminate decoding, so each phrase ran to the 200-step ceiling. The app now opts into an attention-based stopping policy: at least 20 steps (or half the input character count, whichever is greater) and eight consecutive steps with at least 0.9 attention on the actual final EOS character. This adds no learned weights or checkpoint variables. Ordinary upstream CLI inference retains its original default unless explicitly enabled.
- Attention history is reusable when the inference policy is enabled. Overlapping windows must not clear TensorArray entries after reading.
- Every accepted phrase has a sustained EOS check. If a phrase does not complete, the app retries it as two smaller phrases, preserving every word. Retry depth is bounded at two; an unresolved phrase fails explicitly rather than publishing unfinished speech as success. Intentional repeated text is never deduplicated.
- Pauses depend on the preceding phrase: 220 ms after a sentence, 120 ms after a clause, and 40 ms after a length-based continuation. Five-millisecond edge ramps reduce discontinuities at joins. PCM peak headroom remains 0.92.
- TensorFlow sessions use two intra/inter-op threads and GPU memory growth. Checkpoint loading restores variables directly rather than first allocating random initialized copies. GPU registration uses the loaded session; actual GPU operations are still required by the first synthesis trace.

The original embeddings, CBHGs, attention weights, GRUs, projections, spectrogram parameters, dataset and Griffin–Lim reconstruction remain. A stopping heuristic can still make mistakes; independent speech evaluation is required.

## Implemented training and preprocessing changes

- Existing length bucketing is retained. An optional padded-frame budget creates smaller batches when needed and explicitly rejects an oversized utterance rather than dropping it silently.
- Feeder groups fetch `batch_size × 4` examples rather than `batch_size × 32`; queue capacity defaults to two rather than eight. These are 87.5% fewer prefetched example slots and 75% fewer queue slots, not claims about whole-process RAM savings. The frame budget can subdivide a nominal batch.
- Training targets are memory-mapped read-only. Only the current batch is padded and copied. Statistics record the maximum prepared batch bytes and the real/padded frame totals.
- LJSpeech preprocessing defaults to one worker and runs directly in the parent for that setting, avoiding a second TensorFlow process on low-memory Windows machines. Explicit parallel preprocessing remains available.

## Low loss is not a speech-quality certificate

`split_dataset.py` creates deterministic, disjoint training/validation/test manifests, keeping duplicate normalized transcripts together. The local eight examples produce a 6/1/1 split. This supports a functional check only, not statistical evidence of generalization.

`evaluate_checkpoint.py` restores a checkpoint and evaluates separate examples with teacher forcing, dropout disabled, and batch-normalization updates disabled. It reports original padded loss and valid-frame acoustic error. Training/evaluation filename or transcript overlap is rejected. Upstream LJSpeech-pretrained checkpoints may already have seen the split examples; only a fresh run can claim those examples were withheld from that run.

`benchmark_quality.py` independently transcribes saved GPU-generated audio using a cached Whisper tiny.en model without the reference as a prompt. It reports raw word metrics and metrics after applying identical spoken-number/acronym normalization to reference and transcript. The diagnostic gate requires WER at most 0.20, deletion rate at most 0.10, and insertion rate at most 0.10 on every example. These thresholds are explicit engineering checks, not a published perceptual standard or MOS. Naturalness still requires human listening.

## Commands

From the project folder:

```powershell
.\run.ps1 split_dataset.py
.\run-gpu.ps1 train.py --base_dir . --input training/train_split.txt --name fixes --max_steps 3 --seed 2026 --hparams "batch_size=2" --feeder_group_batches 4 --queue_capacity 2 --max_batch_frames 900 --summary_interval 1000 --checkpoint_interval 1000 --results_dir results/fixes/training
.\run-gpu.ps1 evaluate_checkpoint.py --checkpoint logs-fixes/model.ckpt-3 --input training/validation_split.txt --training_input training/train_split.txt --output results/fixes/validation.json
powershell -ExecutionPolicy Bypass -File .\start-app.ps1
```

Run recognition in an existing PyTorch/Transformers environment after releasing the TTS process on this 8 GB host. Provide the local recognition model path; the benchmark never downloads a model implicitly.

```powershell
python benchmark_quality.py --input results/fixes --model_dir D:\Codex\2026-10-05\rea\work\whisper-tiny.en
```

## Verified speech results

Five actual GPU requests passed: a short sentence, a two-sentence greeting, intentional `go, go, go`, numbers/acronyms, and the original long paragraph. Independent Whisper recognition recovered every word on the first three examples. The technical example also had zero spoken-normalized WER (the raw score treats numerals and number words differently). The paragraph retained a raw WER of 6.52%, with acronym/homophone substitutions remaining. Every example passed the explicit diagnostic gate.

The 17 accepted phrases needed 814 decoder steps in total, compared with 3,400 steps if each ran to the old 200-step cap: 76.1% fewer decoder steps on this suite. This is an operation-count comparison, not a hardware-independent speedup claim. The warm paragraph generated 24.46 seconds of audio in 24.75 seconds of synthesis service time, with 24.92-second delivery latency and RTF 1.012. The first short request included execution tracing and kernel startup; its timing is not a warm benchmark. Earlier 429-second paragraph delivery occurred under different memory pressure, so it is not a controlled A/B speed measurement.

Those timings are preserved in `results/fixes/initial_speech/` and are the measurements used in the report and slides. The final app rerun produced identical SHA-256 hashes for all five WAVs, so the independent recognition results apply to those exact files. Its warm paragraph synthesis took 24.96 seconds, delivery latency 25.16 seconds, and RTF 1.020. The latest suite, timing and session records are in `results/fixes/`; hashes and verification scope are recorded in `verification.json`.

## Verified training and validation

A fresh GPU checkpoint completed three updates using six training pairs, nominal batch size two, a 900 padded-frame budget, four feeder group batches and queue capacity two. Peak prepared batch size was 3,359,752 bytes (3.20 MiB); prepared frames were 3,318 real and 3,380 padded (1.83% padding). These are feeder buffer measurements, not total process or GPU memory usage. Losses were 0.872534, 0.900816 and 0.913511 on different batches; this run does not demonstrate convergence.

The one-example held-out validation ran on GPU with no filename/transcript overlap, dropout disabled and no batch-normalization updates. Valid-frame mel L1 was 0.361848 and weighted linear L1 was 0.497758, totaling 0.859605. Teacher-forced acoustic error does not establish autoregressive speech quality. The published pretrained voice remains the app's inference checkpoint. Evidence is in `results/fixes/training/experiment_summary.json` and `results/fixes/validation.json`. Twenty-eight regression tests and fifteen original text/number/CMUDict test functions passed.

## Remaining limits

The speech baseline is pretrained Tacotron, not a freshly trained voice. Eight data pairs and three updates cannot solve generalization. This host still has limited RAM and a 4 GB GPU, and classic Griffin–Lim can introduce audible artifacts. The feasible corrections address concrete execution and inference failures; larger training, broader evaluation and listening studies remain necessary.

Sources: upstream helper and batching code, https://github.com/keithito/tacotron ; model discussion and waveform limitations, https://arxiv.org/html/1703.10135v2#S6 ; local results/fixes/ contains actual verification outputs.
