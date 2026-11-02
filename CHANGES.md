# Exact source and deliverable inventory

GPU follow-up adds `download_gpu.py`, `gpu_runtime.py`, `gpu_check.py`, `setup-gpu.ps1`, `run-gpu.ps1` and `docs/GPU_SETUP.md`. `train.py` and `demo.py` now support required-GPU checks and operation execution traces, while `plot_results.py` accepts `--results_dir`. GPU evidence is separate from the original CPU run. Core model files remain unchanged.

## Modified upstream files

- `train.py`: seed, bounded updates, observed CSV, summary JSON, final checkpoint, nonzero failure exit and clean feeder shutdown.
- `datasets/datafeeder.py`: queue cancellation and shutdown handling.
- `.gitignore`: datasets, arrays, checkpoints, runtime, logs and large generated artifacts excluded.
- `README.md`: project instructions and actual results; original saved as `README.upstream.md`.

## Added source/configuration

`reproducibility.py`, `demo.py`, `download_assets.py`, `plot_results.py`, `requirements-mini.txt`, `requirements-lock.txt`, `setup.ps1`, `run.ps1`, `PROJECT_UNDERSTANDING.md`, `CHANGES.md`.

`FILE_MANIFEST.md` enumerates every modified and added deliverable/evidence file. The source-and-submission archive is `../Improved_Tacotron_Mini_Project.zip`; it includes generated samples and excludes the dataset, runtime and model shards.

## Added documentation and submission files

`docs/DATASET.md`, `docs/ARCHITECTURE.md`, `docs/ML_WORKFLOW.md`, `docs/VERIFICATION.md`.

`deliverables/Mini_Project_Writeup.md`, `deliverables/Mini_Project_Writeup.pdf`, `deliverables/PRESENTATION_CONTENT.md`, `deliverables/VIVA_QA.md`, `deliverables/DEMO_SCRIPT.md`.

## Runtime artifacts

`results/dataset_manifest.json`, `results/preprocessing_summary.json`, `results/experiment_results.csv`, `results/experiment_summary.json`, `results/training_loss.png`, `results/environment.txt`, `results/verification.json` and execution logs.

`samples/sample_1.wav`, `sample_2.wav`, `sample_3.wav`, matching `.png` and `.json`; `output.wav`, `output.png`, `output.json`.

Ignored local resources: `.runtime/python37/`, `LJSpeech-1.1/`, `training/`, `checkpoints/tacotron-20180906/`, `logs-smoke/`.

That initial verification retained the original model source. Later changes are listed below; `git diff` is the authoritative comparison. Learned layer topology, checkpoint variable names, `hparams.py`, text and audio transformation routines remain the originals.

## Local app

Added app.py, app_metrics.py, start-app.ps1, app_static/ and metric/API tests. The final interface is a single text box and Generate button with conditional audio playback. GPU timing, failure counts and transcript evaluation remain in the API and saved reports.


## Speech quality correction

The app normalizes common technical initialisms, splits cleaned text into short phrases, reuses the original librosa Griffin-Lim reconstruction, trims using absolute amplitude and assembles every phrase with brief pauses and PCM headroom. Core model layers, checkpoint weights and training hyperparameters are unchanged. synthesizer.py adds an optional build_waveform flag so the app can fetch acoustic predictions without building an unused TensorFlow waveform graph; its original default behavior is retained.

## Feasible fixes follow-up

Added optional sustained-EOS inference stopping to `models/helpers.py` and `models/tacotron.py`, preserving pretrained checkpoint compatibility. The app uses reusable attention history, bounded smaller-phrase retries, punctuation-aware pauses and click-reducing ramps. `synthesizer.py` uses bounded CPU threads, GPU memory growth and direct restore. Training evaluation can disable dropout and batch-normalization updates independently of teacher forcing.

Added `batching.py`, `split_dataset.py`, `evaluate_checkpoint.py`, `normalize_text.py`, `benchmark_quality.py` and regression tests. `datasets/datafeeder.py` uses configurable smaller groups/queues, an optional padded-frame budget and memory-mapped targets. Sequential LJSpeech preprocessing avoids a child TensorFlow process. Actual outputs and quality gates are under `results/fixes/`; methods, scope and commands are in `docs/FEASIBLE_FIXES.md`.
