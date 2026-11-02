# Improved Tacotron Text-to-Speech Synthesis

## Problem Statement

Convert written English text into a speech WAV through a learned mapping from characters to acoustic features.

## Objective

Demonstrate the original Tacotron architecture with reproducible GPU execution, bounded-memory training, separate validation, and practical inference corrections. Five diagnostic speech cases pass the documented ASR gates; this is limited evidence of intelligibility, not a general speech-quality or human listening score. Current fixes and measurements are in [FEASIBLE_FIXES.md](docs/FEASIBLE_FIXES.md).

## Original Reference

This project reuses [Keith Ito's Tacotron](https://github.com/keithito/tacotron), commit `d26c763342518d4e432e9c4036a1aff3b4fdaa1e`. `README.upstream.md` preserves its instructions and `LICENSE` preserves its MIT license. The supplied `15.pdf` is *End-to-End Text to Speech Synthesis*, Xiao Wang, Yahan Yang and Ye Li, CS 229, Autumn 2018. It compares several models and cites this Tacotron implementation. Its reported scores are not results of this project. The original architecture paper is [Wang et al., 2017](https://arxiv.org/abs/1703.10135).

## Dataset

LJSpeech only. Eight genuine complete WAV/transcript pairs from the official LJSpeech 1.1 archive form the local smoke-test subset. See `docs/DATASET.md` and `results/dataset_manifest.json`. The supplied report also uses LJSpeech. Original Google Tacotron training data should not be confused with this open implementation's dataset.

## Why This Is an ML Problem

Recorded speech supplies supervised targets. Tacotron learns to regress continuous mel/linear spectral values from a variable-length character sequence. Pronunciation and alignment come from learned parameters rather than a hand-coded text-to-waveform rule.

## Architecture

Text -> cleaning/IDs -> embedding -> encoder CBHG -> attention -> decoder -> mel spectrogram -> post-processing CBHG -> linear spectrogram -> Griffin-Lim -> WAV. `docs/ARCHITECTURE.md` describes the original layers and loss. No model source in `models/` is changed.

## ML Workflow

See `docs/ML_WORKFLOW.md` for all 16 stages, each with WHAT, WHY and the ML concept.

## ML Concepts Used

Supervised learning, regression, train/validation/test data, encoding, embeddings, spectrograms, STFT, sequence-to-sequence learning, encoder-decoder, RNN/GRU, attention, autoregressive decoding, teacher forcing, loss, backpropagation, Adam, epoch, batch size, learning rate, overfitting, inference and Griffin-Lim.

## Improvements Made

- Fixed Python, NumPy and TensorFlow seeds, configurable with `--seed`.
- Bounded training with `--max_steps`, final checkpoint and clean queue shutdown.
- Observed step/loss/learning-rate CSV and run settings/time/checkpoint JSON.
- Original TensorBoard reused. Existing norm-1 gradient clipping retained without duplication.
- CLI demo with checkpoint validation, three required examples, spectrogram/alignment plots and WAV metadata/hashes.
- Isolated Windows Python 3.7 / TF 1.15 setup and genuine LJSpeech subset download.

Fixed seeds reduce randomness but do not guarantee identical results across hardware, drivers or concurrent execution. This project uses a historical runtime because TF 2 removed `tf.contrib`, which the original model requires.

## Project Structure

`models/`, `text/`, `datasets/`, `util/`, `hparams.py`, `synthesizer.py`, `eval.py`: original implementation. `train.py` and the feeder contain small execution changes. `demo.py`, `download_assets.py`, `reproducibility.py`, `plot_results.py`, `setup.ps1`, `run.ps1`: additions. `docs/`: explanations. `results/`: actual evidence. `samples/`: generated WAV/plots/metadata. `deliverables/`: report, slide content, viva and demo script. `CHANGES.md`: exact change inventory.

## Installation

GPU support has now been added for the GTX 1650. See `docs/GPU_SETUP.md` for setup, verified results and GPU commands. The original CPU measurements below remain historical evidence of the initial run. Use `run-gpu.ps1` to require CUDA and verify model operations on the GPU. CUDA libraries are local to `.runtime/gpu/`; the graphics driver and system PATH are unchanged.

From a fresh PowerShell terminal on Windows x64:

```powershell
Set-Location 'D:\Codex\2026-10-05\rea\outputs\project'
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The isolated runtime lives in `.runtime/python37/`. It does not replace the system Python. `run.ps1` forwards its arguments to that interpreter. Do not install current TensorFlow or the upstream historical requirements into Python 3.11. For another compatible Python 3.7 environment, use `python -m pip install -r requirements-mini.txt` and replace `.\run.ps1` with `python` in the commands below.

## Dataset Setup

The eight-pair subset is already prepared locally. On a clean clone, run:

```powershell
.\run.ps1 download_assets.py --subset 8
.\run.ps1 download_assets.py --checkpoint
```

The subset command refuses to overwrite existing metadata. The checkpoint command skips existing complete files. The source checkpoint is the original repository's published `tacotron-20180906/model.ckpt` and is separate from our fresh smoke-test model. See `docs/DATASET.md` for downloading full LJSpeech separately.

## Preprocessing

```powershell
.\run.ps1 preprocess.py --dataset ljspeech --base_dir . --output training --num_workers 1
```

This executes upstream preprocessing and creates paired NumPy targets plus `training/train.txt`. Text cleaning runs when batches are loaded. The originals remain untouched.

## Training

```powershell
.\run.ps1 train.py --base_dir . --input training/train.txt --name smoke --max_steps 3 --seed 2026 --summary_interval 1 --checkpoint_interval 1000 --hparams "batch_size=2" --results_dir results
.\run.ps1 plot_results.py
```

The command starts a fresh model with unchanged default architecture. Three updates constitute a smoke test, not a trained voice. `logs-smoke/model.ckpt-3` is the expected saved prefix after three successful updates. Re-running overwrites the CSV/summary and appends to the log. `--restore_step 3` resumes a saved checkpoint. `--max_steps` counts additional updates in this invocation, not a global stopping step.

For TensorBoard, use `.\run.ps1 -m tensorboard.main --logdir logs-smoke`.

## Running the Demo

```powershell
.\run.ps1 demo.py --text "Machine learning can convert written text into speech." --output output.wav
.\run.ps1 demo.py --samples
```

Default inference loads the compatible upstream pretrained checkpoint. To inspect the untrained smoke model, explicitly use `--checkpoint logs-smoke/model.ckpt-3 --hparams "batch_size=2"`; meaningful speech requires a sufficiently trained checkpoint. Pass identical architecture/audio hyperparameters if you change them during a future run.

## Generated Outputs

`output.wav` is the default demo result. Three required examples live at `samples/sample_1.wav`, `sample_2.wav`, `sample_3.wav`. Matching PNGs contain predicted mel spectrograms and attention, and matching JSON files record text, checkpoint, elapsed time, shapes, sample rate and SHA-256. The original published checkpoint restored successfully and generated 3 required WAV examples, with spectrogram/alignment images and JSON metadata. No listening study or MOS evaluation was performed.

## Evaluation

Experiment question: "How does the Tacotron training loss behave during an initial short training run?"

A fresh model completed 3 Adam updates on 8 LJSpeech pairs with batch size 2. Observed first-step loss was 0.918565 and last-step loss was 0.873978. The first observed learning rate was 4.99999999e-07. Graph construction, training and checkpoint saving took 48.40 seconds.

Measured rows are in `results/experiment_results.csv`; settings and checkpoint are in `results/experiment_summary.json`; the graph is `results/training_loss.png`. These are training-batch losses, not a fixed validation curve or proof of intelligible speech. Batch composition and dropout affect the values. Inference uses a separate published checkpoint. No MOS, accuracy, held-out error or improvement against a baseline is claimed.

## Limitations

Due to the mini-project execution-time constraint, the training run is used to validate the complete Tacotron ML pipeline rather than reproduce the full-scale training performed in the original research.

Eight samples and three updates cannot establish generalization or reproduce the original Tacotron paper's speech quality. The latest fresh GPU run uses a disjoint 6/1/1 split and evaluates one validation example; the test partition remains unused. Griffin-Lim can introduce artifacts. The app uses optional sustained-EOS stopping and bounded retries while retaining `max_iters` as a hard ceiling. Windows GPU training and inference are verified. The pretrained voice's complete training history is not independently reproduced here.

## Future Improvements

Train longer on a larger LJSpeech subset, expand held-out evaluation, assess intelligibility with a documented listening study, and compare execution settings under fixed evaluation conditions while retaining the original architecture.

## Team Contribution

Member names were not supplied. Before submission, replace this section with actual contributions. Suggested responsibilities are environment/preprocessing, experiment/inference, and documentation/presentation. Do not credit the supplied paper's authors as members of this new project.

## References

- Xiao Wang, Yahan Yang and Ye Li. *End-to-End Text to Speech Synthesis*, CS 229 Project Report, Autumn 2018 (supplied `15.pdf`).
- Yuxuan Wang et al. *Tacotron: Towards End-to-End Speech Synthesis*, 2017. https://arxiv.org/abs/1703.10135
- Keith Ito. Tacotron implementation. https://github.com/keithito/tacotron
- Keith Ito and Linda Johnson. LJSpeech, 2017. https://keithito.com/LJ-Speech-Dataset/

## Private GitHub preparation

Keep the included license and upstream attribution. `.gitignore` excludes dataset, processed arrays, checkpoint/runtime/log/cache files and generated media. It retains the small measured CSV, summaries, graph and submission documents. `CHANGES.md` lists project modifications. Create a private remote repository and push only the source/docs/evidence files after reviewing `git status`; no remote was created or pushed automatically.
# GPU Speech Lab app

Run `powershell -ExecutionPolicy Bypass -File .\start-app.ps1` from this folder and open **http://127.0.0.1:8765**. The interface contains only a text box and Generate button; an audio player appears when speech is ready. GPU synthesis still records timing metrics, WAVs and diagnostics in `app_runs/`. Metrics and transcript evaluation remain available through the local API. See [metric definitions and scope](docs/APP_METRICS.md).

The app splits long text into short phrases, expands technical initialisms, and reuses upstream librosa Griffin-Lim. It now stops on sustained EOS attention, retries unfinished phrases within a fixed limit, and uses punctuation-aware pauses. [Speech-quality correction and initial comparison](docs/SPEECH_QUALITY.md). [Feasible fixes, evaluation gates and memory controls](docs/FEASIBLE_FIXES.md).

Current submission artifacts: [write-up PDF](deliverables/Mini_Project_Writeup.pdf), [editable Word report](deliverables/Mini_Project_Writeup.docx), and [updated eight-slide PowerPoint](deliverables/Tacotron_Presentation_Final.pptx). Earlier results above describe the initial smoke run; the latest training, validation and speech evidence is under `results/fixes/`.
