# GPU setup and verified execution

GPU: NVIDIA GeForce GTX 1650 (4 GB VRAM). Runtime: TensorFlow 1.15.0, Python 3.7.9, local CUDA 10.0.130 and cuDNN 7.6.5. The original Tacotron and LJSpeech pipeline remain unchanged.

## Setup and commands

From the project directory in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup-gpu.ps1
.\run-gpu.ps1 gpu_check.py
.\run-gpu.ps1 train.py --base_dir . --input training/train.txt --name gpu-verified --max_steps 3 --seed 2026 --summary_interval 1 --checkpoint_interval 1000 --hparams "batch_size=1" --results_dir results/gpu-verified
.\run.ps1 plot_results.py --results_dir results/gpu-verified
.\run-gpu.ps1 demo.py --text "Machine learning can convert written text into speech." --output output-gpu.wav
.\run-gpu.ps1 demo.py --samples
```

On a clean copy run `setup.ps1`, download assets and preprocess as documented in README before training or synthesis. GPU runtime setup verifies SHA-256 checksums of Anaconda's CUDA/cuDNN packages, extracts DLLs into `.runtime/gpu/Library/bin`, and runs a strict GPU calculation test. Driver installation and global PATH remain unchanged. `run-gpu.ps1` adds DLLs to PATH for its child process only and restores environment variables afterwards.

The launcher requires a GPU, enables memory growth and disables convolution autotuning. Batch size one is the verified setting for this 4 GB GPU. An initial batch-size-two run also finished, but autotuning caused memory-pressure warnings and much longer startup, so the recommended command uses batch size one. Shape, sequence, and audio hyperparameters remain the originals.

## Verified results

- Strict `/gpu:0` matrix multiplication and cuDNN convolution/gradient passed with soft placement disabled. See `results/gpu_check.json`.
- Fresh GPU run: 3 updates, 8 LJSpeech samples, batch size 1.
- Observed starting loss: 0.939611; ending loss: 0.933286.
- Learning rate used: 4.99999999e-07 to 1.49999994e-06.
- Measured graph construction/training/checkpoint time: 157.45 seconds. This includes a traced first update and TensorBoard summaries and is not a controlled speed benchmark.
- Saved checkpoint: `logs-gpu-verified/model.ckpt-3`.
- Training trace recorded 12875 GPU convolution/matrix operation executions. These are TensorFlow operation execution traces, not CUPTI hardware profiles.
- GPU inference generated `output-gpu.wav` and all three required `samples/sample_N.wav` files. Every inference has a `.device.json` trace and GPU evidence in its metadata.
- WAVs verified as nonempty mono 16-bit PCM at 20 kHz, with matching durations and SHA-256 hashes.
- This historical GPU run used the original model and transformations. The later app adds an optional attention-based stopping policy without changing learned model layers or checkpoint variables; see `FEASIBLE_FIXES.md`.

CSV, JSON run settings and graph: `results/gpu-verified/`. Overall evidence: `results/gpu-verified/verification.json`.

## Scope of the verification

The three-step fresh model validates execution, not learned speech quality. Inference uses the separate upstream pretrained checkpoint. Original CPU report/results remain historical evidence of the earlier run. GPU follow-up measurements live separately and should not be mixed with the CPU report's loss values. A GPU does not itself change the architecture or establish a quality improvement. No MOS or held-out evaluation is claimed.

Legacy TensorFlow emits deprecation messages. An initial full profiler attempt warned that optional CUPTI was absent; final verification uses software operation traces and requires actual GPU convolution/matrix operations. The original model may execute appropriate host/queue/audio operations on CPU while its neural computation runs on GPU.

References: https://www.tensorflow.org/install/source ; package provenance and hashes are in `results/gpu_runtime_packages.json`.
