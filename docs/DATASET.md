# Dataset: LJSpeech

LJSpeech 1.1 is a public-domain dataset of 13,100 short English recordings from one speaker reading seven nonfiction books. Each WAV has an original and normalized transcript. The official source is https://keithito.com/LJ-Speech-Dataset/ and the archive is https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2.

This mini-project uses eight actual pairs from that official archive. `results/dataset_manifest.json` records IDs, download source and WAV hashes. Selection takes the first eight WAV members in archive order. This is a small convenience sample, not a representative evaluation set. Original recordings and complete transcripts are retained without cropping or invented labels.

Text is the model input. Recorded speech supplies the target mel and linear spectrograms. Known input-target pairs make this supervised learning. The targets contain continuous acoustic magnitudes, so the central prediction task is regression.

Text -> English cleaning -> character IDs -> learned embeddings.

Audio -> mono waveform resampled to 20,000 Hz -> preemphasis -> STFT -> linear and mel magnitude spectrograms -> decibel conversion -> normalized targets -> NumPy arrays.

The native dataset WAVs use 22,050 Hz. The repository resamples to its existing 20,000 Hz setting, which is also used for generated audio. Defaults use 80 mel bins, 1,025 linear bins, a 50 ms window and 12.5 ms hop. Processing uses the original `datasets/ljspeech.py` and `util/audio.py` routines.

`metadata.csv` has `utterance_id|original_text|normalized_text`. Preprocessed `train.txt` has `linear_array|mel_array|frame_count|normalized_text`. The feeder pads and buckets samples by duration and cleans text at batch loading time.

## Train, validation and test

Training data update the weights. Validation data help choose hyperparameters and detect overfitting. Test data estimate generalization after choices are fixed. This eight-pair smoke test uses all eight for training and performs no held-out evaluation. Do not describe its loss as validation or test performance. A longer experiment should split utterance IDs before preprocessing and keep validation/test utterances out of training.

## Setup

Small subset: `.\run.ps1 download_assets.py --subset 8` in a clean project directory. This streams the official archive and stops after matching real WAV/transcript pairs. It refuses to overwrite existing metadata.

For full LJSpeech, download the official archive separately, extract it so that `LJSpeech-1.1/metadata.csv` and `LJSpeech-1.1/wavs/` are directly inside the project, then run the same preprocessing command in the README. Do not upload the dataset to GitHub.
