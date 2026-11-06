# Initial implementation verification (historical)

This records the initial unchanged-model implementation, before the GPU app and optional inference/memory fixes. It is not a verification of the current source. Current checks and their scope are in [FEASIBLE_FIXES.md](FEASIBLE_FIXES.md) and `../results/fixes/verification.json`.

Verified original upstream commit: `d26c763342518d4e432e9c4036a1aff3b4fdaa1e`.

- PASS: entire original model folder, hyperparameters, text/audio transforms, LJSpeech preprocessing and synthesizer match upstream.
- PASS: only LJSpeech used, eight WAV/transcript pairs from the official archive, confirmed by a second downloader run and SHA-256 comparison.
- PASS: original preprocessing executed with one worker, eight samples, 3,436 frames, expected finite mel/linear arrays.
- PASS: 15 original text/number/CMUDict tests.
- PASS: final training source completed three fresh Adam updates, batch size two, recorded loss/learning rate and saved checkpoint.
- PASS: TensorBoard events written; its CLI help command exits successfully after compatibility pins.
- PASS: published original Tacotron checkpoint restored in original `Synthesizer.load`; three required WAVs and the default demo WAV were generated.
- PASS: WAVs are nonempty mono 16-bit 20 kHz PCM; hashes, durations, predicted spectral shapes and plots recorded.
- PASS: all documentation/deliverables present, report exactly two pages, slide content exactly eight slides. Report pages and graph visually inspected.
- PASS: source syntax checked; dataset, model shards, runtime, arrays and generated WAVs ignored by Git.

Actual loss: 0.918565 to 0.873978. Run duration: 48.40 seconds. These values describe the fresh smoke model. All generated sample speech uses the separate upstream published checkpoint.

## Not performed

Full LJSpeech training, held-out validation/test evaluation, human listening/MOS study, independent replication of the pretrained model's training history, and GitHub creation/push. Only Windows x64 CPU execution was verified. Team member names were not supplied and must be added before submission. No PPTX was requested: the presentation deliverable is eight-slide Markdown content.

`results/verification.json` contains machine-readable evidence. `CHANGES.md` lists exact additions and modifications. README contains fresh-terminal commands. Original runtime emits deprecation/GPU-library warnings and falls back to CPU; successful exit codes and generated artifacts verify execution.

## GPU follow-up

GPU training and synthesis now also pass on the GTX 1650. See `GPU_SETUP.md` and `results/gpu-verified/verification.json` for separate measured results and operation traces. The CPU measurements above remain the original run.
