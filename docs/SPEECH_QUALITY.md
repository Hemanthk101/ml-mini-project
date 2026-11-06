# Speech quality correction

The problematic app request was a long paragraph about GPU execution and timing metrics. Its original audio ended at 12.5375 seconds, corresponding to the original decoder's 200-step / 1000-frame inference ceiling. Attention jumped over parts of the long text. Independent Whisper recognition recovered repeated, missing and substituted words, while both short test sentences were recognized correctly through the original TensorFlow and librosa reconstruction paths.

The app now cleans numbers and expands common initialisms (GPU, CPU, RTF, TTS, ML, AI, F1) into spoken letter names, splits normalized text into phrases of at most 80 characters where possible, synthesizes every phrase separately, and joins the segments with 180-ms pauses. Words are preserved rather than discarding text at the old whole-request duration limit. Each model call still uses the original inference settings and checkpoint.

Reconstruction reuses the upstream `audio.inv_spectrogram` librosa Griffin-Lim implementation with the original 60 iterations and power 1.5. This uses the same STFT/ISTFT conventions as preprocessing and applies inverse preemphasis exactly once. Silence detection uses absolute waveform amplitude; per-phrase normalization and PCM writing retain a 0.92 peak ceiling. No neural vocoder, new acoustic architecture or different training dataset was introduced.

`Synthesizer.load` adds an optional `build_waveform=False` flag, used by the app to fetch acoustic tensors without building the unused TensorFlow waveform graph. The default CLI behavior is unchanged. Neural Tacotron inference runs on GPU; this app's Griffin-Lim reconstruction runs on CPU. Recorded synthesis time and RTF include that reconstruction. Model-execution time now sums the neural TensorFlow session calls across phrases, excluding CPU reconstruction.

## Actual evidence

- `results/quality/asr_comparison.json`: independent recognition of short sentences and the original paragraph. Recognizer is Whisper tiny.en, without the expected text as a prompt.
- `results/quality/improved9.wav`: the same paragraph synthesized as eight phrases; duration 24.6821 seconds.
- `results/quality/improved9.json`: actual segment texts, durations, GPU status and timing measurements. This validation ran under memory pressure and is not a clean speed benchmark.
- `results/quality/asr_before_after.json`: independent recognition of the old and improved paragraph, from the completed final comparison.
- `tests/test_speech_quality.py`: five checks covering preserved words, phrase lengths, initialisms, negative amplitudes, silent/invalid data and assembly headroom.

ASR word matches are an intelligibility proxy for these examples. They are not a human MOS or a claim of studio-quality speech. The underlying model still uses classic Tacotron and Griffin-Lim. The interface remains a text box and Generate button, with audio after generation.

On this paragraph, raw normalized word error fell from 73.913% to 6.522%, F1 rose from 0.32 to 0.9348, and deletions fell from 17 to zero. The recognizer still substituted GPU with GPE and Queue wait with Ku weight. These are actual recognizer outputs, not corrected transcripts, and this is one example rather than a general model benchmark.
