# Complete ML workflow

## 1. Dataset loading

**WHAT:** Read LJSpeech normalized transcripts and corresponding WAV files.

**WHY:** Training needs correctly paired observations.

**ML CONCEPT:** Supervised learning.

## 2. Text preprocessing

**WHAT:** Expand numbers, transliterate and normalize whitespace/case with the original English cleaners.

**WHY:** Equivalent written forms should have consistent input symbols.

**ML CONCEPT:** Feature preparation.

## 3. Audio preprocessing

**WHAT:** Resample to 20 kHz, apply preemphasis, STFT, mel filtering and magnitude normalization.

**WHY:** Acoustic targets are easier to learn than raw waveform samples.

**ML CONCEPT:** Signal processing and target representation.

## 4. Character encoding

**WHAT:** Map permitted characters and optional ARPAbet symbols to integer IDs and append EOS.

**WHY:** The network needs numeric inputs with an explicit sequence boundary.

**ML CONCEPT:** Categorical encoding.

## 5. Embedding

**WHAT:** Look up a trainable dense vector for each symbol ID.

**WHY:** Integer IDs alone do not express useful relationships between symbols.

**ML CONCEPT:** Representation learning.

## 6. Encoder

**WHAT:** Apply the prenet and encoder CBHG to embedded input sequences.

**WHY:** Local character patterns and sentence context inform pronunciation.

**ML CONCEPT:** Sequence learning and bidirectional recurrence.

## 7. Attention

**WHAT:** Compute Bahdanau weights and an encoder context for each decoder step.

**WHY:** Text and speech have different lengths and no supplied frame alignment.

**ML CONCEPT:** Learned soft alignment.

## 8. Decoder

**WHAT:** Use GRUs to predict five mel frames per recurrent step.

**WHY:** Speech depends on both text context and earlier acoustic frames.

**ML CONCEPT:** Encoder-decoder sequence-to-sequence learning.

## 9. Spectrogram prediction

**WHAT:** Reshape decoder frames and use post-processing CBHG plus linear projection.

**WHY:** Mel features guide synthesis while a linear magnitude spectrum supports reconstruction.

**ML CONCEPT:** Continuous multivariate regression.

## 10. Loss calculation

**WHAT:** Add mel MAE to weighted linear MAE, using recorded speech targets.

**WHY:** The optimizer needs a measurable discrepancy between prediction and target.

**ML CONCEPT:** Loss function / supervised objective.

## 11. Backpropagation

**WHAT:** TensorFlow differentiates the loss through the graph and recurrent time steps.

**WHY:** Each parameter needs a signal indicating how it contributes to error.

**ML CONCEPT:** Chain rule and backpropagation through time.

## 12. Parameter update

**WHAT:** Clip gradients by global norm 1.0 and use the original Adam optimizer.

**WHY:** Stable updates adjust parameters in a direction that tends to reduce loss.

**ML CONCEPT:** Gradient descent with adaptive moments.

## 13. Checkpoint saving

**WHAT:** Save TensorFlow variable state and global step; record path and run settings.

**WHY:** A process can resume or restore learned parameters later.

**ML CONCEPT:** Experiment management and reproducibility.

## 14. Inference

**WHAT:** Clean unseen text, restore a compatible checkpoint and autoregressively predict frames.

**WHY:** Generation must work without target audio or teacher forcing.

**ML CONCEPT:** Application of learned parameters / inference.

## 15. Griffin-Lim reconstruction

**WHAT:** Denormalize the predicted linear magnitude and iteratively estimate phase.

**WHY:** Magnitude alone is insufficient for inverse STFT waveform recovery.

**ML CONCEPT:** Iterative signal reconstruction.

## 16. WAV generation

**WHAT:** Invert preemphasis, trim the original endpoint condition and save signed 16-bit PCM.

**WHY:** A standard audio file makes the result playable and inspectable.

**ML CONCEPT:** End-to-end pipeline output.

## Concepts used in training

A recurrent neural network (RNN) carries a hidden state through a sequence. A GRU uses gates to decide which information to retain and update. Teacher forcing uses the previous target frame during training, whereas autoregressive decoding uses a previous prediction during inference.

An epoch means one pass through all training examples. Batch size is the number of examples in one optimizer update. A step is one update and is not an epoch. The original feeder shuffles and repeats examples in queued groups, so we report measured steps rather than claim an epoch count.

Learning rate controls update scale. The original Noam-style schedule warms up over 4,000 steps. At the beginning its actual value is much smaller than the `initial_learning_rate=0.002` coefficient. The CSV records the learning rate used for each observed update.

Overfitting means learning training-specific patterns that fail to generalize. Validation data can expose this when validation error worsens while training error improves. Test data provide a final separate assessment. Neither is measured in this smoke test.

A spectrogram represents signal magnitude over time and frequency. STFT applies Fourier analysis to short overlapping windows so changing sounds remain localized in time. Mel spectrograms combine frequencies according to a perceptual scale. Sequence-to-sequence learning maps a variable-length text sequence into a variable-length sequence of acoustic frames.
