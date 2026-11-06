# Original Tacotron architecture

```text
                          LJSPEECH
                              |
                +-------------+--------------+
                |                            |
               TEXT                         AUDIO
                |                            |
        English cleaning                Waveform/resampling
                |                            |
        Character encoding               Preemphasis + STFT
                |                            |
           Embedding                 Mel + linear targets
                |                            |
         Encoder prenet                      |
                |                            |
          Encoder CBHG                       |
                |                            |
       Bahdanau attention                    |
                |                            |
        GRU decoder/prenet                   |
                |                            |
       Predicted mel spectrogram --------- L1 loss
                |
       Post-processing CBHG
                |
       Predicted linear spectrogram ------ Weighted L1 loss
                |
           Griffin-Lim
                |
      Inverse preemphasis / WAV
```

The original embedding has 256 features. Encoder and post-processing CBHG output depths are 256. The attention and decoder GRUs also use depth 256. The reduction factor is five mel frames per decoder step. No layers, dimensions or attention mechanisms are replaced.

## CBHG

CBHG means Convolution Bank, Highway network and bidirectional GRU. The convolution bank applies filters with several temporal widths to detect local patterns. The encoder uses widths 1 to 16 and the post-processing bank uses 1 to 8. Max pooling and projection convolutions combine the features. A residual connection preserves the input information.

Highway layers use learned gates to balance transformed features and carried input. They let information pass through several layers. The bidirectional gated recurrent unit reads a sequence in both directions, adding preceding and following context. The full text is available before synthesis, so using both directions in the encoder is appropriate.

## Attention and decoder

Bahdanau attention computes a soft distribution over encoder positions. A context vector helps the decoder use the relevant characters while producing acoustic frames. The GRU decoder carries information between time steps. During training, teacher forcing supplies recorded mel frames to the next step. During inference, the decoder supplies its own predicted frames, making generation autoregressive. The original test helper has a zero-frame stopping condition and also enforces `max_iters`.

## Targets and loss

`models/tacotron.py:add_loss` computes mean absolute mel error and a linear error consisting of half the all-frequency MAE and half the MAE below approximately 3 kHz. Total loss is their sum. Adam updates the parameters. Existing `tf.clip_by_global_norm(..., 1.0)` limits the total gradient norm. TensorBoard records loss, learning rate, outputs and gradient statistics.

## Waveform reconstruction

The model predicts normalized magnitude spectrograms rather than waveform phase. Griffin-Lim iteratively estimates phase using STFT/inverse STFT. The original TensorFlow implementation then yields a waveform, and inverse preemphasis and PCM WAV writing finish inference. The existing default uses 60 reconstruction iterations.
