"""Length bucketing with an explicit padded-frame budget; no dropped examples."""
import math


def make_batches(examples, batch_size, outputs_per_step, max_batch_frames=0):
    if batch_size < 1 or outputs_per_step < 1 or max_batch_frames < 0:
        raise ValueError('Invalid batch size, reduction factor or frame budget.')
    batches, batch = [], []
    for example in sorted(examples, key=lambda x: x[-1]):
        padded = int(math.ceil((example[-1]+1)/outputs_per_step))*outputs_per_step
        if max_batch_frames and padded > max_batch_frames:
            raise ValueError('Utterance exceeds the padded-frame budget; increase it or filter explicitly.')
        if batch and (len(batch) >= batch_size or
                      (max_batch_frames and (len(batch)+1)*padded > max_batch_frames)):
            batches.append(batch)
            batch = []
        batch.append(example)
    if batch:
        batches.append(batch)
    return batches
