"""Held-out teacher-forced acoustic evaluation; never label low loss as good speech."""
import argparse, json, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from models import create_model
from hparams import hparams
from text import text_to_sequence
from reproducibility import set_seed
from gpu_runtime import require_gpu, traced_run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--input',default='training/validation_split.txt');p.add_argument('--training_input',default='training/train_split.txt');p.add_argument('--output',default='results/fixes/validation.json');a=p.parse_args()
    source=Path(a.input); rows=[x.split('|',3) for x in source.read_text(encoding='utf-8').splitlines() if x.strip()]
    train_rows=[x.split('|',3) for x in Path(a.training_input).read_text(encoding='utf-8').splitlines() if x.strip()]
    train_ids={x[0] for x in train_rows}; train_text={' '.join(x[3].casefold().split()) for x in train_rows}
    if any(x[0] in train_ids or ' '.join(x[3].casefold().split()) in train_text for x in rows):
        raise ValueError('Training/evaluation overlap detected.')
    require_gpu();set_seed(2026)
    inputs=tf.placeholder(tf.int32,[1,None]);lengths=tf.placeholder(tf.int32,[1]);mel=tf.placeholder(tf.float32,[1,None,hparams.num_mels]);linear=tf.placeholder(tf.float32,[1,None,hparams.num_freq])
    with tf.variable_scope('model'):
        model=create_model('tacotron',hparams)
        model.initialize(inputs,lengths,mel,linear,is_training=False)
        model.add_loss()
    if tf.get_collection(tf.GraphKeys.UPDATE_OPS):
        raise RuntimeError('Evaluation must not update batch-normalization statistics.')
    saver=tf.train.Saver();config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2);config.gpu_options.allow_growth=True
    measurements=[]; started=time.perf_counter()
    with tf.Session(config=config) as sess:
        saver.restore(sess,a.checkpoint)
        for index,row in enumerate(rows):
            seq=np.asarray(text_to_sequence(row[3],hparams.cleaners.split(',')),dtype=np.int32)
            m=np.load(str(source.parent/row[1]),allow_pickle=False);l=np.load(str(source.parent/row[0]),allow_pickle=False)
            frames=len(m);padded=((frames+1+hparams.outputs_per_step-1)//hparams.outputs_per_step)*hparams.outputs_per_step
            feed={inputs:[seq],lengths:[len(seq)],mel:[np.pad(m,[(0,padded-frames),(0,0)])],linear:[np.pad(l,[(0,padded-frames),(0,0)])]}
            fetches=[model.loss,model.mel_outputs,model.linear_outputs]
            if index==0:
                (loss,pm,pl),evidence=traced_run(sess,fetches,feed,str(Path(a.output).with_suffix('.device.json')))
            else:loss,pm,pl=sess.run(fetches,feed_dict=feed)
            ml=float(np.mean(np.abs(pm[0,:frames]-m)));d=np.abs(pl[0,:frames]-l);priority=int(3000/(hparams.sample_rate*.5)*hparams.num_freq)
            ll=float(.5*np.mean(d)+.5*np.mean(d[:,:priority]))
            measurements.append({'file':row[0],'text':row[3],'frames':frames,'original_padded_loss':float(loss),'valid_frame_mel_l1':ml,'valid_frame_linear_l1':ll,'valid_frame_loss':ml+ll})
    report={'checkpoint':a.checkpoint,'examples':measurements,'mean_valid_frame_loss':float(np.mean([x['valid_frame_loss'] for x in measurements])),'evaluation_seconds':time.perf_counter()-started,'gpu_verified':evidence['verified_gpu_compute'],'teacher_forced':True,'dropout_disabled':True,'batchnorm_updates_disabled':True,'training_overlap':False,'speech_quality':'Not inferred from acoustic loss; run benchmark_quality.py and human listening separately.','scope':'New-run validation only; LJSpeech-pretrained checkpoints may have dataset leakage.'}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
