"""Independent ASR quality gate over saved WAVs; run after releasing the TTS GPU process.

Uses an existing PyTorch/Transformers environment and a local recognition checkpoint.
No reference prompt is passed to ASR. This is not a human listening score.
"""
import argparse, json, subprocess, sys
from pathlib import Path
from math import gcd
import os
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_TORCH','1')
import torch
torch.set_num_threads(2)
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from scipy.signal import resample_poly
from app_metrics import transcript_metrics, aggregate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/fixes');p.add_argument('--model_dir',required=True);p.add_argument('--max_wer',type=float,default=.2)
    a=p.parse_args(); root=Path(a.input);project=Path(__file__).resolve().parent
    processor=WhisperProcessor.from_pretrained(a.model_dir,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(a.model_dir,local_files_only=True,attn_implementation='eager');model.eval()
    report={'recognizer':'Whisper tiny.en, CPU, no reference prompt','scope':'Five diagnostic inputs; ASR proxy, not MOS or a held-out population benchmark. Spoken normalization applies equally to reference and transcript.','threshold':a.max_wer,'examples':[]}
    for record in json.loads((root/'synthesis_suite.json').read_text()):
        path=root/(record['case']+'.wav');wav,sr=sf.read(str(path),dtype='float32')
        if sr!=16000: factor=gcd(sr,16000);wav=resample_poly(wav,16000//factor,sr//factor)
        batch=processor(wav,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        with torch.inference_mode():ids=model.generate(batch.input_features,attention_mask=batch.attention_mask,max_new_tokens=160)
        transcript=processor.batch_decode(ids,skip_special_tokens=True)[0].strip()
        raw=transcript_metrics(record['text'],transcript)
        cleaned=json.loads(subprocess.check_output([str(project/'.runtime/python37/python.exe'),str(project/'normalize_text.py')],input=json.dumps([record['text'],transcript]),text=True))
        spoken=transcript_metrics(*cleaned)
        passed=spoken['wer']<=a.max_wer and spoken['deletions']/spoken['reference_words']<=.1 and spoken['insertions']/spoken['reference_words']<=.1
        row=dict(case=record['case'],text=record['text'],transcript=transcript,raw_metrics=raw,spoken_metrics=spoken,quality_gate_passed=passed,timing=record['metrics'])
        report['examples'].append(row);report['all_passed']=all(x['quality_gate_passed'] for x in report['examples'])
        (root/'quality_benchmark.json').write_text(json.dumps(report,indent=2))
        print(record['case']+': '+transcript+' | raw WER %.3f / spoken WER %.3f / pass %s'%(raw['wer'],spoken['wer'],passed),flush=True)
    weighted=sum(x['spoken_metrics']['reference_words'] for x in report['examples'])
    report['aggregate_spoken_wer']=sum(sum(x['spoken_metrics'][k] for k in ['substitutions','deletions','insertions']) for x in report['examples'])/weighted
    (root/'quality_benchmark.json').write_text(json.dumps(report,indent=2))
    print('ALL QUALITY GATES PASSED:',report['all_passed'],flush=True)
if __name__=='__main__':main()
