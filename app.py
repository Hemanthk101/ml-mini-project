"""Local GPU Tacotron app. Start with run-gpu.ps1 app.py. No extra packages."""
import argparse
import csv
import json
import mimetypes
import os
from pathlib import Path
import queue
import threading
import time
import traceback
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from app_metrics import aggregate, timing_metrics, transcript_metrics

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'app_runs'
STATE = dict(status='loading', message='Loading TensorFlow and the original checkpoint…',
             jobs={}, records=[], failed=0, submitted=0, rejected=0, gpu_verified=False)
LOCK = threading.RLock()
QUEUE = queue.Queue(maxsize=8)


def worker(checkpoint):
    started = time.perf_counter()
    try:
        import numpy as np
        import tensorflow as tf
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from hparams import hparams
        from reproducibility import set_seed
        from synthesizer import Synthesizer
        from gpu_runtime import require_gpu, traced_run
        from text import text_to_sequence
        from util import audio
        from text.cleaners import english_cleaners
        from speech_quality import (split_phrases, trim_silence, join_phrases,
                                    expand_initialisms, attention_diagnostics, retry_phrases)
        from scipy.io import wavfile
        if os.environ.get('TACOTRON_REQUIRE_GPU') != '1':
            raise RuntimeError('GPU is required. Start this app with run-gpu.ps1 app.py.')
        set_seed(2026)
        engine = Synthesizer()
        engine.load(checkpoint, build_waveform=False, attention_stop=True)
        devices = [d.name for d in engine.session.list_devices() if d.device_type == 'GPU']
        if not devices:
            raise RuntimeError('No GPU attached to the synthesis session.')
        with LOCK:
            STATE.update(status='ready', message='Ready to synthesize on GPU',
                         gpu=devices, model_load_seconds=time.perf_counter()-started,
                         checkpoint=checkpoint, sample_rate=hparams.sample_rate)
    except Exception as exc:
        traceback.print_exc()
        with LOCK:
            STATE.update(status='error', message=str(exc))
        return
    while True:
        job_id = QUEUE.get()
        with LOCK:
            job = STATE['jobs'][job_id]
            job.update(status='running', message='Generating speech and acoustic features…')
        start = time.perf_counter()
        wait = start-job['_submitted_at']
        try:
            evidence = None
            phrases = split_phrases(english_cleaners(expand_initialisms(job['text'])))
            waves, segment_info = [], []
            inference = 0.0
            mel_parts = []
            pending = [(phrase, 0) for phrase in phrases]
            attempts = 0
            accepted_texts = []
            while pending:
                phrase, depth = pending.pop(0)
                attempts += 1
                seq = text_to_sequence(phrase, hparams.cleaners.split(','))
                feed = {engine.model.inputs: [np.asarray(seq, dtype=np.int32)],
                        engine.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)}
                fetches = [engine.model.linear_outputs, engine.model.mel_outputs, engine.model.alignments]
                inference_start = time.perf_counter()
                if not STATE['gpu_verified']:
                    (linear, mel, alignment), evidence = traced_run(engine.session, fetches, feed,
                        str(RUNS / (job_id+'.device.json')))
                    with LOCK:
                        STATE['gpu_verified'] = True
                else:
                    linear, mel, alignment = engine.session.run(fetches, feed_dict=feed)
                inference += time.perf_counter()-inference_start
                diagnostics = attention_diagnostics(alignment[0])
                if not diagnostics['reached_eos']:
                    # Do not publish a duration-limited, unfinished phrase as a
                    # successful result. Retry shorter preserved text, bounded.
                    pending[0:0] = [(part, depth+1) for part in retry_phrases(phrase, depth)]
                    continue
                # Reuse upstream librosa Griffin-Lim, whose STFT/ISTFT matches training.
                # inv_spectrogram already applies inverse preemphasis exactly once.
                wav = audio.inv_spectrogram(linear[0].T)
                wav = wav[:audio.find_endpoint(np.abs(wav))]
                wav = trim_silence(wav, hparams.sample_rate)
                waves.append(wav)
                accepted_texts.append(phrase)
                frames = min(mel.shape[1], int(len(wav)/(hparams.sample_rate*.0125))+1)
                mel_parts.append(mel[0, :frames])
                segment_info.append(dict(text=phrase, audio_seconds=len(wav)/hparams.sample_rate,
                                         attention=diagnostics))
            wav = join_phrases(waves, hparams.sample_rate, texts=accepted_texts)
            output = RUNS / (job_id+'.wav')
            wavfile.write(str(output), hparams.sample_rate, (np.clip(wav,-1,1)*32767).astype(np.int16))
            with wave.open(str(output), 'rb') as stream:
                duration = stream.getnframes()/stream.getframerate()
            synthesis = time.perf_counter()-start
            # Diagnostic rendering is outside RTF's synthesis timer, inside delivery latency.
            fig, axes = plt.subplots(2, 1, figsize=(10, 5))
            axes[0].imshow(np.concatenate(mel_parts).T, origin='lower', aspect='auto', interpolation='nearest')
            axes[0].set(title='Predicted mel spectrogram', ylabel='Mel bin')
            axes[1].imshow(alignment[0], origin='lower', aspect='auto', interpolation='nearest')
            axes[1].set(title='Attention alignment (last phrase)', xlabel='Decoder step', ylabel='Input character')
            fig.tight_layout()
            fig.savefig(str(RUNS / (job_id+'.png')), dpi=110)
            plt.close(fig)
            metrics = timing_metrics(inference, synthesis, time.perf_counter()-job['_submitted_at'], wait, duration)
            result = dict(id=job_id, text=job['text'], status='completed', metrics=metrics,
                          audio='/runs/'+job_id+'.wav', plot='/runs/'+job_id+'.png',
                          report='/runs/'+job_id+'.json', checkpoint=checkpoint, seed=2026,
                          gpu_verified=True, trace_included=evidence is not None,
                          sample_rate=hparams.sample_rate, mel_shape=list(mel.shape),
                          reconstruction='Upstream librosa Griffin-Lim, original power and iterations',
                          segments=segment_info, peak_headroom=.92,
                          attention_stop=True,
                          phrase_attempts=attempts,
                          transcript_evaluation=None)
            (RUNS / (job_id+'.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
            with LOCK:
                job.update(result)
                STATE['records'].append(result)
        except Exception as exc:
            traceback.print_exc()
            with LOCK:
                job.update(status='failed', message=str(exc))
                STATE['failed'] += 1
        finally:
            QUEUE.task_done()


def public_job(job):
    return {k: v for k, v in job.items() if not k.startswith('_')}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith('POST'):
            super().log_message(fmt, *args)

    def respond(self, data, status=200):
        body = json.dumps(data, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/status':
            with LOCK:
                status = {k: v for k, v in STATE.items() if k not in ['jobs', 'records']}
                status['metrics'] = aggregate(STATE['records'], STATE['failed'], STATE['submitted'])
                status['history'] = [public_job(j) for j in list(STATE['jobs'].values())[-20:]][::-1]
            training = ROOT / 'results/gpu-verified/experiment_results.csv'
            status['training'] = list(csv.DictReader(training.open(encoding='utf-8'))) if training.exists() else []
            return self.respond(status)
        if path.startswith('/api/jobs/'):
            with LOCK:
                job = STATE['jobs'].get(path.split('/')[-1])
                return self.respond(public_job(job) if job else dict(error='Job not found'), 200 if job else 404)
        if path == '/':
            target = ROOT / 'app_static/index.html'
        elif path.startswith('/static/'):
            target = ROOT / 'app_static' / path[len('/static/'):]
            if (ROOT / 'app_static').resolve() not in target.resolve().parents:
                return self.respond(dict(error='Invalid path'), 400)
        elif path.startswith('/runs/'):
            name = path[len('/runs/'):]
            if '/' in name or '\\' in name or not name.endswith(('.wav', '.png', '.json')):
                return self.respond(dict(error='Invalid path'), 400)
            target = RUNS / name
        else:
            return self.respond(dict(error='Not found'), 404)
        if not target.is_file():
            return self.respond(dict(error='Not found'), 404)
        body = target.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 16384:
                raise ValueError('Request must contain a JSON object under 16 KB.')
            data = json.loads(self.rfile.read(length).decode('utf-8'))
            if not isinstance(data, dict):
                raise ValueError('JSON object required.')
            if path == '/api/synthesize':
                text = data.get('text', '')
                if not isinstance(text, str) or not text.strip() or len(text) > 300:
                    raise ValueError('Enter 1–300 characters of English text.')
                with LOCK:
                    if STATE['status'] != 'ready':
                        return self.respond(dict(error=STATE['message']), 503)
                    if QUEUE.full():
                        return self.respond(dict(error='Queue full; wait for the current requests.'), 429)
                    job_id = uuid.uuid4().hex
                    job = dict(id=job_id, text=text.strip(), status='queued', message='Waiting for GPU',
                               _submitted_at=time.perf_counter())
                    STATE['jobs'][job_id] = job
                    STATE['submitted'] += 1
                    QUEUE.put_nowait(job_id)
                    return self.respond(public_job(job), 202)
            if path == '/api/evaluate':
                job_id = data.get('id')
                transcript = data.get('transcript')
                if not isinstance(transcript, str) or len(transcript) > 3000:
                    raise ValueError('Supply a transcript under 3,000 characters; empty is allowed for silent audio.')
                with LOCK:
                    job = STATE['jobs'].get(job_id)
                    if not job or job['status'] != 'completed':
                        raise ValueError('Select a completed synthesis first.')
                    metrics = transcript_metrics(job['text'], transcript)
                    evaluation = dict(metrics=metrics, transcript=transcript,
                                      source='User-supplied transcript, not automated recognition')
                    job['transcript_evaluation'] = evaluation
                    for record in STATE['records']:
                        if record['id'] == job_id:
                            record['transcript_evaluation'] = evaluation
                    (RUNS / (job_id+'.json')).write_text(json.dumps(public_job(job), indent=2), encoding='utf-8')
                    return self.respond(evaluation)
            return self.respond(dict(error='Not found'), 404)
        except (ValueError, TypeError) as exc:
            with LOCK:
                STATE['rejected'] += 1
            return self.respond(dict(error=str(exc)), 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--checkpoint', default='checkpoints/tacotron-20180906/model.ckpt')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    RUNS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=worker, args=(args.checkpoint,), daemon=True).start()
    print('Tacotron app: http://127.0.0.1:%d (GPU model loading)' % args.port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
