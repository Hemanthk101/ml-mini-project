import unittest
import json
from pathlib import Path
import tempfile
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import app
from app_metrics import aggregate, timing_metrics, transcript_metrics


class TranscriptTests(unittest.TestCase):
    def test_identical_normalized_words(self):
        m = transcript_metrics('Hello, WORLD!', 'hello world')
        self.assertEqual((m['precision'], m['recall'], m['f1'], m['wer'], m['cer']), (1, 1, 1, 0, 0))

    def test_substitution_and_insertion(self):
        m = transcript_metrics('one two three', 'one five three four')
        self.assertEqual((m['correct'], m['substitutions'], m['deletions'], m['insertions']), (2, 1, 0, 1))
        self.assertAlmostEqual(m['precision'], .5)
        self.assertAlmostEqual(m['recall'], 2/3)
        self.assertAlmostEqual(m['f1'], 4/7)
        self.assertAlmostEqual(m['wer'], 2/3)

    def test_deletion_and_silence(self):
        m = transcript_metrics('one two three', 'one three')
        self.assertAlmostEqual(m['precision'], 1)
        self.assertAlmostEqual(m['recall'], 2/3)
        self.assertAlmostEqual(m['wer'], 1/3)
        m = transcript_metrics('one two', '')
        self.assertEqual((m['precision'], m['recall'], m['f1'], m['wer']), (0, 0, 0, 1))

    def test_word_order_and_repeated_words(self):
        self.assertGreater(transcript_metrics('a b c', 'c b a')['wer'], 0)
        self.assertEqual(transcript_metrics('go go go', 'go go')['deletions'], 1)

    def test_empty_reference_is_undefined(self):
        with self.assertRaises(ValueError):
            transcript_metrics('...', 'hello')


class TimingTests(unittest.TestCase):
    def test_timing_scope_and_units(self):
        m = timing_metrics(2, 3, 5, 1, 6)
        self.assertEqual(m['rtf'], .5)
        self.assertEqual(m['rtfx'], 2)
        self.assertAlmostEqual(m['requests_per_second'], 1/3)
        self.assertEqual(m['latency_seconds'], 5)

    def test_aggregate_weighted_by_duration(self):
        a = timing_metrics(1, 2, 3, 0, 4)
        b = timing_metrics(1, 4, 5, 0, 2)
        m = aggregate([dict(metrics=a), dict(metrics=b)], failed=1, submitted=3)
        self.assertEqual(m['rtf'], 1)
        self.assertAlmostEqual(m['requests_per_second'], 1/3)
        self.assertAlmostEqual(m['error_rate'], 1/3)
        self.assertEqual(m['p50_latency_seconds'], 4)
        self.assertAlmostEqual(m['p95_latency_seconds'], 4.9)

    def test_unmeasured_and_invalid_timings(self):
        m = aggregate([], failed=0, submitted=0)
        self.assertIsNone(m['error_rate'])
        self.assertIsNone(m['rtf'])
        with self.assertRaises(ValueError):
            timing_metrics(1, 1, 1, 0, 0)


class HttpContractTests(unittest.TestCase):
    """Isolated HTTP fixtures; these are never attributed to generated speech."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_runs, self.old_state = app.RUNS, app.STATE
        app.RUNS = Path(self.temp.name)
        app.STATE = dict(status='ready', message='unit test fixture', submitted=0,
                         failed=0, rejected=0, records=[], jobs={
                             'fixture': dict(id='fixture', status='completed', text='one two three')})
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.base = 'http://127.0.0.1:%d' % self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        app.RUNS, app.STATE = self.old_runs, self.old_state
        self.temp.cleanup()

    def post(self, path, data):
        request = Request(self.base+path, data=json.dumps(data).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request) as response:
            return json.load(response)

    def test_evaluate_route_and_saved_report(self):
        result = self.post('/api/evaluate', dict(id='fixture', transcript='one three'))
        self.assertAlmostEqual(result['metrics']['f1'], .8)
        report = json.loads((app.RUNS/'fixture.json').read_text())
        self.assertEqual(report['transcript_evaluation'], result)

    def test_invalid_input_and_unknown_job(self):
        for path, payload in [('/api/synthesize', dict(text='')),
                              ('/api/evaluate', dict(id='missing', transcript='one'))]:
            with self.assertRaises(HTTPError) as caught:
                self.post(path, payload)
            self.assertEqual(caught.exception.code, 400)
        self.assertEqual(app.STATE['rejected'], 2)


if __name__ == '__main__':
    unittest.main()
