import unittest
import numpy as np
from speech_quality import (split_phrases, trim_silence, join_phrases,
                            expand_initialisms, phrase_pause, attention_diagnostics)
from speech_quality import retry_phrases


class PhraseTests(unittest.TestCase):
    def test_retry_is_bounded_and_preserves_repeated_words(self):
        parts=retry_phrases('go go go please keep speaking',0)
        self.assertEqual(' '.join(parts),'go go go please keep speaking')
        for text,depth in [('one',0),('two words',2)]:
            with self.assertRaises(RuntimeError):retry_phrases(text,depth)
    def test_sentence_clause_and_continuation_pauses(self):
        self.assertLess(phrase_pause('keep speaking'),phrase_pause('a clause,'))
        self.assertLess(phrase_pause('a clause,'),phrase_pause('sentence.'))
        waves=[np.ones(200),np.ones(200)]
        result=join_phrases(waves,1000,texts=['continue','done.'])
        self.assertEqual(len(result),440)
        self.assertEqual((result[0],result[199],result[240],result[-1]),(0,0,0,0))
    def test_intended_repetitions_are_not_deleted(self):
        self.assertEqual(' '.join(split_phrases('go, go, go.')), 'go, go, go.')
    def test_attention_needs_sustained_eos(self):
        a=np.eye(10)
        self.assertFalse(attention_diagnostics(a)['reached_eos'])
        a=np.concatenate([a,np.tile(a[:,-1:],(1,8))],axis=1)
        self.assertTrue(attention_diagnostics(a)['reached_eos'])
    def test_technical_initialisms_are_spoken_as_letters(self):
        self.assertEqual(expand_initialisms('GPU and RTF'), 'gee pee you and ar tee eff')
        self.assertEqual(expand_initialisms('HELLO'), 'HELLO')
    def test_long_text_is_preserved_and_bounded(self):
        text = 'hello world. ' + 'short phrases improve long speech synthesis. '*8
        phrases = split_phrases(text)
        self.assertGreater(len(phrases), 2)
        self.assertEqual(' '.join(phrases), ' '.join(text.split()))
        self.assertTrue(all(len(p) <= 80 for p in phrases))

    def test_negative_signal_is_not_silence(self):
        signal = np.concatenate([np.zeros(100), -np.ones(20), np.zeros(100)])
        result = trim_silence(signal, 1000, pad_seconds=.01)
        self.assertEqual(len(result), 40)
        self.assertEqual(result.min(), -1)

    def test_join_preserves_all_segments_and_headroom(self):
        result = join_phrases([np.ones(100), -2*np.ones(100)], 1000)
        self.assertEqual(len(result), 380)
        self.assertAlmostEqual(np.max(np.abs(result)), .92)
        self.assertTrue(np.all(result[100:280] == 0))

    def test_invalid_audio_fails(self):
        for wav in [np.zeros(10), np.array([float('nan')])]:
            with self.assertRaises(RuntimeError):
                trim_silence(wav, 20000)


if __name__ == '__main__':
    unittest.main()
