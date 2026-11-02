"""Small TensorFlow-free text normalizer shared by independent speech evaluation."""
import json, sys
from text.cleaners import english_cleaners
from speech_quality import expand_initialisms
if __name__=='__main__':
    data=json.load(sys.stdin)
    print(json.dumps([english_cleaners(expand_initialisms(x)) for x in data]))
