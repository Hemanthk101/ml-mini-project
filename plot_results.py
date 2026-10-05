import csv
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='results')
args = parser.parse_args()
result_dir = Path(args.results_dir)
rows = list(csv.DictReader((result_dir/'experiment_results.csv').open()))
if not rows:
    raise SystemExit('No observed training values; no graph generated.')
plt.plot([int(x['step']) for x in rows], [float(x['loss']) for x in rows], marker='o')
plt.xlabel('Training step')
plt.ylabel('Tacotron training loss (L1 mel + weighted L1 linear)')
plt.title('Initial short run on LJSpeech; not a quality evaluation')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(str(result_dir/'training_loss.png'), dpi=160)
