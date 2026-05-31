#!/usr/bin/env python3
import json, os, argparse
import numpy as np
from collections import defaultdict

METRICS = [
    'fvd','kvd','fad','audio_quality',
    'clip_score','clap_score','cavp_score',
    'avh_score','javis_score',
    'ib_tv','ib_ta','ib_av',
    'av_align','desync'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--n_seeds', type=int, default=3)
    args = parser.parse_args()

    prompt_dict = defaultdict(lambda: {m: [] for m in METRICS})
    for fname in os.listdir(args.result_dir):
        if not fname.endswith('.json'): continue
        parts = fname.replace('.json','').split('_')
        if len(parts) >= 3:
            prompt_id = parts[0] + '_' + parts[1]
        else:
            continue
        with open(os.path.join(args.result_dir, fname)) as f:
            data = json.load(f)
        for m in METRICS:
            if m in data:
                prompt_dict[prompt_id][m].append(data[m])

    summary = {}
    for pid, md in prompt_dict.items():
        summary[pid] = {}
        for m, vals in md.items():
            if len(vals) >= 2:
                mean = float(np.mean(vals))
                std  = float(np.std(vals, ddof=1))
            elif len(vals) == 1:
                mean = float(vals[0])
                std  = 0.0
            else:
                continue
            summary[pid][m] = {'mean': mean, 'std': std, 'n': len(vals)}

    overall = {}
    for m in METRICS:
        means = [summary[pid][m]['mean'] for pid in summary if m in summary[pid]]
        if means:
            overall[m] = {
                'mean_of_means': float(np.mean(means)),
                'mean_of_stds':  float(np.mean([summary[pid][m]['std'] for pid in summary if m in summary[pid]]))
            }

    final = {
        'model': args.model,
        'n_prompts': len(summary),
        'n_seeds_per_prompt': args.n_seeds,
        'per_prompt': summary,
        'overall': overall
    }

    out_path = os.path.join(os.path.dirname(args.result_dir), f'{args.model}_summary.json')
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f'Summary saved to {out_path}')

if __name__ == '__main__':
    main()
