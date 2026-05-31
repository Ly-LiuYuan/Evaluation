#!/usr/bin/env python3
import json, argparse

METRICS = [
    ('fvd','FVD','↓'), ('kvd','KVD','↓'), ('fad','FAD','↓'),
    ('audio_quality','Audio Quality','↑'),
    ('clip_score','CLIPScore','↑'), ('clap_score','CLAPScore','↑'),
    ('cavp_score','CAVP Score','↑'), ('avh_score','AVHScore','↑'),
    ('javis_score','JavisScore','↑'),
    ('ib_tv','IB-TV','↑'), ('ib_ta','IB-TA','↑'), ('ib_av','IB-AV','↑'),
    ('av_align','AV-Align','↑'), ('desync','DeSync','↓')
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--format', choices=['latex','csv'], default='latex')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    overall = data.get('overall', {})

    if args.format == 'latex':
        print(r'\begin{table}[h]')
        print(r'\centering')
        print(r'\caption{Model: ' + data['model'] + '}')
        print(r'\begin{tabular}{lcc}')
        print(r'\hline')
        print(r'Metric & Mean $\pm$ Std & Direction \\ \hline')
        for key, name, dir_ in METRICS:
            if key in overall:
                m = overall[key]['mean_of_means']
                s = overall[key]['mean_of_stds']
                print(f'{name} & ${m:.4f} \\pm {s:.4f}$ & {dir_} \\\\')
        print(r'\hline')
        print(r'\end{tabular}')
        print(r'\end{table}')
    else:
        print('Metric,Mean,Std,Direction')
        for key, name, dir_ in METRICS:
            if key in overall:
                m = overall[key]['mean_of_means']
                s = overall[key]['mean_of_stds']
                print(f'{name},{m:.4f},{s:.4f},{dir_}')

if __name__ == '__main__':
    main()
