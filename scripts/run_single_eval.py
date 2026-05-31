#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess

def run_cmd(cmd, env=None):
    print('Executing: ' + ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--output_dir', default='results/single_results')
    args = parser.parse_args()

    model_name = args.model_name or os.path.basename(os.path.abspath(args.model_dir))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_root = os.path.dirname(script_dir)
    os.chdir(eval_root)
    sys.path.insert(0, os.path.join(eval_root, 'javisbench', 'src', 'ImageBind'))

    csv_file = os.path.join(args.model_dir, 'eval_metadata.csv')
    if not os.path.exists(csv_file):
        print('eval_metadata.csv not found. Run prepare_data.py first.')
        return

    patch_script = os.path.join(script_dir, 'patch_metrics.py')
    output_dir = os.path.join(eval_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    fvd_out = os.path.join(output_dir, f'{model_name}_fvd.json')
    semantic_out = os.path.join(output_dir, f'{model_name}_semantic.json')
    align_out = os.path.join(output_dir, f'{model_name}_align.json')

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f'/usr/lib/x86_64-linux-gnu:{env.get("CONDA_PREFIX","")}/lib:{env.get("LD_LIBRARY_PATH","")}'
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'

    try:
        run_cmd(['python', patch_script, '--apply'], env=env)

        run_cmd([
            'python', '-m', 'javisbench.main',
            '--input_file', csv_file,
            '--infer_data_dir', args.model_dir,
            '--output_file', fvd_out,
            '--max_audio_len_s', '10',
            '--fvd_avcache_path', os.path.join(eval_root, 'javisbench', 'data', 'eval', 'JavisBench', 'cache', 'fvd_fad', 'JavisBench-vanilla-max4s.pt'),
            '--metrics', 'fvd+kvd+fad', 'audio-quality',
            '--num_workers', '0'
        ], env=env)

        run_cmd([
            'python', '-m', 'javisbench.main',
            '--input_file', csv_file,
            '--infer_data_dir', args.model_dir,
            '--output_file', semantic_out,
            '--max_audio_len_s', '10',
            '--metrics', 'cxxp-score', 'av-score',
            '--num_workers', '0'
        ], env=env)

        run_cmd([
            'python', '-m', 'javisbench.main',
            '--input_file', csv_file,
            '--infer_data_dir', args.model_dir,
            '--output_file', align_out,
            '--max_audio_len_s', '10',
            '--metrics', 'imagebind-score', 'av-align', 'desync',
            '--num_workers', '0'
        ], env=env)

        merged = {}
        for f in [fvd_out, semantic_out, align_out]:
            if os.path.exists(f):
                with open(f) as fp:
                    merged.update(json.load(fp))
        final_path = os.path.join(output_dir, f'{model_name}_metrics.json')
        with open(final_path, 'w') as fp:
            json.dump(merged, fp, indent=2)
        print(f'Evaluation done. Results saved to {final_path}')

    finally:
        run_cmd(['python', patch_script, '--restore'], env=env)

if __name__ == '__main__':
    main()
