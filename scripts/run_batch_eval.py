#!/usr/bin/env python3
import os, sys, json, argparse, subprocess, glob, csv

def run_cmd(cmd, env=None):
    print("Executing: " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_base', default='data/videos')
    parser.add_argument('--result_base', default='results/batch_results')
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--skip_audio', action='store_true')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_root = os.path.dirname(script_dir)
    os.chdir(eval_root)
    sys.path.insert(0, os.path.join(eval_root, 'javisbench', 'src', 'ImageBind'))

    model_dir = os.path.join(eval_root, args.data_base, args.model)
    prompt_dirs = sorted(glob.glob(os.path.join(model_dir, 'prompt_*')))
    if not prompt_dirs:
        print("No prompt directories found in " + model_dir)
        return

    prompts_file = os.path.join(model_dir, 'prompts.json')
    if not os.path.exists(prompts_file):
        print("prompts.json not found in " + model_dir)
        return
    with open(prompts_file) as f:
        prompts = json.load(f)
    id2text = {p['id']: p['text'] for p in prompts}

    result_dir = os.path.join(eval_root, args.result_base, args.model)
    os.makedirs(result_dir, exist_ok=True)

    patch_script = os.path.join(script_dir, 'patch_metrics.py')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f'/usr/lib/x86_64-linux-gnu:{env.get("CONDA_PREFIX","")}/lib:{env.get("LD_LIBRARY_PATH","")}'
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'

    try:
        run_cmd(['python', patch_script, '--apply'], env=env)

        for pd_dir in prompt_dirs:
            prompt_id = os.path.basename(pd_dir)
            text = id2text.get(prompt_id, "")
            print(f"Processing {args.model}/{prompt_id}")

            if not args.skip_audio:
                for seed in range(args.n_seeds):
                    seed_name = f'seed_{seed:02d}'
                    mp4 = os.path.join(pd_dir, f'{seed_name}.mp4')
                    wav = os.path.join(pd_dir, f'{seed_name}.wav')
                    if os.path.exists(mp4) and not os.path.exists(wav):
                        subprocess.run(['ffmpeg', '-y', '-i', mp4, '-vn',
                                        '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav],
                                       capture_output=True, stderr=subprocess.DEVNULL)

            csv_path = os.path.join(pd_dir, 'batch_metadata.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['path', 'audio_path', 'text', 'category'])
                for seed in range(args.n_seeds):
                    seed_name = f'seed_{seed:02d}'
                    vp = os.path.join(pd_dir, f'{seed_name}.mp4')
                    ap = os.path.join(pd_dir, f'{seed_name}.wav')
                    if os.path.exists(vp):
                        writer.writerow([vp, ap, text, 'unknown'])

            for seed in range(args.n_seeds):
                seed_name = f'seed_{seed:02d}'
                res_file = os.path.join(result_dir, f'{prompt_id}_{seed_name}.json')
                if os.path.exists(res_file):
                    print(f"  {seed_name} already done, skipping")
                    continue
                print(f"  Evaluating {seed_name}")
                run_cmd([
                    'python', '-m', 'javisbench.main',
                    '--input_file', csv_path,
                    '--infer_data_dir', pd_dir,
                    '--output_file', res_file,
                    '--max_audio_len_s', '10',
                    '--metrics', 'all',
                    '--exclude', 'video-quality',
                    '--num_workers', '0'
                ], env=env)

        print("All evaluations done. Computing variance...")
        run_cmd([
            'python', 'scripts/collect_results.py',
            '--model', args.model,
            '--result_dir', result_dir,
            '--n_seeds', str(args.n_seeds)
        ], env=env)

    finally:
        run_cmd(['python', patch_script, '--restore'], env=env)

if __name__ == '__main__':
    main()
