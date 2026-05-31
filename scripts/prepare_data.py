#!/usr/bin/env python3
import os
import sys
import csv
import subprocess
import argparse
import re

def extract_audio(video_dir):
    for f in sorted(os.listdir(video_dir)):
        if f.endswith('.mp4'):
            base = os.path.splitext(f)[0]
            mp4 = os.path.join(video_dir, f)
            wav = os.path.join(video_dir, base + '.wav')
            if not os.path.exists(wav):
                subprocess.run(['ffmpeg', '-y', '-i', mp4, '-vn',
                                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav],
                               capture_output=True, stderr=subprocess.DEVNULL)

def generate_csv(video_dir, output_csv):
    prompts_dict = {}
    for fname in sorted(os.listdir(video_dir)):
        if fname.startswith('prompts_shard_') and fname.endswith('.txt'):
            with open(os.path.join(video_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        try:
                            idx = int(parts[0].strip())
                            text = parts[1].strip()
                            prompts_dict[idx] = text
                        except ValueError:
                            match = re.match(r'(\d+)\s+(.*)', line)
                            if match:
                                prompts_dict[int(match.group(1))] = match.group(2)
    rows = []
    for idx in range(100):
        vname = f'sample_{idx:04d}.mp4'
        aname = f'sample_{idx:04d}.wav'
        vpath = os.path.join(video_dir, vname)
        if not os.path.exists(vpath):
            break
        apath = os.path.join(video_dir, aname)
        text = prompts_dict.get(idx, '')
        rows.append([vpath, apath, text, 'unknown'])

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'audio_path', 'text', 'category'])
        writer.writerows(rows)
    print(f'Generated {output_csv} with {len(rows)} rows')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--skip_audio', action='store_true')
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(args.video_dir, 'eval_metadata.csv')
    if not args.skip_audio:
        extract_audio(args.video_dir)
    generate_csv(args.video_dir, args.output)
    print('Preparation done.')

if __name__ == '__main__':
    main()
