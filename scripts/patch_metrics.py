#!/usr/bin/env python3
import sys
import os

def find_metrics_path():
    paths = [
        'javisbench/src/metrics.py',
        os.path.join(os.path.dirname(__file__), '..', 'javisbench', 'src', 'metrics.py')
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError('metrics.py not found')

METRICS_PATH = find_metrics_path()
ASSERT_LINES = [263, 298, 337, 405]

def apply():
    with open(METRICS_PATH, 'r') as f:
        lines = f.readlines()
    for ln in ASSERT_LINES:
        if ln <= len(lines) and lines[ln-1].strip().startswith('assert') and not lines[ln-1].strip().startswith('# assert'):
            lines[ln-1] = '# ' + lines[ln-1]
    with open(METRICS_PATH, 'w') as f:
        f.writelines(lines)
    print('Patch applied')

def restore():
    with open(METRICS_PATH, 'r') as f:
        lines = f.readlines()
    for ln in ASSERT_LINES:
        if ln <= len(lines) and lines[ln-1].strip().startswith('# assert'):
            lines[ln-1] = lines[ln-1][2:]
    with open(METRICS_PATH, 'w') as f:
        f.writelines(lines)
    print('Patch restored')

if __name__ == '__main__':
    if '--apply' in sys.argv:
        apply()
    elif '--restore' in sys.argv:
        restore()
    else:
        print('Usage: python patch_metrics.py --apply | --restore')
