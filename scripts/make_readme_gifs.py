#!/usr/bin/env python3
"""Transcode the complete demo videos into compact, looping README previews."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-dir', type=Path, default=ROOT / 'release-assets')
    args = parser.parse_args()
    records = {}
    output = ROOT / 'docs/images'
    output.mkdir(parents=True, exist_ok=True)
    for name in ('app_preview', 'mujoco_preview'):
        source = args.media_dir / f'{name}.mp4'
        destination = output / f'{name}.gif'
        filters = ('[0:v]setpts=(PTS-STARTPTS)/5,fps=6,scale=640:-1:flags=lanczos,split[a][b];'
                   '[a]palettegen=max_colors=64:stats_mode=diff[p];'
                   '[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle[v]')
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-threads', '4', '-i', str(source), '-filter_complex_threads', '2',
                        '-filter_complex', filters, '-map', '[v]', '-an', '-loop', '0',
                        str(destination)], check=True)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
            '-select_streams', 'v:0', '-count_frames', '-show_entries',
            'stream=width,height,nb_read_frames:format=duration', '-of', 'json', str(destination)]))
        if destination.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f'{destination.name} exceeds the 10 MiB README budget')
        if not 35 <= float(probe['format']['duration']) <= 37:
            raise ValueError('Unexpected preview duration')
        if int(probe['streams'][0]['nb_read_frames']) < 210:
            raise ValueError('Incomplete animated preview')
        records[name] = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            gif_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                            bytes=destination.stat().st_size, speed=5, fps=6,
                            full_source_timeline=True, probe=probe)
        print(f'{destination.name}: {destination.stat().st_size} bytes', flush=True)
    (ROOT / 'docs/gif_previews.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
