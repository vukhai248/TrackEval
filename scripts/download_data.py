"""
download_data.py - Công cụ tải dữ liệu benchmark chính thức cho TrackEval.

Hỗ trợ:
- Tải 'data.zip' (~143MB) chứa ground-truth và trackers mẫu cho MOTChallenge, KITTI, BDD100k, TAO, DAVIS.
- Tải 'person_path_22_data.zip' (~3MB).
- Tải 'rob_mots_train_data.zip' (~750MB).
- Tự động giải nén vào thư mục `TrackEval/data/`.

Cách dùng:
    python scripts/download_data.py --dataset default
    python scripts/download_data.py --dataset personpath22
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import time


URLS = {
    'default': {
        'url': 'https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip',
        'filename': 'data.zip',
        'desc': 'Du lieu mac dinh (MOTChallenge, KITTI, BDD100k, TAO, DAVIS) ~143MB'
    },
    'personpath22': {
        'url': 'https://tracking-dataset-eccv-2022.s3.us-east-2.amazonaws.com/person_path_22_data.zip',
        'filename': 'person_path_22_data.zip',
        'desc': 'PersonPath22 benchmark ~3MB'
    },
    'robmots': {
        'url': 'https://omnomnom.vision.rwth-aachen.de/data/RobMOTS/train_data.zip',
        'filename': 'rob_mots_train_data.zip',
        'desc': 'RobMOTS train data ~750MB'
    }
}


def download_with_progress(url, dest_path):
    print(f"[INFO] Bat dau tai tu: {url}")
    print(f"[INFO] Luu vao: {dest_path}")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get('content-length', 0))
        block_size = 1024 * 64  # 64KB
        downloaded = 0
        start_time = time.time()
        
        with open(dest_path, 'wb') as f:
            while True:
                buffer = resp.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                f.write(buffer)
                
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    elapsed = time.time() - start_time
                    speed = downloaded / (1024 * 1024 * max(elapsed, 0.001))
                    sys.stdout.write(f"\r[DANG TAI] {downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({percent:.1f}%) - {speed:.2f} MB/s")
                    sys.stdout.flush()
        print("\n[INFO] Tai hoan tat!")


def extract_zip(zip_path, extract_to):
    print(f"[INFO] Dang giai nen {zip_path} vao {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("[INFO] Giai nen thanh cong!")


def main():
    parser = argparse.ArgumentParser(description="Tai du lieu benchmark cho TrackEval.")
    parser.add_argument('--dataset', default='default', choices=list(URLS.keys()),
                        help="Bo du lieu can tai: default (143MB), personpath22 (3MB), robmots (750MB)")
    parser.add_argument('--extract_dir', default=None, help="Thu muc giai nen (mac dinh la root TrackEval)")
    parser.add_argument('--keep_zip', action='store_true', help="Giu lai file zip sau khi giai nen")

    args = parser.parse_args()
    code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    extract_to = args.extract_dir if args.extract_dir else code_path

    info = URLS[args.dataset]
    print(f"[TARGET] {info['desc']}")
    
    dest_zip = os.path.join(code_path, info['filename'])
    download_with_progress(info['url'], dest_zip)
    extract_zip(dest_zip, extract_to)
    
    if not args.keep_zip and os.path.isfile(dest_zip):
        os.remove(dest_zip)
        print(f"[INFO] Da xoa file zip tam: {dest_zip}")


if __name__ == '__main__':
    main()
