"""
eval_custom_tracker.py - Công cụ đánh giá nhanh tracker tuỳ biến (Custom Tracker) với TrackEval.

Hỗ trợ:
- Đánh giá trực tiếp từ file Ground Truth (gt.txt) và file Tracker (tracker.txt).
- Tự động chuẩn hoá các định dạng phổ biến (MOTChallenge, XYXY, YOLO/ByteTrack).
- Tính toán và in bảng chỉ số HOTA, MOTA, IDF1, DetA, AssA,...
- Lưu báo cáo chi tiết ra file CSV / Markdown / TXT.

Cách dùng:
    python scripts/eval_custom_tracker.py --gt path/to/gt.txt --tracker path/to/tracker.txt
    python scripts/eval_custom_tracker.py --gt path/to/gt.txt --tracker path/to/tracker.txt --box_format xyxy
"""

import os
import sys
import argparse
import tempfile
import shutil
import numpy as np

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval


def parse_and_convert_format(input_file, output_file, is_gt=False, box_format='xywh', default_conf=1.0):
    """
    Chuẩn hoá file txt về chuẩn MOTChallenge:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class/x>, <visibility/y>, <z>
    """
    converted_lines = []
    max_frame = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Tách dấu phẩy hoặc khoảng trắng
            parts = [p.strip() for p in (line.split(',') if ',' in line else line.split())]
            if len(parts) < 6:
                continue

            frame = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            max_frame = max(max_frame, frame)

            x = float(parts[2])
            y = float(parts[3])
            w_or_x2 = float(parts[4])
            h_or_y2 = float(parts[5])

            if box_format.lower() == 'xyxy':
                left = x
                top = y
                w = max(0.0, w_or_x2 - x)
                h = max(0.0, h_or_y2 - y)
            else:  # xywh
                left = x
                top = y
                w = w_or_x2
                h = h_or_y2

            conf = float(parts[6]) if len(parts) > 6 else default_conf

            if is_gt:
                # MOT17 GT: frame, id, left, top, w, h, conf=1, class=1 (pedestrian), visibility=1.0
                cls_id = int(float(parts[7])) if len(parts) > 7 else 1
                vis = float(parts[8]) if len(parts) > 8 else 1.0
                out_line = f"{frame},{obj_id},{left:.2f},{top:.2f},{w:.2f},{h:.2f},{conf:.2f},{cls_id},{vis:.2f}"
            else:
                # MOT17 Tracker: frame, id, left, top, w, h, conf, -1, -1, -1
                out_line = f"{frame},{obj_id},{left:.2f},{top:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"

            converted_lines.append(out_line)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for l in converted_lines:
            f.write(l + '\n')

    return max_frame


def evaluate_custom(gt_path, tracker_path, tracker_name='CustomTracker', seq_name='seq_01',
                    box_format='xywh', output_dir=None, metrics=('HOTA', 'CLEAR', 'Identity')):
    """Thực hiện đánh giá tracker với dữ liệu đầu vào tuỳ biến."""
    temp_dir = tempfile.mkdtemp(prefix='trackeval_custom_')
    try:
        gt_root = os.path.join(temp_dir, 'gt', 'mot_challenge', 'CUSTOM-eval')
        trackers_root = os.path.join(temp_dir, 'trackers', 'mot_challenge', 'CUSTOM-eval')
        seqmaps_dir = os.path.join(temp_dir, 'gt', 'mot_challenge', 'seqmaps')
        os.makedirs(seqmaps_dir, exist_ok=True)

        # 1. Chuẩn hoá GT
        gt_seq_dir = os.path.join(gt_root, seq_name)
        gt_dest = os.path.join(gt_seq_dir, 'gt', 'gt.txt')
        max_frame_gt = parse_and_convert_format(gt_path, gt_dest, is_gt=True, box_format=box_format)

        # 2. Chuẩn hoá Tracker
        tr_seq_dir = os.path.join(trackers_root, tracker_name, 'data')
        tr_dest = os.path.join(tr_seq_dir, f"{seq_name}.txt")
        max_frame_tr = parse_and_convert_format(tracker_path, tr_dest, is_gt=False, box_format=box_format)

        total_frames = max(max_frame_gt, max_frame_tr, 1)

        # 3. Tạo seqinfo.ini
        seqinfo_path = os.path.join(gt_seq_dir, 'seqinfo.ini')
        with open(seqinfo_path, 'w', encoding='utf-8') as f:
            f.write(f"[Sequence]\nname={seq_name}\nimDir=img1\nframeRate=30\nseqLength={total_frames}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n")

        # 4. Tạo seqmaps file
        seqmap_file = os.path.join(seqmaps_dir, 'CUSTOM-eval.txt')
        with open(seqmap_file, 'w', encoding='utf-8') as f:
            f.write(f"name\n{seq_name}\n")

        # 5. Cấu hình Evaluator
        eval_config = {
            'USE_PARALLEL': False,
            'PRINT_RESULTS': True,
            'PRINT_ONLY_COMBINED': False,
            'OUTPUT_SUMMARY': True,
            'OUTPUT_DETAILED': True,
            'PLOT_CURVES': False,
            'OUTPUT_FOLDER': output_dir if output_dir else os.path.join(trackers_root, tracker_name),
        }

        dataset_config = {
            'BENCHMARK': 'CUSTOM',
            'SPLIT_TO_EVAL': 'eval',
            'TRACKERS_TO_EVAL': [tracker_name],
            'GT_FOLDER': os.path.join(temp_dir, 'gt', 'mot_challenge'),
            'TRACKERS_FOLDER': os.path.join(temp_dir, 'trackers', 'mot_challenge'),
            'OUTPUT_FOLDER': output_dir,
            'DO_PREPROC': False,  # Không tiền xử lý loại bỏ distractor đối với dataset tùy biến
        }

        evaluator = trackeval.Evaluator(eval_config)
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)

        metric_classes = {
            'HOTA': trackeval.metrics.HOTA,
            'CLEAR': trackeval.metrics.CLEAR,
            'Identity': trackeval.metrics.Identity,
            'VACE': trackeval.metrics.VACE,
        }
        selected_metrics = [metric_classes[m]({'THRESHOLD': 0.5}) for m in metrics if m in metric_classes]

        res, msg = evaluator.evaluate([dataset], selected_metrics)

        # Xuất tóm tắt ra console
        seq_res = res['MotChallenge2DBox'][tracker_name][seq_name]['pedestrian']
        summary = {
            'HOTA': float(np.mean(seq_res['HOTA']['HOTA']) * 100) if 'HOTA' in seq_res else None,
            'DetA': float(np.mean(seq_res['HOTA']['DetA']) * 100) if 'HOTA' in seq_res else None,
            'AssA': float(np.mean(seq_res['HOTA']['AssA']) * 100) if 'HOTA' in seq_res else None,
            'MOTA': float(seq_res['CLEAR']['MOTA'] * 100) if 'CLEAR' in seq_res else None,
            'MOTP': float(seq_res['CLEAR']['MOTP'] * 100) if 'CLEAR' in seq_res else None,
            'IDF1': float(seq_res['Identity']['IDF1'] * 100) if 'Identity' in seq_res else None,
            'IDSW': int(seq_res['CLEAR']['IDSW']) if 'CLEAR' in seq_res else None,
            'Dets': int(seq_res['Count']['Dets']),
            'GT_Dets': int(seq_res['Count']['GT_Dets']),
        }

        print("\n" + "=" * 60)
        print(f"=== KET QUA DANH GIA: {tracker_name} tren {seq_name} ===")
        print("=" * 60)
        for k, v in summary.items():
            if v is not None:
                if isinstance(v, float):
                    print(f"  - {k:<10}: {v:.2f}%")
                else:
                    print(f"  - {k:<10}: {v}")
        print("=" * 60)

        if output_dir:
            summary_txt = os.path.join(output_dir, f"{tracker_name}_summary.txt")
            with open(summary_txt, 'w', encoding='utf-8') as f:
                for k, v in summary.items():
                    f.write(f"{k}: {v}\n")
            print(f"[SUCCESS] Da luu ket qua vao: {summary_txt}")

        return summary

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Đánh giá nhanh Tracker với Ground Truth.")
    parser.add_argument('--gt', required=True, help="Đường dẫn file ground-truth (gt.txt)")
    parser.add_argument('--tracker', required=True, help="Đường dẫn file kết quả tracker (txt)")
    parser.add_argument('--name', default='CustomTracker', help="Tên tracker để hiển thị trong báo cáo")
    parser.add_argument('--seq', default='Sequence_01', help="Tên chuỗi/video đánh giá")
    parser.add_argument('--box_format', default='xywh', choices=['xywh', 'xyxy'],
                        help="Định dạng bounding box trong file: xywh hoặc xyxy")
    parser.add_argument('--output_dir', default=None, help="Thư mục lưu báo cáo (mặc định xuất ra màn hình)")
    parser.add_argument('--metrics', nargs='+', default=['HOTA', 'CLEAR', 'Identity'],
                        help="Danh sách metric cần đo (HOTA, CLEAR, Identity, VACE)")

    args = parser.parse_args()
    evaluate_custom(
        gt_path=args.gt,
        tracker_path=args.tracker,
        tracker_name=args.name,
        seq_name=args.seq,
        box_format=args.box_format,
        output_dir=args.output_dir,
        metrics=args.metrics
    )
