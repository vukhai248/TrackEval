# Hướng Dẫn Sử Dụng TrackEval (Chuẩn Bị Cho Bài Lab & Đồ Án)

Tài liệu này tóm tắt toàn bộ quy trình thiết lập, kiểm thử và sử dụng thư viện **TrackEval** để đánh giá các thuật toán theo dõi đa đối tượng (Multi-Object Tracking - MOT).

---

## 1. Yêu Cầu Môi Trường & Cài Đặt

Repo đã được tối ưu và sửa toàn diện để tương thích hoàn toàn với **Python 3.13**, **NumPy 2.x** trên môi trường Conda `DL`.

### Kích hoạt môi trường và kiểm tra:
```powershell
# Chạy trực tiếp qua conda run (khuyến nghị trên Windows):
conda run -n DL python --version

# Hoặc kích hoạt môi trường:
conda activate DL
```

### Cài đặt thư viện:
```powershell
# Cài đặt TrackEval ở chế độ editable
conda run -n DL pip install -e . --no-deps

# Cài đặt các thư viện phụ trợ (nếu chưa có)
conda run -n DL pip install tabulate pytest
```

*(Lưu ý: TrackEval chạy 100% trên CPU bằng NumPy và SciPy, **không cần GPU** và **không cần train model**).*

---

## 2. Kiểm Thử Hệ Thống (Automated Testing)

Chạy bộ kiểm thử tự động để xác nhận toàn bộ các phép đo HOTA, CLEAR (MOTA), Identity (IDF1) và bộ nạp benchmark hoạt động chính xác:

```powershell
conda run -n DL pytest tests/test_metrics.py tests/test_mini_benchmark.py -v
```

Kết quả mong đợi: **10 passed (100%)**.

---

## 3. Chạy Đánh Giá Trên Benchmark Mẫu (MOT17-mini)

Repo đã tích hợp sẵn một bộ dữ liệu mẫu chuẩn format quốc tế tại thư mục `data/` (`MOT17-mini`):

```powershell
conda run -n DL python scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL mini --METRICS HOTA CLEAR Identity --TRACKERS_TO_EVAL DemoTracker
```

Kết quả sẽ xuất trực tiếp ra màn hình và tự động lưu các file:
- `data/trackers/mot_challenge/MOT17-mini/DemoTracker/pedestrian_summary.txt`: Tóm tắt các chỉ số chính.
- `data/trackers/mot_challenge/MOT17-mini/DemoTracker/pedestrian_detailed.csv`: Chi tiết từng frame và chuỗi.
- `data/trackers/mot_challenge/MOT17-mini/DemoTracker/pedestrian_plot.png`: Đồ thị HOTA / MOTA / curves.

---

## 4. Đánh Giá Tracker Tùy Biến Của Nhóm (Custom Tracker Adapter)

Nếu nhóm có kết quả chạy từ mô hình riêng (ví dụ: **YOLOv8 + ByteTrack / DeepSORT**), sử dụng công cụ tiện ích [scripts/eval_custom_tracker.py](scripts/eval_custom_tracker.py):

### Cú pháp:
```powershell
conda run -n DL python scripts/eval_custom_tracker.py `
    --gt "path/to/gt.txt" `
    --tracker "path/to/tracker_output.txt" `
    --name "ByteTrack_YOLOv8" `
    --seq "Video_GiaoThong_01" `
    --output_dir "results_eval"
```

### Các định dạng hỗ trợ:
1. **Chuẩn MOTChallenge (Mặc định `--box_format xywh`):**
   - `<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, [conf]`
2. **Định dạng góc toạ độ (`--box_format xyxy`):**
   - `<frame>, <id>, <x1>, <y1>, <x2>, <y2>, [conf]`

---

## 5. Ý Nghĩa Các Chỉ Số Đánh Giá Chính (Metric Cheat Sheet)

| Chỉ số | Tên đầy đủ | Ý nghĩa thực tế | Điểm số tốt |
|---|---|---|---|
| **HOTA** | Higher Order Tracking Accuracy | Cân bằng giữa chất lượng phát hiện (Detection) và chất lượng bám vết (Association). Là metric chuẩn mực nhất hiện nay. | Càng cao càng tốt (0 - 100%) |
| **DetA** | Detection Accuracy | Độ chính xác vị trí và số lượng bounding box phát hiện được. | Càng cao càng tốt |
| **AssA** | Association Accuracy | Khả năng giữ đúng ID của vật thể qua các khung hình liên tiếp. | Càng cao càng tốt |
| **MOTA** | Multi-Object Tracking Accuracy | Đánh giá tổng hợp lỗi phát hiện sai (FP), bỏ sót (FN) và nhảy ID (IDSW). | Càng cao càng tốt |
| **MOTP** | Multi-Object Tracking Precision | Độ chính xác vị trí bounding box (IoU trung bình giữa box và ground-truth). | Càng cao càng tốt |
| **IDF1** | ID F1-Score | Đo lường mức độ định danh đối tượng chính xác xuyên suốt video. | Càng cao càng tốt |
| **IDSW** | Identity Switches | Số lần vật thể bị đổi nhầm sang ID khác (do che khuất, giao cắt). | Càng thấp càng tốt |

---

## 6. Tải Toàn Bộ Dataset Chính Thức (Tùy Chọn)

Nếu nhóm cần benchmark trên toàn bộ tập dữ liệu chính thức của cuộc thi MOTChallenge (khoảng 143MB):

```powershell
conda run -n DL python scripts/download_data.py --dataset default
```
Script sẽ tự động tải, hiển thị tiến độ và giải nén chuẩn vào thư mục `TrackEval/data/`.
