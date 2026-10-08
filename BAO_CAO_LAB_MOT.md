# BÁO CÁO THỰC NGHIỆM LAB: MULTI-OBJECT TRACKING EVALUATION

## I. THÔNG TIN NHÓM THỰC HIỆN (2 THÀNH VIÊN)

| STT | Họ và Tên | Mã Sinh Viên | Vai Trò / Phân Công Nhiệm Vụ |
|:---:|---|:---:|---|
| **1** | **Vũ Gia Khải** *(Trưởng nhóm)* | `2A202602786` | Thiết lập môi trường, cấu hình bộ công cụ TrackEval, sửa lỗi tương thích NumPy 2.x, tích hợp pipeline đánh giá tự động. |
| **2** | *[Điền Tên Thành Viên 2]* | *[Điền MSSV 2]* | Thu thập dữ liệu mẫu, chuẩn hoá nhãn Ground-Truth / Tracker, chạy thực nghiệm so sánh các thuật toán và phân tích số liệu. |

- **Lớp / Khóa:** ...
- **Đề tài / Bài Lab:** Đánh giá và so sánh hiệu năng các thuật toán Multi-Object Tracking (MOT) bằng bộ chỉ số HOTA, MOTA và IDF1.
- **Ngày thực hiện:** Tháng 10/2026.

---

## II. MỤC TIÊU VÀ ĐẶT VẤN ĐỀ

1. **Bối cảnh:**
   Trong các bài toán thị giác máy tính cho xe tự hành và robot (Computer Vision / Autonomous Driving), việc chỉ phát hiện vật thể (Object Detection) theo từng khung hình riêng lẻ là chưa đủ. Hệ thống cần theo dõi quỹ đạo chuyển động liên tục của từng đối tượng theo thời gian (Multi-Object Tracking - MOT).
2. **Vấn đề đặt ra:**
   Làm thế nào để đánh giá một thuật toán tracking là "tốt"? Nếu chỉ dùng độ chính xác vị trí (như mAP) thì chưa phản ánh được hiện tượng nhảy ID (ID switch) hoặc mất dấu khi bị che khuất.
3. **Mục tiêu bài lab:**
   - Xây dựng và làm chủ pipeline đánh giá chuẩn quốc tế sử dụng thư viện **TrackEval**.
   - Làm rõ và phân tích bản chất của 3 họ chỉ số đánh giá cốt lõi: **HOTA (IJCV 2020)**, **CLEAR-MOT (MOTA, MOTP)**, và **Identity (IDF1)**.
   - Thực nghiệm đo đạc và đánh giá trên bộ benchmark MOTChallenge.

---

## III. PHƯƠNG PHÁP ĐÁNH GIÁ VÀ CÁC CHỈ SỐ CỐT LÕI

### 1. Thuật toán ghép cặp (Bipartite Matching & Hungarian Algorithm)
TrackEval sử dụng chỉ số tương đồng (thường là IoU - Intersection over Union) giữa các bounding box của Ground-Truth và Tracker dự đoán ở từng frame, sau đó áp dụng giải thuật Hungarian để tìm phép ghép cặp tối ưu (cực đại hóa tổng IoU).

### 2. Các hệ chỉ số đánh giá
- **HOTA (Higher Order Tracking Accuracy):**
  $$HOTA_\alpha = \sqrt{DetA_\alpha \cdot AssA_\alpha}$$
  - $DetA$ (Detection Accuracy): Đánh giá việc phát hiện đúng box.
  - $AssA$ (Association Accuracy): Đánh giá việc giữ đúng ID xuyên suốt quỹ đạo.
  - HOTA tính trung bình hình học trên nhiều ngưỡng $\alpha \in [0.05, 0.95]$, khắc phục nhược điểm phụ thuộc quá nhiều vào bộ detector của MOTA.
- **CLEAR-MOT (MOTA):**
  $$MOTA = 1 - \frac{\sum (FN + FP + IDSW)}{\sum GT}$$
  - Chỉ số truyền thống rất phổ biến, đo lường tổng hợp các lỗi phát hiện sai (FP), bỏ sót (FN) và đổi nhầm ID (IDSW).
- **Identity (IDF1):**
  $$IDF1 = \frac{2 IDTP}{2 IDTP + IDFP + IDFN}$$
  - Đo lường khả năng gán nhãn ID nhất quán trong toàn bộ video, rất quan trọng đối với bài toán giám sát an ninh và điều khiển xe tự hành.

---

## IV. QUÁ TRÌNH TRIỂN KHAI VÀ XỬ LÝ KỸ THUẬT

1. **Khắc phục lỗi tương thích NumPy 2.x:**
   - Phiên bản NumPy hiện đại (2.x) đã loại bỏ hoàn toàn các alias kiểu dữ liệu cũ (`np.int`, `np.float`, `np.bool`).
   - Nhóm đã rà soát và chuẩn hóa 20 file mã nguồn trong `trackeval/metrics/` và `trackeval/datasets/` về các kiểu dữ liệu chuẩn (`int`, `float`, `bool`), đảm bảo repo vận hành ổn định trên Python 3.13 / Conda DL.
2. **Xây dựng bộ kiểm thử hồi quy tự động:**
   - Tạo bộ unit test toán học độc lập (`test_metrics.py`) và kiểm thử luồng tích hợp (`test_mini_benchmark.py`).
   - Kết quả test: **10/10 passed**.
3. **Phát triển module Adapter cho Custom Tracker:**
   - Xây dựng công cụ [scripts/eval_custom_tracker.py](file:///d:/create/vin/TrackEval/scripts/eval_custom_tracker.py) hỗ trợ nhóm nạp trực tiếp kết quả từ các model YOLOv8 / ByteTrack / DeepSORT ở cả hai định dạng `xywh` và `xyxy`.

---

## V. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH

### 1. Bảng số liệu thực nghiệm mẫu (MOT17-mini)

| Mô hình Tracker | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DemoTracker (Baseline)** | 100.00 | 100.00 | 100.00 | 100.00 | 96.40 | 100.00 | 0 |
| **Tracker Thực Nghiệm của Nhóm** | *[Điền]* | *[Điền]* | *[Điền]* | *[Điền]* | *[Điền]* | *[Điền]* | *[Điền]* |

*(Ghi chú: Lệnh chạy tái hiện: `conda run -n DL python scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL mini --METRICS HOTA CLEAR Identity --TRACKERS_TO_EVAL DemoTracker`)*

### 2. Phân tích kết quả
- **Về khả năng phát hiện (Detection):** Quan sát tỷ lệ FP và FN để đánh giá ngưỡng confidence threshold của model detector.
- **Về khả năng bám vết (Association):** Số lượng IDSW phản ánh chất lượng module Re-ID hoặc thuật toán Kalman Filter khi đối tượng đi vào vùng bị che khuất tạm thời (occlusion).
- **So sánh HOTA vs MOTA:** Trong trường hợp detector bỏ sót một số frame nhưng tracker vẫn duy trì được ID khi xuất hiện lại, điểm số HOTA phản ánh tính nhất quán này tốt hơn so với MOTA.

---

## VI. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

1. **Kết luận:**
   - Nhóm đã làm chủ hoàn toàn pipeline đánh giá chất lượng mô hình theo dõi đa đối tượng bằng TrackEval.
   - Mã nguồn được cấu hình chuẩn, không yêu cầu GPU mạnh để đánh giá, phù hợp triển khai trên máy tính cá nhân hoặc kết hợp suy luận từ Kaggle.
2. **Hướng phát triển đồ án:**
   - Tích hợp model YOLOv8-nano + ByteTrack chạy trên video camera giao thông hoặc dữ liệu camera xe tự hành VinFast.
   - Xuất dữ liệu tracking tự động và đưa vào bộ công cụ TrackEval để tối ưu hóa siêu tham số (Hyperparameter Tuning: matching threshold, track buffer length).
