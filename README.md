# Semantic Communication an toàn sử dụng Graph Neural Network

Khóa luận tốt nghiệp - Khoa Công nghệ Thông tin, Trường Đại học Công Thương TP. HCM (ngành An toàn thông tin).

Đề tài: *Nghiên cứu và triển khai Semantic Communication sử dụng Graph Neural Network (GNN)*.

## Cấu trúc thư mục

| File | Vai trò |
|---|---|
| `graph_builder.py` | Chuyển bộ ba (chủ thể, quan hệ, khách thể) của WebNLG thành đồ thị PyTorch Geometric |
| `semantic_encoder.py` | Semantic Encoder dùng GAT có kết hợp đặc trưng cạnh, xuất vector 128 chiều |
| `channel.py` | Mô phỏng kênh truyền AWGN |
| `decoder.py` | Semantic Decoder (Node Classifier + Relation Classifier) |
| `gradio_app.py` | Giao diện demo: Encoder → Channel → Decoder, có ô chọn mô hình (checkpoint) để so sánh trước/sau khi học |
| `train.py` | Vòng lặp huấn luyện Encoder + Decoder trên **toàn bộ** bộ ba WebNLG (chia 80/10/10), lưu checkpoint vào `checkpoints/` |
| `evaluate.py` | Đánh giá mô hình theo nhiều mức SNR (Accuracy, F1, PSNR, SSIM) trên đúng tập test 10% giữ riêng lúc train |
| `attack.py` | Mô phỏng tấn công đối kháng FGSM (Threat Model), đo ASR/BLEU/BERTScore/F1 theo từng mức ngân sách nhiễu ε |
| `checkpoints/` | Chứa các file mô hình đã huấn luyện (`.pt`), nằm sẵn trong repo |
| `requirements.txt` | Danh sách thư viện cần cài |

## Tiến độ

- [x] Tuần 4: Semantic Encoder (GAT) và giao diện demo
- [x] Tuần 5: Kênh truyền AWGN + kiến trúc Semantic Decoder (chưa huấn luyện)
- [x] Tuần 5 (tiếp): Vòng lặp huấn luyện đầu tiên (train.py) - đã chạy thử 300 câu, 5 epoch
- [x] Tuần 6: Huấn luyện trên toàn bộ dữ liệu (80/10/10, Early Stopping) + đánh giá theo nhiều mức SNR (Accuracy/F1/PSNR/SSIM) - `checkpoint_week6.pt`
- [x] Tuần 7: Mô hình đe dọa (Threat Model) và tấn công đối kháng FGSM (`attack.py`) - đo ASR/BLEU/BERTScore/F1 theo ε
- [ ] Tuần 8-9: Cơ chế phòng vệ (Adversarial Training)
- [ ] Tuần 10-12: Tối ưu hóa, ablation study, viết báo cáo tổng kết

> Lưu ý: `checkpoints/checkpoint_week5.pt` chỉ học từ 300 câu, 5 epoch - dùng để kiểm tra pipeline, KHÔNG phải số liệu chính thức. Số liệu chính thức của khóa luận (Bảng 3.4, 3.5, 3.6) dùng `checkpoints/checkpoint_week6.pt` (huấn luyện trên toàn bộ dữ liệu, dừng tại epoch 58 nhờ Early Stopping, val_loss tốt nhất = 0.4373 ở epoch 43).

## Checkpoint đã huấn luyện

Các file mô hình đã huấn luyện (`.pt`) được lưu **trực tiếp trong repo**, ở thư mục `checkpoints/` - không cần Google Drive hay upload thủ công. Chỉ cần `git clone` là có sẵn để dùng ngay, kể cả khi giảng viên tự chạy thử.

Giao diện (`gradio_app.py`) tự động quét thư mục này và liệt kê mọi checkpoint tìm được vào ô **"Chọn mô hình"**, kèm tuỳ chọn "Không dùng AI (ngẫu nhiên)" để so sánh trước/sau khi huấn luyện.

**Khi có checkpoint mới ở các tuần sau** (tuần 8, 9...):
1. Trong `train.py`, đổi `CHECKPOINT_LABEL` (ví dụ "Tuần 8 - Adversarial Training") và `CHECKPOINT_PATH` (ví dụ `checkpoints/checkpoint_week8.pt`) - **đặt tên khác**, không ghi đè file cũ.
2. Chạy `train.py`, kiểm tra dung lượng file trước khi up: `!ls -lh checkpoints/`.
3. Upload file mới vào đúng thư mục `checkpoints/` trên GitHub.
   - Dưới 25 MB: kéo-thả qua giao diện web như bình thường.
   - 25–100 MB: web không nhận nữa, cần đẩy bằng dòng lệnh Git (`git add`, `git commit`, `git push`).
   - Trên 100 MB: GitHub từ chối hẳn, lúc đó cần dùng Git LFS.

## Cài đặt và chạy trên Google Colab

1. Mở [Google Colab](https://colab.research.google.com), tạo notebook mới, chọn **Runtime → Change runtime type → T4 GPU**.
2. Chạy lần lượt các ô sau:

```python
!git clone https://github.com/liemdang961-cell/semcom-gnn.git
%cd semcom-gnn
!pip install -r requirements.txt -q
```

3. Từ đây, tùy việc cần làm mà chạy ô tương ứng ở mục **"Hướng dẫn sử dụng từng file"** bên dưới.

Lần đầu chạy sẽ tự tải mô hình `all-MiniLM-L6-v2` (khoảng 80 MB) và dữ liệu WebNLG.

## Cập nhật code mới

Khi repo có thay đổi, trong Colab mới chỉ cần chạy lại ô `!git clone` (luôn lấy bản mới nhất). Nếu đã clone từ trước trong cùng phiên:

```python
%cd /content/semcom-gnn
!git pull
```

## Cài đặt trên máy cá nhân

Yêu cầu Python 3.10 trở lên. Nên cài `torch` theo hướng dẫn tại pytorch.org (chọn đúng phiên bản CUDA), sau đó:

```bash
pip install -r requirements.txt
```

---

## Hướng dẫn sử dụng từng file

> Tất cả lệnh dưới đây chạy tại thư mục gốc của repo (chỗ chứa `train.py`, `evaluate.py`...). Trên Colab thêm dấu `!` ở đầu mỗi lệnh `python ...`; trên máy cá nhân/terminal thì bỏ dấu `!`.

### 1. `train.py` - Huấn luyện mô hình

Huấn luyện Semantic Encoder + Decoder trên toàn bộ dữ liệu WebNLG, chia 80% train / 10% validation / 10% test (giữ riêng, lưu sẵn vào checkpoint để `evaluate.py`/`attack.py` dùng lại đúng tập này). Tự dừng sớm (Early Stopping) khi validation loss không cải thiện sau 15 epoch liên tiếp.

```bash
python train.py
```

- Kết quả: file `checkpoints/checkpoint_week6.pt` (hoặc tên mới nếu đã đổi `CHECKPOINT_PATH` trong code).
- Thời gian chạy: khá lâu (vài giờ tùy GPU) vì xử lý từng câu một, không gộp batch thật sự - cứ để chạy, có in tiến độ mỗi 500 câu/epoch để biết chương trình không bị treo.

### 2. `evaluate.py` - Đánh giá theo mức SNR (Tuần 6, Bảng 3.5)

Đo Accuracy, F1-score, PSNR, SSIM của mô hình tại 9 mức SNR khác nhau (−6dB đến 18dB, bước 3dB), trên đúng 10% tập test đã giữ riêng lúc `train.py` chạy (không tải lại dữ liệu test khác).

```bash
python evaluate.py
```

- Cần có sẵn `checkpoints/checkpoint_week6.pt` (chạy `train.py` trước nếu chưa có).
- Kết quả: in bảng số liệu ra màn hình (copy thẳng vào Bảng 3.5 khóa luận) + lưu biểu đồ `snr_evaluation_plot.png`.
- Xem ảnh trên Colab: `from IPython.display import Image; Image('snr_evaluation_plot.png')`.

### 3. `attack.py` - Tấn công đối kháng FGSM (Tuần 7, Bảng 3.6)

Mô phỏng kẻ tấn công chèn nhiễu đối kháng FGSM vào vector ngữ nghĩa đang truyền trên kênh (Mục 2.4 khóa luận), đo ASR, BLEU, BERTScore (xấp xỉ), F1-score tại 5 mức ngân sách nhiễu ε (0.01 → 0.2).

```bash
python attack.py
```

- Cần có sẵn `checkpoints/checkpoint_week6.pt`.
- Kết quả: in bảng số liệu (copy vào Bảng 3.6 khóa luận) + lưu biểu đồ `fgsm_attack_plot.png`.

### 4. `gradio_app.py` - Giao diện demo trực quan

Demo 1 câu: nhập bộ ba, chọn mức SNR và chọn checkpoint, xem đồ thị gốc vs đồ thị AI khôi phục được sau khi qua kênh nhiễu.

```bash
python gradio_app.py
```

- Mở link dạng `https://xxxxx.gradio.live` hiện ra sau khi chạy (chờ ~30 giây).
- Dùng để **demo trực tiếp cho giảng viên xem**, không dùng để chạy đánh giá hàng loạt (việc đó là của `evaluate.py`/`attack.py`).

### Thứ tự chạy khuyến nghị khi cần tái tạo lại toàn bộ kết quả

```bash
python train.py        # 1. Huấn luyện, ra checkpoint_week6.pt (chạy 1 lần, khá lâu)
python evaluate.py     # 2. Đánh giá theo SNR -> Bảng 3.5 + snr_evaluation_plot.png
python attack.py       # 3. Tấn công FGSM -> Bảng 3.6 + fgsm_attack_plot.png
python gradio_app.py   # 4. (tùy chọn) Mở giao diện demo trực quan
```

## Dữ liệu

Bộ dữ liệu WebNLG được tải tự động qua thư viện `datasets`, không lưu trong repo này.

## Nhóm thực hiện

- Đặng Hoàng Liêm - 2033230158
- Lê Đức Tuấn - 2033230265
- Võ Xuân Kiên - 2033230121

Giảng viên hướng dẫn: Nguyễn Thị Hồng Thảo
