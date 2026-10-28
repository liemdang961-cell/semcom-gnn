# Semantic Communication an toàn sử dụng Graph Neural Network

Khóa luận tốt nghiệp - Khoa Công nghệ Thông tin, Trường Đại học Công Thương TP. HCM (ngành An toàn thông tin).

Đề tài: *Nghiên cứu và triển khai Semantic Communication sử dụng Graph Neural Network (GNN)*.

## Cấu trúc thư mục

### Nhóm file gốc (kênh AWGN, Tuần 4-10)

| File | Vai trò |
|---|---|
| `graph_builder.py` | Chuyển bộ ba (chủ thể, quan hệ, khách thể) của WebNLG thành đồ thị PyTorch Geometric |
| `semantic_encoder.py` | Bộ mã hóa ngữ nghĩa dùng GAT có kết hợp đặc trưng cạnh, xuất vector 128 chiều |
| `channel.py` | Mô phỏng kênh truyền **AWGN** và **Rayleigh Fading**, có hàm `get_channel()` để chọn kênh theo tên |
| `decoder.py` | Bộ giải mã ngữ nghĩa (Node Classifier + Relation Classifier) |
| `gradio_app.py` | Giao diện demo: Encoder → Channel → Decoder, có ô chọn mô hình (checkpoint) |
| `train.py` | Huấn luyện Encoder + Decoder qua kênh AWGN trên toàn bộ bộ ba WebNLG (chia 80/10/10), lưu checkpoint vào `checkpoints/` |
| `evaluate.py` | Đánh giá mô hình qua kênh AWGN theo nhiều mức SNR (Accuracy, F1, PSNR, SSIM) |
| `attack.py` | Tấn công đối kháng FGSM qua kênh AWGN, đo ASR/BLEU/BERTScore/F1 theo ε và SNR (ma trận SNR × ε) |
| `train_adversarial.py` | Phòng vệ Adversarial Training qua kênh AWGN (Tuần 9), ra `checkpoint_week9.pt` |
| `compare_defense.py` | So sánh trước/sau phòng vệ trên kênh AWGN (Bảng 3.7, 3.8) |
| `compare_secom.py` | Bản so sánh cũ "chưa SeCom (không qua kênh)" và "có SeCom (qua AWGN)". **Đã được thay bằng `compare_secom_channels.py`** (xem bên dưới) vì cô yêu cầu cả hai trường hợp đều phải đi qua kênh truyền |

### Nhóm file mở rộng (hai kênh AWGN và Rayleigh, theo yêu cầu của giảng viên)

Các file này **không ghi đè** file gốc. Cả hai nhóm cùng dùng chung `channel.py`.

| File | Vai trò | Thay cho |
|---|---|---|
| `evaluate_channels.py` | Đánh giá 1 checkpoint qua **cả hai kênh** theo các mức SNR | `evaluate.py` |
| `train_channel.py` | Huấn luyện qua kênh tùy chọn: `rayleigh` (GNN + SeCom) hoặc `none` (GNN thường, huấn luyện sạch) | `train.py` |
| `train_adversarial_channel.py` | Huấn luyện đối kháng qua kênh tùy chọn | `train_adversarial.py` |
| `attack_channel.py` | Tấn công FGSM qua kênh tùy chọn | `attack.py` |
| `compare_defense_channel.py` | So sánh trước/sau phòng vệ trên kênh tùy chọn | `compare_defense.py` |
| `compare_secom_channels.py` | So sánh **GNN thường** với **GNN + SeCom** qua cả hai kênh | `compare_secom.py` |

Ngoài ra: `checkpoints/` chứa các mô hình đã huấn luyện (`.pt`), `requirements.txt` là danh sách thư viện.

## Tiến độ

- [x] Tuần 4: Semantic Encoder (GAT) và giao diện demo
- [x] Tuần 5: Kênh truyền AWGN + kiến trúc Semantic Decoder
- [x] Tuần 6: Huấn luyện trên toàn bộ dữ liệu (80/10/10, Early Stopping) + đánh giá theo nhiều mức SNR - `checkpoint_week6.pt`
- [x] Tuần 7: Mô hình đe dọa (Threat Model) và tấn công đối kháng FGSM (`attack.py`)
- [x] Tuần 8: Quét tấn công qua 9 mức SNR × 5 mức ε (ma trận ASR, Bảng 3.9)
- [x] Tuần 9: Phòng vệ Adversarial Training (`train_adversarial.py`) - `checkpoint_week9.pt`; so sánh trước/sau (`compare_defense.py`)
- [x] Tuần 10: Tổng hợp, so sánh các kịch bản, tối ưu và phân tích kết quả
- [ ] Mở rộng theo góp ý của cô: thêm kênh Rayleigh Fading, so sánh GNN thường với GNN + SeCom qua hai kênh, rồi tấn công và phòng vệ trên kênh có nhiễu mạnh nhất (xem mục "Phần mở rộng" bên dưới)
- [ ] Tuần 11-12: Hoàn thiện báo cáo, slide, nộp khóa luận

> Lưu ý: `checkpoints/checkpoint_week5.pt` chỉ học từ 300 câu, 5 epoch - dùng để kiểm tra pipeline, KHÔNG phải số liệu chính thức. Số liệu chính thức dùng `checkpoint_week6.pt` (huấn luyện toàn bộ dữ liệu qua AWGN, dừng tại epoch 58, val_loss tốt nhất = 0.4373 ở epoch 43) và `checkpoint_week9.pt` (đã Adversarial Training qua AWGN).

## Checkpoint đã huấn luyện

Các file mô hình (`.pt`) được lưu **trực tiếp trong repo**, ở thư mục `checkpoints/` (mỗi file khoảng 4 MB) - không cần Google Drive. Chỉ cần `git clone` là có sẵn.

| Checkpoint | Mô hình | Tạo bởi |
|---|---|---|
| `checkpoint_week6.pt` | GNN + SeCom, huấn luyện qua AWGN | `train.py` |
| `checkpoint_week9.pt` | GNN + SeCom + Adversarial Training qua AWGN | `train_adversarial.py` |
| `checkpoint_gnn_clean.pt` | **GNN thường**, huấn luyện sạch (không nhiễu) | `train_channel.py --channel none` |
| `checkpoint_rayleigh.pt` | GNN + SeCom, huấn luyện qua Rayleigh | `train_channel.py --channel rayleigh` |
| `checkpoint_rayleigh_adv.pt` | GNN + SeCom + Adversarial Training qua Rayleigh | `train_adversarial_channel.py --channel rayleigh` |

Giao diện (`gradio_app.py`) tự quét thư mục này và liệt kê mọi checkpoint vào ô **"Chọn mô hình"**.

**Khi có checkpoint mới:** đặt tên khác, **không ghi đè file cũ**. Chạy `!ls -lh checkpoints/` để kiểm tra dung lượng rồi upload vào `checkpoints/` trên GitHub (dưới 25 MB thì kéo-thả qua web được; 25-100 MB phải đẩy bằng Git; trên 100 MB phải dùng Git LFS).

## Cài đặt và chạy trên Google Colab

1. Mở [Google Colab](https://colab.research.google.com), tạo notebook mới, chọn **Runtime → Change runtime type → T4 GPU**.
2. Chạy các ô sau:

```python
!git clone https://github.com/liemdang961-cell/semcom-gnn.git
%cd semcom-gnn
!pip install -r requirements.txt -q
```

3. Chạy tiếp các lệnh ở mục **"Hướng dẫn sử dụng từng file"** hoặc **"Phần mở rộng"** bên dưới.

Lần đầu chạy sẽ tự tải mô hình `all-MiniLM-L6-v2` (khoảng 80 MB) và dữ liệu WebNLG.

**Cập nhật code mới:** nếu đã clone từ trước trong cùng phiên, chạy `%cd /content/semcom-gnn` rồi `!git pull`.

**Cài trên máy cá nhân:** Python 3.10+, cài `torch` theo hướng dẫn tại pytorch.org (chọn đúng CUDA), rồi `pip install -r requirements.txt`.

> Mọi lệnh bên dưới chạy tại thư mục gốc của repo. Trên Colab thêm dấu `!` ở đầu mỗi lệnh `python ...`; trên terminal thì bỏ dấu `!`.

---

## Phần mở rộng: hai kênh AWGN và Rayleigh Fading (theo góp ý của cô)

### Mục đích

Cô yêu cầu: (1) thêm một kênh nhiễu mạnh hơn AWGN, (2) so sánh **GNN thường** và **GNN + SeCom** khi **cả hai đều đi qua kênh truyền**, (3) lấy kênh có nhiễu mạnh nhất để **tấn công rồi phòng vệ** lại.

- **GNN thường:** huấn luyện trên dữ liệu **sạch** (không nhiễu), rồi đem đi qua kênh khi đánh giá.
- **GNN + SeCom:** huấn luyện **ngay trong lúc có kênh nhiễu** (mã hóa và giải mã học chung qua kênh), nên vector ngữ nghĩa học cách chịu nhiễu.
- **Kênh AWGN:** nhiễu Gauss trắng cộng thêm, `Y = X + N`.
- **Kênh Rayleigh Fading:** `Y = H·X + N` với `H = √(h_r² + h_i²)/√2`, `h_r, h_i ~ N(0,1)`, nên `E[H²] = 1`. Mỗi phần tử của vector có một hệ số `H` riêng; bên nhận **không biết H** và không cân bằng kênh. Cùng mức SNR, Rayleigh khó hơn AWGN vì thỉnh thoảng `H` rất nhỏ (deep fade).
- **SNR (dB):** tỷ số công suất tín hiệu trên công suất nhiễu của kênh. SNR càng cao thì kênh càng sạch.

Tất cả so sánh dùng **cùng tập test** (seed 42, chia 80/10/10) và **cùng chuỗi nhiễu** (seed theo từng SNR và kênh), nên các cột so sánh được trực tiếp với nhau.

### Thứ tự chạy

```bash
# Bước 1 (tùy chọn, nhanh, không huấn luyện): đo checkpoint cũ qua cả hai kênh
python evaluate_channels.py

# Bước 2: huấn luyện 2 mô hình mới (độc lập nhau, chạy song song được)
python train_channel.py --channel none        # GNN thường (sạch)      -> checkpoint_gnn_clean.pt
python train_channel.py --channel rayleigh    # GNN + SeCom (Rayleigh) -> checkpoint_rayleigh.pt

# Bước 3: so sánh GNN thường với GNN + SeCom qua hai kênh
python compare_secom_channels.py

# Bước 4: xem kênh nào nhiễu mạnh nhất (accuracy thấp nhất ở Bước 3), lấy kênh đó để tấn công.
#         Ví dụ nếu là Rayleigh:
python attack_channel.py --channel rayleigh

# Bước 5: huấn luyện đối kháng trên kênh đó (huấn luyện từ đầu, không cần Bước 2) rồi so sánh
python train_adversarial_channel.py --channel rayleigh
python compare_defense_channel.py --channel rayleigh --snrs 10 -6
```

Chạy thử nhanh để kiểm tra code (KHÔNG dùng số liệu này cho báo cáo):

```bash
python train_channel.py --channel rayleigh --max-samples 300 --max-epochs 2 --out checkpoints/test.pt
```

### Chi tiết từng lệnh

#### `evaluate_channels.py` - Đánh giá 1 checkpoint qua hai kênh

Cho mô hình đi qua kênh AWGN và Rayleigh ở 9 mức SNR (−6 → 18 dB), đo MSE, PSNR, SSIM, Accuracy, F1.

```bash
python evaluate_channels.py                                          # mặc định checkpoint_week6.pt
python evaluate_channels.py --checkpoint checkpoints/checkpoint_rayleigh.pt
```

- Cần: checkpoint để đánh giá. Không huấn luyện gì.
- Kết quả: bảng Accuracy/F1 hai kênh theo SNR, `results_channels_<tên>.json`, `channel_accuracy_comparison_<tên>.png`.
- Dùng để: cho thấy kênh nào khó hơn; điền bảng so sánh hai kênh.

#### `train_channel.py` - Huấn luyện qua kênh tùy chọn

```bash
python train_channel.py --channel none        # GNN thường
python train_channel.py --channel rayleigh    # GNN + SeCom qua Rayleigh
```

- Tùy chọn: `--snr` (mặc định 10 dB như `train.py`), `--out` (đường dẫn checkpoint), `--max-samples`, `--max-epochs`.
- `--channel none`: không qua kênh khi huấn luyện, lưu `checkpoints/checkpoint_gnn_clean.pt`. Nhanh nhất trong các lần huấn luyện.
- `--channel rayleigh`: huấn luyện qua Rayleigh, lưu `checkpoints/checkpoint_rayleigh.pt`.
- Cùng seed 42 và cùng cách chia 80/10/10 với `train.py`, nên tập test giống hệt `checkpoint_week6.pt`.
- Tự dừng sớm (Early Stopping) khi validation loss không giảm sau 15 epoch.

#### `compare_secom_channels.py` - GNN thường so với GNN + SeCom (ý chính của cô)

Đo Accuracy và F1 của ba mô hình qua **cả hai kênh** ở mọi mức SNR:
GNN thường (`checkpoint_gnn_clean.pt`), GNN + SeCom huấn luyện AWGN (`checkpoint_week6.pt`), GNN + SeCom huấn luyện Rayleigh (`checkpoint_rayleigh.pt`).

```bash
python compare_secom_channels.py
```

- Cần: ba checkpoint trên (checkpoint nào chưa có sẽ bị bỏ qua kèm cảnh báo).
- Kết quả: bảng so sánh in ra màn hình, `results_secom_channels.json`, `secom_channels_accuracy.png`, `secom_channels_f1.png`.
- Kỳ vọng: SNR cao thì các mô hình gần bằng nhau; SNR thấp thì GNN thường tụt mạnh hơn GNN + SeCom. Đây chỉ là kỳ vọng, số liệu thật mới là kết luận.

#### `attack_channel.py` - Tấn công FGSM qua kênh đã chọn

Giống `attack.py` (chèn nhiễu FGSM vào vector `z` đang truyền, kịch bản white-box) nhưng `z` đi qua kênh được chọn. Quét 9 mức SNR × 5 mức ε.

```bash
python attack_channel.py --channel rayleigh
python attack_channel.py --channel rayleigh --checkpoint checkpoints/checkpoint_rayleigh.pt
```

- Cần: `checkpoint_rayleigh.pt` (hoặc `--checkpoint` khác). Mặc định kênh `awgn` dùng `checkpoint_week6.pt`.
- Kết quả: ma trận ASR theo SNR × ε, bảng chi tiết (ASR/Accuracy/F1/BLEU/BERTScore), `results_attack_<kênh>.json`, `fgsm_attack_plot_<kênh>.png`, `fgsm_attack_heatmap_<kênh>.png`.
- Dùng để: thay Bảng 3.6, 3.9 và Hình 3.5 cho kênh mới.

#### `train_adversarial_channel.py` - Huấn luyện đối kháng qua kênh đã chọn

Giống `train_adversarial.py` (mỗi câu train thêm một lượt nhiễu FGSM ε=0.1 ngay trong vòng lặp, **không có dataset đối kháng lưu sẵn**), nhưng cả nhánh sạch lẫn nhánh bị tấn công đều đi qua kênh được chọn. Huấn luyện từ đầu, cùng cấu hình như bản AWGN.

```bash
python train_adversarial_channel.py --channel rayleigh
```

- Tùy chọn: `--snr` (mặc định 10), `--epsilon` (mặc định 0.1), `--out`.
- Kết quả: `checkpoints/checkpoint_rayleigh_adv.pt`.

#### `compare_defense_channel.py` - So sánh trước/sau phòng vệ trên kênh đã chọn

Cho cả hai checkpoint (không phòng vệ và có phòng vệ) chịu cùng tấn công FGSM ở 5 mức ε.

```bash
python compare_defense_channel.py --channel rayleigh --snrs 10 -6
```

- Cần: `checkpoint_rayleigh.pt` và `checkpoint_rayleigh_adv.pt`.
- `--snrs`: các mức SNR dùng khi tấn công (`10` giống Bảng 3.7 cũ, `-6` là mức nhiễu cao nhất).
- Kết quả: bảng ASR(%) trước/sau phòng vệ theo ε, `results_defense_<kênh>.json`, `defense_comparison_<kênh>_snr<SNR>.png`.
- Dùng để: thay Bảng 3.7, 3.8 cho kênh mới.

### Ghi chú về cách làm (để trả lời khi bảo vệ)

- Kẻ tấn công vẫn là white-box: biết Encoder/Decoder nhưng không biết hệ số fading `H` của từng lần truyền.
- File gốc (AWGN) và file mở rộng cho cùng thuật toán tấn công và huấn luyện; chỉ khác kênh.
- Checkpoint cũ không bị ghi đè; số liệu AWGN cũ dùng lại làm cột đối chiếu.

---

## Hướng dẫn sử dụng từng file gốc (kênh AWGN)

### 1. `train.py` - Huấn luyện mô hình gốc

Huấn luyện Encoder + Decoder qua kênh AWGN (SNR = 10 dB) trên toàn bộ WebNLG, chia 80/10/10, Early Stopping 15 epoch.

```bash
python train.py
```

- Kết quả: `checkpoints/checkpoint_week6.pt`.
- Thời gian: khá lâu (vài giờ tùy GPU) vì xử lý từng câu một; có in tiến độ mỗi 500 câu.

### 2. `evaluate.py` - Đánh giá theo mức SNR (Bảng 3.5)

```bash
python evaluate.py
```

- Cần `checkpoint_week6.pt`. In bảng Accuracy/F1/PSNR/SSIM theo 9 mức SNR và lưu `snr_evaluation_plot.png`.

### 3. `attack.py` - Tấn công FGSM (Bảng 3.6, 3.9)

```bash
python attack.py
```

- Cần `checkpoint_week6.pt`. Ra ma trận ASR, bảng chi tiết, `fgsm_attack_plot.png` và `fgsm_attack_heatmap.png`.

### 4. `train_adversarial.py` - Phòng vệ Adversarial Training (Tuần 9)

```bash
python train_adversarial.py
```

- Kết quả: `checkpoints/checkpoint_week9.pt`.

### 5. `compare_defense.py` - So sánh trước/sau phòng vệ (Bảng 3.7, 3.8)

```bash
python compare_defense.py
```

- Cần `checkpoint_week6.pt` và `checkpoint_week9.pt`. Ra bảng ASR(%) theo ε và `defense_comparison_plot.png`.

### 6. `compare_secom.py` - Bản so sánh cũ (đã được thay)

Bản cũ coi "chưa SeCom" là không qua kênh (lý tưởng). Cô nhận xét cách này chưa đúng vì GNN cũng phải đi qua kênh truyền, nên dùng `compare_secom_channels.py` ở mục "Phần mở rộng". File này giữ lại để đối chiếu.

### 7. `gradio_app.py` - Giao diện demo

Nhập bộ ba, chọn mức SNR và checkpoint, xem đồ thị gốc so với đồ thị AI khôi phục sau khi qua kênh nhiễu.

```bash
python gradio_app.py
```

- Mở link dạng `https://xxxxx.gradio.live` hiện ra sau khi chạy. Dùng để demo trực tiếp, không dùng để chạy đánh giá hàng loạt.

### Thứ tự chạy khi tái tạo toàn bộ kết quả AWGN

```bash
python train.py              # 1. Mô hình gốc -> checkpoint_week6.pt (chạy 1 lần, khá lâu)
python evaluate.py           # 2. Đánh giá theo SNR -> Bảng 3.5
python attack.py             # 3. Tấn công FGSM -> Bảng 3.6/3.9
python train_adversarial.py  # 4. Phòng vệ -> checkpoint_week9.pt (chạy 1 lần, khá lâu)
python compare_defense.py    # 5. Trước/sau phòng vệ -> Bảng 3.7/3.8
python gradio_app.py         # 6. (tùy chọn) Giao diện demo
```

Phần mở rộng hai kênh xem mục "Phần mở rộng" ở trên.

## Dữ liệu

Bộ dữ liệu WebNLG được tải tự động qua thư viện `datasets`, không lưu trong repo này.

## Nhóm thực hiện

- Đặng Hoàng Liêm - 2033230158
- Lê Đức Tuấn - 2033230265
- Võ Xuân Kiên - 2033230121

Giảng viên hướng dẫn: Nguyễn Thị Hồng Thảo
