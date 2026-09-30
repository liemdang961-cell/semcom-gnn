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
| `gradio_app.py` | Giao diện demo: Encoder → Channel → Decoder, có ô tick "Sử dụng AI đã huấn luyện" để so sánh trước/sau khi học |
| `train.py` | Vòng lặp huấn luyện Encoder + Decoder trên bộ ba WebNLG, lưu checkpoint |
| `requirements.txt` | Danh sách thư viện cần cài |

## Tiến độ

- [x] Tuần 4: Semantic Encoder (GAT) và giao diện demo
- [x] Tuần 5: Kênh truyền AWGN + kiến trúc Semantic Decoder (chưa huấn luyện)
- [x] Tuần 5 (tiếp): Vòng lặp huấn luyện đầu tiên (train.py) - đã chạy thử 300 câu, 5 epoch
- [ ] Tuần 6: Huấn luyện quy mô lớn hơn + đánh giá theo nhiều mức SNR
- [ ] Tuần 7-8: Mô hình đe dọa và tấn công đối kháng
- [ ] Tuần 9: Adversarial Training
- [ ] Tuần 10-12: Tối ưu, viết báo cáo

> Lưu ý: Checkpoint hiện tại (`checkpoint_week5.pt`) mới học từ 300 câu, 5 epoch - chỉ để kiểm tra pipeline học được, chưa phải số liệu khoa học chính thức của khóa luận.

## Sử dụng checkpoint đã huấn luyện (tùy chọn)

Giao diện có ô tick **"Sử dụng AI đã huấn luyện để khôi phục"**. Muốn dùng được ô này:

1. File `checkpoint_week5.pt` (sinh ra sau khi chạy `train.py`) được lưu ở Google Drive, **không** đưa lên GitHub (xem mục "Lưu ý" cuối trang).
2. Trong Colab, mount Drive trước khi chạy giao diện:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Đảm bảo file nằm đúng đường dẫn `/content/drive/MyDrive/KhoaLuan_SemCom/checkpoint_week5.pt` (đường dẫn này đã được ghi cứng trong `gradio_app.py`; nếu nhóm lưu ở chỗ khác, cần sửa lại danh sách `CHECKPOINT_CANDIDATES` trong file đó).

Không tick ô này (mặc định): giao diện vẫn cho xem thử bước khôi phục, nhưng dùng AI **chưa học** (trọng số ngẫu nhiên) - minh họa rõ sự khác biệt trước/sau khi huấn luyện.

## Cài đặt và chạy trên Google Colab

1. Mở [Google Colab](https://colab.research.google.com), tạo notebook mới, chọn **Runtime → Change runtime type → T4 GPU**.
2. Chạy lần lượt các ô sau:

```python
!git clone https://github.com/liemdang961-cell/semcom-gnn.git
%cd semcom-gnn
!pip install -r requirements.txt -q
```

Nếu muốn dùng ô tick "Sử dụng AI đã huấn luyện" (xem mục bên dưới), chạy thêm ở 1 ô riêng:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Cuối cùng, chạy giao diện:

```python
!python gradio_app.py
```

3. Chờ khoảng 30 giây, bấm vào đường link dạng `https://xxxxx.gradio.live` hiện ra để mở giao diện.

Lần đầu chạy sẽ tự tải mô hình `all-MiniLM-L6-v2` (khoảng 80 MB).

## Cập nhật code mới

Khi repo có thay đổi, trong Colab mới chỉ cần chạy lại các ô trên (lệnh `git clone` luôn lấy bản mới nhất). Nếu đã clone từ trước trong cùng phiên:

```python
!git pull
```

## Cài đặt trên máy cá nhân

Yêu cầu Python 3.10 trở lên. Nên cài `torch` theo hướng dẫn tại pytorch.org (chọn đúng phiên bản CUDA), sau đó:

```bash
pip install -r requirements.txt
python gradio_app.py
```

## Dữ liệu

Bộ dữ liệu WebNLG được tải tự động qua thư viện `datasets`, không lưu trong repo này.

## Nhóm thực hiện

- Đặng Hoàng Liêm - 2033230158
- Lê Đức Tuấn - 2033230265
- Võ Xuân Kiên - 2033230121

Giảng viên hướng dẫn: Nguyễn Thị Hồng Thảo
