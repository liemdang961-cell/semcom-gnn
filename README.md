# Semantic Communication an toàn sử dụng Graph Neural Network

Khóa luận tốt nghiệp - Khoa Công nghệ Thông tin, Trường Đại học Công Thương TP. HCM (ngành An toàn thông tin).

Đề tài: *Nghiên cứu và triển khai Semantic Communication sử dụng Graph Neural Network (GNN)*.

## Cấu trúc thư mục

| File | Vai trò |
|---|---|
| `graph_builder.py` | Chuyển bộ ba (chủ thể, quan hệ, khách thể) của WebNLG thành đồ thị PyTorch Geometric |
| `semantic_encoder.py` | Semantic Encoder dùng GAT có kết hợp đặc trưng cạnh, xuất vector 128 chiều |
| `gradio_app.py` | Giao diện demo: nhập bộ ba, xem đồ thị và vector ngữ nghĩa |
| `requirements.txt` | Danh sách thư viện cần cài |

## Tiến độ

- [x] Tuần 4: Semantic Encoder (GAT) và giao diện demo
- [ ] Tuần 5: Kênh truyền AWGN
- [ ] Tuần 6: Đánh giá theo nhiều mức SNR
- [ ] Tuần 7-8: Mô hình đe dọa và tấn công đối kháng
- [ ] Tuần 9: Adversarial Training
- [ ] Tuần 10-12: Tối ưu, viết báo cáo

> Lưu ý: Encoder hiện **chưa được huấn luyện**. Giao diện demo chỉ kiểm tra pipeline chạy đúng, số liệu chưa có ý nghĩa khoa học.

## Cài đặt và chạy trên Google Colab

1. Mở [Google Colab](https://colab.research.google.com), tạo notebook mới, chọn **Runtime → Change runtime type → T4 GPU**.
2. Chạy lần lượt các ô sau:

```python
!git clone https://github.com/liemdang961-cell/semcom-gnn.git
%cd semcom-gnn
!pip install -r requirements.txt -q
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
