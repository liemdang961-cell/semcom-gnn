"""
Tuần 5 & 6 - Kênh truyền vật lý AWGN và Rayleigh Fading (Mục 2.3.2 khóa luận).

Công thức AWGN:             Y = X + N
Công thức Rayleigh Fading:  Y = H * X + N, với H là hệ số suy hao đa đường.

Công suất nhiễu N được tính từ SNR mong muốn theo
SNR(dB) = 10*log10(Ps/Pn)  (đúng công thức đã ghi ở Mục 3.2.2).

Vector đầu vào là đầu ra 128 chiều của Semantic Encoder.

Bản này giữ nguyên hàm awgn_channel() cũ và hàm rayleigh_channel() do Kiên viết,
chỉ bổ sung get_channel(name) để các file train/tấn công/phòng thủ chọn kênh bằng
tên ("awgn" | "rayleigh" | "none") mà không phải sửa code ở nhiều nơi.

LƯU Ý (ghi rõ khi bảo vệ):
  - Rayleigh: H = sqrt(h_r^2 + h_i^2)/sqrt(2) với h_r, h_i ~ N(0,1) độc lập
    => E[H^2] = 1 (công suất trung bình không đổi), nhưng H thỉnh thoảng rất nhỏ
    (deep fade) nên kênh khó hơn AWGN ở cùng mức SNR.
  - Mỗi phần tử của z có 1 hệ số H riêng (fading nhanh theo từng phần tử).
  - Bên nhận KHÔNG biết H và không cân bằng kênh: bộ giải mã nhận thẳng H*z + N.
"""

import torch


def awgn_channel(z: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Thêm nhiễu Gauss trắng vào vector ngữ nghĩa theo mức SNR cho trước.

    Args:
        z: tensor [N, D] - vector ngữ nghĩa đầu ra từ Semantic Encoder.
        snr_db: mức SNR mong muốn, đơn vị dB. SNR cao = ít nhiễu.
    Returns:
        tensor [N, D] cùng shape với z, đã bị nhiễu AWGN.
    """
    signal_power = z.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(z) * torch.sqrt(noise_power)
    return z + noise


def rayleigh_channel(z: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Thêm hiệu ứng suy hao đa đường Rayleigh Fading và nhiễu AWGN.

    Args:
        z: tensor [N, D] - vector ngữ nghĩa đầu ra từ Semantic Encoder.
        snr_db: mức SNR mong muốn, đơn vị dB.
    Returns:
        tensor [N, D] cùng shape với z, đã qua kênh Rayleigh Fading.
    """
    signal_power = z.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    # Hệ số suy hao Rayleigh (phần thực và phần ảo tuân theo phân phối Gauss)
    h_r = torch.randn_like(z)
    h_i = torch.randn_like(z)
    h = torch.sqrt(h_r ** 2 + h_i ** 2) / (2.0 ** 0.5)

    noise = torch.randn_like(z) * torch.sqrt(noise_power)

    # Tín hiệu nhận được: y = h * x + n
    return h * z + noise


def no_channel(z: torch.Tensor, snr_db: float = 0.0) -> torch.Tensor:
    """Không qua kênh (lý tưởng, không nhiễu) - dùng để huấn luyện "GNN thường"
    trên dữ liệu sạch. snr_db chỉ để giữ chung chữ ký hàm với 2 kênh trên."""
    return z


CHANNELS = {
    "awgn": awgn_channel,
    "rayleigh": rayleigh_channel,
    "none": no_channel,
}

CHANNEL_DISPLAY = {
    "awgn": "AWGN",
    "rayleigh": "Rayleigh Fading",
    "none": "Không qua kênh",
}


def get_channel(name: str):
    """Trả về hàm kênh theo tên: 'awgn' | 'rayleigh' | 'none'."""
    key = name.lower()
    if key not in CHANNELS:
        raise ValueError(f"Kênh '{name}' không hợp lệ. Chọn một trong: {list(CHANNELS)}")
    return CHANNELS[key]


if __name__ == "__main__":
    # Test nhanh cả 2 kênh truyền
    torch.manual_seed(0)
    z = torch.randn(3, 128)
    print("--- Test nhanh AWGN Channel ---")
    for snr in [-6, 0, 10, 18]:
        z_noisy = awgn_channel(z, snr)
        mse = (z - z_noisy).pow(2).mean().item()
        print(f"SNR = {snr:>3} dB -> MSE (AWGN) = {mse:.4f}")

    print("\n--- Test nhanh Rayleigh Fading Channel ---")
    for snr in [-6, 0, 10, 18]:
        z_noisy = rayleigh_channel(z, snr)
        mse = (z - z_noisy).pow(2).mean().item()
        print(f"SNR = {snr:>3} dB -> MSE (Rayleigh) = {mse:.4f}")
