"""
Tuan 5 - Kenh truyen vat ly AWGN (Muc 2.3.2 khoa luan).

Cong thuc: Y = X + N, trong do cong suat nhieu duoc tinh tu SNR mong muon
theo SNR(dB) = 10*log10(Ps/Pn)  (dung cong thuc da ghi o Muc 3.2.2).

Vector dau vao la dau ra 128 chieu cua Semantic Encoder (tuan 4).
"""

import torch


def awgn_channel(z: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Them nhieu Gauss trang vao vector ngu nghia theo muc SNR cho truoc.

    Args:
        z: tensor [N, D] - vector ngu nghia dau ra tu Semantic Encoder.
        snr_db: muc SNR mong muon, don vi dB. SNR cao = it nhieu.
    Returns:
        tensor [N, D] cung shape voi z, da bi nhieu.
    """
    signal_power = z.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(z) * torch.sqrt(noise_power)
    return z + noise


if __name__ == "__main__":
    # Test nhanh: SNR cao thi nhieu them vao rat nho, SNR thap thi nhieu lon
    z = torch.randn(3, 128)
    for snr in [-6, 0, 10, 18]:
        z_noisy = awgn_channel(z, snr)
        mse = (z - z_noisy).pow(2).mean().item()
        print(f"SNR = {snr:>3} dB -> sai lech trung binh (MSE) = {mse:.4f}")
