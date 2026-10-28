"""
Đánh giá 1 checkpoint qua HAI kênh truyền (AWGN và Rayleigh Fading) theo nhiều
mức SNR. Dựa trên evaluate.py bản của Kiên (giữ nguyên phần tính MSE/PSNR/SSIM/
Accuracy/F1), chỉ bổ sung:
  - đặt seed cho từng (SNR, kênh) -> kết quả lặp lại được, so sánh giữa các
    checkpoint dùng CÙNG một chuỗi nhiễu (so sánh công bằng);
  - nhận checkpoint từ dòng lệnh, lưu kết quả ra file JSON để điền bảng;
  - tách hàm load_model() để compare_secom_channels.py dùng lại.

Không ghi đè evaluate.py cũ.

Chạy:
    python evaluate_channels.py                                   # checkpoint_week6.pt
    python evaluate_channels.py --checkpoint checkpoints/checkpoint_rayleigh.pt
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from semantic_encoder import SemanticEncoder
from decoder import SemanticDecoder
from channel import awgn_channel, rayleigh_channel
from graph_builder import WebNLGGraphBuilder

DEFAULT_CHECKPOINT = "checkpoints/checkpoint_week6.pt"
# Mức SNR khảo sát (dB), giống Bảng 3.5
SNR_LEVELS = [-6, -3, 0, 3, 6, 9, 12, 15, 18]
SEED = 42

CHANNELS = {
    "AWGN Channel": awgn_channel,
    "Rayleigh Fading Channel": rayleigh_channel,
}


def compute_psnr(original: torch.Tensor, noisy: torch.Tensor) -> float:
    """PSNR = 10*log10(MAX^2 / MSE)."""
    mse = F.mse_loss(noisy, original).item()
    if mse == 0:
        return float("inf")
    data_range = (original.max() - original.min()).item()
    if data_range == 0:
        return float("inf")
    return 10 * math.log10((data_range ** 2) / mse)


def compute_ssim(original: torch.Tensor, noisy: torch.Tensor) -> float:
    """SSIM toàn cục trên ma trận embedding của đồ thị."""
    data_range = (original.max() - original.min()).item()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x, mu_y = original.mean().item(), noisy.mean().item()
    var_x = original.var(unbiased=False).item()
    var_y = noisy.var(unbiased=False).item()
    cov_xy = ((original - mu_x) * (noisy - mu_y)).mean().item()

    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return numerator / denominator if denominator != 0 else 1.0


def build_graphs(builder: WebNLGGraphBuilder, triples_list):
    """Đổi từng bộ ba thành đồ thị, bỏ qua câu lỗi."""
    graphs = []
    for triples in triples_list:
        try:
            g = builder.triples_to_graph(triples)
        except ValueError:
            continue
        graphs.append((g, triples))
    return graphs


def make_node_targets(graph, entity_vocab, device):
    return torch.tensor(
        [entity_vocab[name] for name in graph.entities], dtype=torch.long, device=device
    )


def load_model(path, device):
    """Nạp encoder/decoder/vocab/test_triples từ 1 checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    entity_vocab = checkpoint["entity_vocab"]
    relation_vocab = checkpoint["relation_vocab"]

    encoder = SemanticEncoder().to(device)
    decoder = SemanticDecoder(
        embed_dim=128, num_entities=len(entity_vocab), num_relations=len(relation_vocab)
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    if "test_triples" not in checkpoint:
        raise RuntimeError(f"{path} chưa có 'test_triples' (cần checkpoint từ train.py).")
    return encoder, decoder, entity_vocab, relation_vocab, checkpoint["test_triples"], checkpoint


def evaluate_channels_with_accuracy(encoder, decoder, graphs_with_triples, entity_vocab,
                                    snr_levels, device):
    encoder.eval()
    decoder.eval()

    results = {
        name: {"mse": [], "psnr": [], "ssim": [], "accuracy": [], "f1": []}
        for name in CHANNELS.keys()
    }

    with torch.no_grad():
        for snr_idx, snr in enumerate(snr_levels):
            print(f"\n--- Đang đánh giá tại SNR = {snr:2d} dB ---")
            for ch_idx, (ch_name, ch_fn) in enumerate(CHANNELS.items()):
                # Seed theo (SNR, kênh): chạy lại hay đem checkpoint khác ra so
                # sánh đều gặp cùng một chuỗi nhiễu/fading.
                torch.manual_seed(SEED + 1000 * snr_idx + ch_idx)

                total_mse = total_psnr = total_ssim = 0.0
                total_correct = total_nodes = 0
                n_graphs = 0
                tp, fp, fn = {}, {}, {}

                for graph, _triples in graphs_with_triples:
                    graph = graph.to(device)
                    node_targets = make_node_targets(graph, entity_vocab, device)

                    # 1. Semantic Encoder
                    z = encoder(graph)

                    # 2. Truyền qua kênh nhiễu tương ứng (AWGN hoặc Rayleigh)
                    z_noisy = ch_fn(z, snr_db=snr)

                    # 3. Decoder giải mã từ tín hiệu sau kênh nhiễu
                    node_preds, _ = decoder(z_noisy)

                    total_mse += F.mse_loss(z_noisy, z).item()
                    total_psnr += compute_psnr(z, z_noisy)
                    total_ssim += compute_ssim(z, z_noisy)
                    n_graphs += 1

                    pred_labels = node_preds.argmax(dim=-1)
                    total_correct += (pred_labels == node_targets).sum().item()
                    total_nodes += node_targets.numel()

                    for true_c, pred_c in zip(node_targets.tolist(), pred_labels.tolist()):
                        if true_c == pred_c:
                            tp[true_c] = tp.get(true_c, 0) + 1
                        else:
                            fn[true_c] = fn.get(true_c, 0) + 1
                            fp[pred_c] = fp.get(pred_c, 0) + 1

                avg_mse = total_mse / max(n_graphs, 1)
                avg_psnr = total_psnr / max(n_graphs, 1)
                avg_ssim = total_ssim / max(n_graphs, 1)
                accuracy = total_correct / max(total_nodes, 1)

                classes = set(tp) | set(fp) | set(fn)
                f1_list = []
                for c in classes:
                    tp_c, fp_c, fn_c = tp.get(c, 0), fp.get(c, 0), fn.get(c, 0)
                    precision = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
                    recall = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    f1_list.append(f1)
                macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

                results[ch_name]["mse"].append(avg_mse)
                results[ch_name]["psnr"].append(avg_psnr)
                results[ch_name]["ssim"].append(avg_ssim)
                results[ch_name]["accuracy"].append(accuracy)
                results[ch_name]["f1"].append(macro_f1)

                print(
                    f"Kênh {ch_name:23s} | MSE: {avg_mse:.4f} | PSNR: {avg_psnr:6.2f}dB | "
                    f"SSIM: {avg_ssim:.4f} | Accuracy: {accuracy * 100:5.2f}% | F1: {macro_f1:.4f}"
                )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {device}")

    encoder, decoder, entity_vocab, relation_vocab, test_triples, ckpt = load_model(args.checkpoint, device)
    print(f"Đã load checkpoint '{ckpt.get('label', args.checkpoint)}' "
          f"({len(entity_vocab)} thực thể, {len(relation_vocab)} quan hệ)")
    print(f"Kênh đã dùng khi huấn luyện: {ckpt.get('channel', 'awgn (checkpoint cũ)')}")

    print(f"\nĐang tải bộ dữ liệu test ({len(test_triples)} câu)...")
    builder = WebNLGGraphBuilder(device=str(device))
    test_graphs = build_graphs(builder, test_triples)
    print(f"Đã dựng được {len(test_graphs)} đồ thị hợp lệ từ tập test.")

    print("\nBắt đầu đánh giá so sánh giữa kênh AWGN và Rayleigh Fading qua các mức SNR...")
    all_results = evaluate_channels_with_accuracy(encoder, decoder, test_graphs, entity_vocab,
                                                  SNR_LEVELS, device)

    with open(f"results_channels_{stem}.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "snr_levels": SNR_LEVELS, "results": all_results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu số liệu vào 'results_channels_{stem}.json'.")

    # === VẼ BIỂU ĐỒ SO SÁNH 2 KÊNH (2x2) ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plots = [
        ("mse", axes[0, 0], "So sánh MSE theo SNR", "MSE (càng thấp càng tốt)", 1),
        ("psnr", axes[0, 1], "So sánh PSNR theo SNR", "PSNR (dB)", 1),
        ("ssim", axes[1, 0], "So sánh SSIM theo SNR", "SSIM", 1),
        ("accuracy", axes[1, 1], "So sánh Accuracy (%) theo SNR", "Accuracy (%)", 100),
    ]
    for key, ax, title, ylabel, scale in plots:
        ax.plot(SNR_LEVELS, [v * scale for v in all_results["AWGN Channel"][key]],
                marker="o", color="r", label="AWGN")
        ax.plot(SNR_LEVELS, [v * scale for v in all_results["Rayleigh Fading Channel"][key]],
                marker="s", color="b", label="Rayleigh Fading")
        ax.set_title(title)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
    fig.suptitle("So sánh hiệu năng Semantic Communication: AWGN vs Rayleigh Fading", fontsize=14)
    fig.tight_layout()
    out_png = f"channel_accuracy_comparison_{stem}.png"
    fig.savefig(out_png)
    print(f"Đã lưu biểu đồ so sánh vào '{out_png}'.")

    # === IN BẢNG TỔNG HỢP ===
    print("\n" + "=" * 95)
    print(f"{'SNR(dB)':>8} | {'AWGN Acc':>10} | {'Rayleigh Acc':>12} | {'AWGN F1':>9} | {'Rayleigh F1':>11}")
    print("=" * 95)
    for i, snr in enumerate(SNR_LEVELS):
        a, r = all_results["AWGN Channel"], all_results["Rayleigh Fading Channel"]
        print(f"{snr:>8} | {a['accuracy'][i] * 100:9.2f}% | {r['accuracy'][i] * 100:11.2f}% | "
              f"{a['f1'][i]:9.4f} | {r['f1'][i]:11.4f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
