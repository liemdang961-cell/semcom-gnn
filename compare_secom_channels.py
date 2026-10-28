"""
So sánh "GNN thường" và "GNN + SeCom" - CẢ HAI đều đi qua kênh truyền (đúng ý cô):

  - GNN thường            : huấn luyện SẠCH (không nhiễu), rồi đem qua kênh khi đánh giá
                            (checkpoint_gnn_clean.pt, từ train_channel.py --channel none)
  - GNN + SeCom (AWGN)    : huấn luyện qua kênh AWGN          (checkpoint_week6.pt, đã có)
  - GNN + SeCom (Rayleigh): huấn luyện qua kênh Rayleigh      (checkpoint_rayleigh.pt)

Mỗi mô hình được đo Accuracy/F1 qua CẢ HAI kênh (AWGN, Rayleigh) ở các mức SNR
của Bảng 3.5, trên CÙNG tập test và CÙNG chuỗi nhiễu (seed theo từng SNR/kênh).
Thay cho compare_secom.py cũ (cũ coi "chưa SeCom" là không qua kênh - chưa đúng ý cô).

Chạy:  python compare_secom_channels.py
Checkpoint nào chưa có sẽ bị bỏ qua kèm cảnh báo.
"""
import json
import os

import torch
import matplotlib.pyplot as plt

from graph_builder import WebNLGGraphBuilder
import evaluate_channels as ev

MODELS = {
    "GNN thường (huấn luyện sạch)": "checkpoints/checkpoint_gnn_clean.pt",
    "GNN + SeCom (huấn luyện AWGN)": "checkpoints/checkpoint_week6.pt",
    "GNN + SeCom (huấn luyện Rayleigh)": "checkpoints/checkpoint_rayleigh.pt",
}
STYLES = {"GNN thường (huấn luyện sạch)": ("k", "^"),
          "GNN + SeCom (huấn luyện AWGN)": ("r", "o"),
          "GNN + SeCom (huấn luyện Rayleigh)": ("b", "s")}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên: {device}\n")
    builder = WebNLGGraphBuilder(device=str(device))

    results = {}
    ref_test, test_graphs = None, None
    for label, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[BỎ QUA] {label}: chưa có {path}")
            continue
        print(f"\n########## {label} ({path}) ##########")
        encoder, decoder, entity_vocab, relation_vocab, test_triples, ckpt = ev.load_model(path, device)
        if ref_test is None:
            ref_test = test_triples
            test_graphs = ev.build_graphs(builder, ref_test)
            print(f"Tập test dùng chung: {len(test_graphs)} đồ thị hợp lệ.")
        elif test_triples != ref_test:
            print("CẢNH BÁO: tập test của checkpoint này KHÁC tập test của checkpoint đầu tiên "
                  "(có thể train với dữ liệu/seed khác) -> so sánh có thể không công bằng.")
        results[label] = ev.evaluate_channels_with_accuracy(
            encoder, decoder, test_graphs, entity_vocab, ev.SNR_LEVELS, device)

    if not results:
        print("Không có checkpoint nào để so sánh.")
        return

    with open("results_secom_channels.json", "w", encoding="utf-8") as f:
        json.dump({"snr_levels": ev.SNR_LEVELS, "results": results}, f, ensure_ascii=False, indent=2)

    labels = list(results.keys())
    for ch_name in ev.CHANNELS:
        for metric, scale, title in (("accuracy", 100, "Accuracy (%)"), ("f1", 1, "F1-score (macro)")):
            print(f"\n==== {title} qua kênh {ch_name} ====")
            print(f"{'SNR(dB)':>8} | " + " | ".join(f"{l[:32]:>32}" for l in labels))
            for i, snr in enumerate(ev.SNR_LEVELS):
                vals = " | ".join(
                    (f"{results[l][ch_name][metric][i] * scale:31.2f}%" if metric == "accuracy"
                     else f"{results[l][ch_name][metric][i]:32.4f}") for l in labels)
                print(f"{snr:>8} | {vals}")

    for metric, scale, ylabel, fname in (("accuracy", 100, "Accuracy (%)", "secom_channels_accuracy.png"),
                                         ("f1", 1, "F1-score (macro)", "secom_channels_f1.png")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
        for ax, ch_name in zip(axes, ev.CHANNELS):
            for label in labels:
                color, marker = STYLES.get(label, ("g", "d"))
                ax.plot(ev.SNR_LEVELS, [v * scale for v in results[label][ch_name][metric]],
                        marker=marker, color=color, label=label)
            ax.set_title(f"Qua kênh {ch_name}")
            ax.set_xlabel("SNR (dB)")
            ax.grid(True)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"GNN thường vs GNN + SeCom: {ylabel} theo SNR, qua hai kênh truyền")
        fig.tight_layout()
        fig.savefig(fname)
        print(f"Đã lưu {fname}")
    print("Đã lưu results_secom_channels.json")


if __name__ == "__main__":
    main()
