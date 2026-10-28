"""
Tấn công đối kháng FGSM qua KÊNH TÙY CHỌN (AWGN hoặc Rayleigh) - bản của attack.py.

Dùng lại TOÀN BỘ thuật toán của attack.py (fgsm_perturb_z, run_pass, BLEU,
"BERTScore" xấp xỉ, ASR...) để tấn công giống hệt Tuần 7-8; chỉ đổi kênh vật lý
mà z đi qua SAU khi bị chèn nhiễu FGSM. Kịch bản vẫn là white-box: kẻ tấn công
biết encoder/decoder nhưng không biết hệ số fading H của từng lần truyền.

Không ghi đè attack.py. File ảnh/JSON kết quả có hậu tố tên kênh.

Chạy (sau khi có checkpoint_rayleigh.pt):
    python attack_channel.py --channel rayleigh
    python attack_channel.py --channel rayleigh --checkpoint checkpoints/checkpoint_rayleigh.pt
Kết quả: ma trận ASR theo SNR x epsilon -> thay Bảng 3.6 / 3.9 và Hình 3.5 cho kênh mới.
"""
import argparse
import json

import torch
import matplotlib.pyplot as plt

import attack as base
from channel import get_channel
from semantic_encoder import SemanticEncoder
from decoder import SemanticDecoder
from graph_builder import WebNLGGraphBuilder

SEED = 42


def use_channel(name: str):
    """Đổi kênh vật lý mà attack.run_pass() dùng (mặc định là AWGN).
    run_pass gọi tên `awgn_channel` trong module attack lúc chạy, nên chỉ cần gán lại."""
    base.awgn_channel = get_channel(name)


def seeded_run_pass(*args, seed=SEED, **kwargs):
    """run_pass có đặt seed: cùng chuỗi nhiễu/fading giữa các lần chạy và giữa các checkpoint."""
    torch.manual_seed(seed)
    return base.run_pass(*args, **kwargs)


def load_for_attack(path, device):
    checkpoint = torch.load(path, map_location=device)
    entity_vocab = checkpoint["entity_vocab"]
    relation_vocab = checkpoint["relation_vocab"]
    idx_to_entity = {i: n for n, i in entity_vocab.items()}
    idx_to_relation = {i: n for n, i in relation_vocab.items()}
    encoder = SemanticEncoder().to(device)
    decoder = SemanticDecoder(
        embed_dim=128, num_entities=len(entity_vocab), num_relations=len(relation_vocab)
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    if "test_triples" not in checkpoint:
        raise RuntimeError(f"{path} chưa có 'test_triples'.")
    return (encoder, decoder, entity_vocab, relation_vocab, idx_to_entity, idx_to_relation,
            checkpoint["test_triples"], checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", choices=["awgn", "rayleigh"], default="rayleigh")
    parser.add_argument("--checkpoint", default=None,
                        help="mặc định: checkpoints/checkpoint_<channel>.pt (awgn -> checkpoint_week6.pt)")
    args = parser.parse_args()
    ckpt_path = args.checkpoint or (
        "checkpoints/checkpoint_week6.pt" if args.channel == "awgn"
        else f"checkpoints/checkpoint_{args.channel}.pt")
    tag = args.channel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên: {device} | Kênh: {args.channel} | Checkpoint: {ckpt_path}")
    use_channel(args.channel)

    (encoder, decoder, entity_vocab, relation_vocab, idx_to_entity, idx_to_relation,
     test_triples, ckpt) = load_for_attack(ckpt_path, device)
    print(f"Đã load '{ckpt.get('label', ckpt_path)}' "
          f"({len(entity_vocab)} thực thể, {len(relation_vocab)} quan hệ)")
    if ckpt.get("channel", "awgn") != args.channel:
        print(f"CẢNH BÁO: checkpoint được huấn luyện qua kênh '{ckpt.get('channel', 'awgn')}' "
              f"nhưng đang tấn công qua kênh '{args.channel}'.")

    builder = WebNLGGraphBuilder(device=str(device))
    test_graphs = base.build_graphs(builder, test_triples)
    print(f"Tập test giữ riêng: {len(test_graphs)} đồ thị hợp lệ.\n")

    SNR_LEVELS, EPSILONS = base.SNR_LEVELS, base.EPSILONS
    results = {}
    for snr in SNR_LEVELS:
        print(f"\n=== SNR = {snr} dB ===")
        clean = seeded_run_pass(encoder, decoder, builder, test_graphs, entity_vocab, relation_vocab,
                                idx_to_entity, idx_to_relation, device, epsilon=None, snr_db=snr)
        acc_clean = clean["accuracy"]
        print(f"  Acc_clean(SNR={snr}dB) = {acc_clean * 100:.2f}% | F1 = {clean['f1']:.4f} | "
              f"BLEU = {clean['bleu']:.4f} | BERTScore(xấp xỉ) = {clean['bertscore_approx']:.4f}")
        results[snr] = {"clean": clean}
        for eps in EPSILONS:
            r = seeded_run_pass(encoder, decoder, builder, test_graphs, entity_vocab, relation_vocab,
                                idx_to_entity, idx_to_relation, device, epsilon=eps, snr_db=snr)
            asr = max(0.0, (acc_clean - r["accuracy"]) / acc_clean) if acc_clean > 0 else 0.0
            results[snr][eps] = {"asr": asr, "accuracy": r["accuracy"], "f1": r["f1"],
                                 "bleu": r["bleu"], "bertscore_approx": r["bertscore_approx"]}
            print(f"    epsilon={eps:.2f} | ASR: {asr * 100:5.2f}% | Accuracy: {r['accuracy'] * 100:5.2f}% | "
                  f"F1: {r['f1']:.4f} | BLEU: {r['bleu']:.4f} | BERTScore(xấp xỉ): {r['bertscore_approx']:.4f}")

    with open(f"results_attack_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"channel": args.channel, "checkpoint": ckpt_path, "snr_levels": SNR_LEVELS,
                   "epsilons": EPSILONS,
                   "results": {str(s): {str(k): v for k, v in d.items()} for s, d in results.items()}},
                  f, ensure_ascii=False, indent=2)

    print(f"\n\n==== ASR(%) theo SNR (hàng) x epsilon (cột) - kênh {args.channel} ====")
    print("SNR\\eps".rjust(8) + "".join(f"{eps:>9.2f}" for eps in EPSILONS))
    for snr in SNR_LEVELS:
        print(f"{snr:>8}" + "".join(f"{results[snr][eps]['asr'] * 100:9.2f}" for eps in EPSILONS))

    print("\n==== Bảng chi tiết (ASR/Accuracy/F1/BLEU/BERTScore) ====")
    print(f"{'SNR':>5} | {'epsilon':>7} | {'ASR':>7} | {'Accuracy':>9} | {'F1':>7} | {'BLEU':>7} | {'BERTScore':>10}")
    for snr in SNR_LEVELS:
        for eps in EPSILONS:
            r = results[snr][eps]
            print(f"{snr:>5} | {eps:>7.2f} | {r['asr'] * 100:6.2f}% | {r['accuracy'] * 100:8.2f}% | "
                  f"{r['f1']:7.4f} | {r['bleu']:7.4f} | {r['bertscore_approx']:10.4f}")

    # ----- Biểu đồ đường 4 chỉ số -----
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(SNR_LEVELS) - 1, 1)) for i in range(len(SNR_LEVELS))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    metric_axes = [
        ("asr", axes[0, 0], "ASR theo epsilon, từng mức SNR", "ASR (%)", 100),
        ("f1", axes[0, 1], "F1-score (sau tấn công) theo epsilon", "F1", 1),
        ("bleu", axes[1, 0], "BLEU theo epsilon", "BLEU", 1),
        ("bertscore_approx", axes[1, 1], "\"BERTScore\" xấp xỉ theo epsilon", "Độ tương đồng", 1),
    ]
    for key, ax, title, ylabel, scale in metric_axes:
        for i, snr in enumerate(SNR_LEVELS):
            ax.plot(EPSILONS, [results[snr][eps][key] * scale for eps in EPSILONS],
                    marker="o", color=colors[i], label=f"{snr}dB")
        ax.set_title(title); ax.set_xlabel("epsilon"); ax.set_ylabel(ylabel); ax.grid(True)
    axes[0, 0].legend(title="SNR", fontsize=7, ncol=2)
    fig.suptitle(f"Tấn công FGSM vào z qua kênh {args.channel}, quét SNR x epsilon")
    fig.tight_layout()
    fig.savefig(f"fgsm_attack_plot_{tag}.png")

    # ----- Heatmap ASR -----
    asr_matrix = [[results[snr][eps]["asr"] * 100 for eps in EPSILONS] for snr in SNR_LEVELS]
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im = ax2.imshow(asr_matrix, cmap="Reds", aspect="auto")
    ax2.set_xticks(range(len(EPSILONS))); ax2.set_xticklabels([f"{e:.2f}" for e in EPSILONS])
    ax2.set_yticks(range(len(SNR_LEVELS))); ax2.set_yticklabels([f"{s}dB" for s in SNR_LEVELS])
    ax2.set_xlabel("epsilon"); ax2.set_ylabel("SNR")
    ax2.set_title(f"ASR (%) theo SNR x epsilon - kênh {args.channel}")
    for i in range(len(SNR_LEVELS)):
        for j in range(len(EPSILONS)):
            ax2.text(j, i, f"{asr_matrix[i][j]:.1f}", ha="center", va="center", fontsize=8)
    fig2.colorbar(im, ax=ax2, label="ASR (%)")
    fig2.tight_layout()
    fig2.savefig(f"fgsm_attack_heatmap_{tag}.png")
    print(f"\nĐã lưu: results_attack_{tag}.json, fgsm_attack_plot_{tag}.png, fgsm_attack_heatmap_{tag}.png")


if __name__ == "__main__":
    main()
