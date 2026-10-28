"""
So sánh KHÔNG phòng vệ vs CÓ phòng vệ (Adversarial Training) trên KÊNH TÙY CHỌN -
bản của compare_defense.py. Dùng lại attack.run_pass (qua attack_channel) nên
thuật toán tấn công đúng y hệt Tuần 7-8; chỉ đổi kênh và checkpoint.

Không ghi đè compare_defense.py.

Chạy (sau khi có checkpoint_rayleigh.pt và checkpoint_rayleigh_adv.pt):
    python compare_defense_channel.py --channel rayleigh
    python compare_defense_channel.py --channel rayleigh --snrs 10 -6   # thêm mức nhiễu cao nhất

-> điền Bảng 3.7 / 3.8 cho kênh mới (ASR, Accuracy, F1, BLEU, BERTScore trước/sau phòng vệ).
"""
import argparse
import json

import torch
import matplotlib.pyplot as plt

import attack as base
from attack_channel import use_channel, seeded_run_pass, load_for_attack
from graph_builder import WebNLGGraphBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", choices=["awgn", "rayleigh"], default="rayleigh")
    parser.add_argument("--before", default=None, help="checkpoint KHÔNG phòng vệ")
    parser.add_argument("--after", default=None, help="checkpoint CÓ phòng vệ (adversarial training)")
    parser.add_argument("--snrs", type=float, nargs="+", default=[10],
                        help="các mức SNR (dB) dùng khi tấn công, mặc định 10 giống Bảng 3.7 cũ")
    args = parser.parse_args()

    before = args.before or ("checkpoints/checkpoint_week6.pt" if args.channel == "awgn"
                             else f"checkpoints/checkpoint_{args.channel}.pt")
    after = args.after or ("checkpoints/checkpoint_week9.pt" if args.channel == "awgn"
                           else f"checkpoints/checkpoint_{args.channel}_adv.pt")
    checkpoints = {"Không phòng vệ": before, "Có phòng vệ (Adversarial Training)": after}
    tag = args.channel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên: {device} | Kênh: {args.channel}\n")
    use_channel(args.channel)
    builder = WebNLGGraphBuilder(device=str(device))
    EPSILONS = base.EPSILONS

    all_results = {}  # all_results[snr][label] = {...}
    for label, path in checkpoints.items():
        print(f"=== {label} ({path}) ===")
        (encoder, decoder, entity_vocab, relation_vocab, idx_to_entity, idx_to_relation,
         test_triples, ckpt) = load_for_attack(path, device)
        if ckpt.get("channel", "awgn") != args.channel:
            print(f"  CẢNH BÁO: checkpoint huấn luyện qua kênh '{ckpt.get('channel', 'awgn')}', "
                  f"đang đánh giá qua kênh '{args.channel}'.")
        test_graphs = base.build_graphs(builder, test_triples)
        print(f"  Tập test: {len(test_graphs)} đồ thị hợp lệ.")

        for snr in args.snrs:
            clean = seeded_run_pass(encoder, decoder, builder, test_graphs, entity_vocab, relation_vocab,
                                    idx_to_entity, idx_to_relation, device, epsilon=None, snr_db=snr)
            acc_clean = clean["accuracy"]
            print(f"  [SNR={snr}dB] Acc_clean = {acc_clean * 100:.2f}% | F1 = {clean['f1']:.4f}")
            res = {"acc_clean": acc_clean, "f1_clean": clean["f1"],
                   "asr": [], "accuracy": [], "f1": [], "bleu": [], "bertscore_approx": []}
            for eps in EPSILONS:
                r = seeded_run_pass(encoder, decoder, builder, test_graphs, entity_vocab, relation_vocab,
                                    idx_to_entity, idx_to_relation, device, epsilon=eps, snr_db=snr)
                asr = max(0.0, (acc_clean - r["accuracy"]) / acc_clean) if acc_clean > 0 else 0.0
                res["asr"].append(asr)
                res["accuracy"].append(r["accuracy"])
                res["f1"].append(r["f1"])
                res["bleu"].append(r["bleu"])
                res["bertscore_approx"].append(r["bertscore_approx"])
                print(f"      epsilon={eps:.2f} | ASR: {asr * 100:5.2f}% | Accuracy: {r['accuracy'] * 100:5.2f}% | "
                      f"F1: {r['f1']:.4f} | BLEU: {r['bleu']:.4f} | BERTScore: {r['bertscore_approx']:.4f}")
            all_results.setdefault(snr, {})[label] = res
        print()

    labels = list(checkpoints.keys())
    with open(f"results_defense_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"channel": args.channel, "checkpoints": checkpoints, "epsilons": EPSILONS,
                   "results": {str(s): v for s, v in all_results.items()}}, f, ensure_ascii=False, indent=2)

    for snr in args.snrs:
        print(f"\n==== ASR(%) trước/sau phòng vệ theo epsilon (kênh {args.channel}, SNR={snr}dB) ====")
        print(f"{'epsilon':>8} | " + " | ".join(f"{l[:34]:>34}" for l in labels))
        for i, eps in enumerate(EPSILONS):
            print(f"{eps:>8.2f} | " + " | ".join(f"{all_results[snr][l]['asr'][i] * 100:33.2f}%" for l in labels))
        print(f"{'Acc_clean':>8} | " + " | ".join(f"{all_results[snr][l]['acc_clean'] * 100:33.2f}%" for l in labels))

    # Biểu đồ: mỗi mức SNR một hình 2x2
    for snr in args.snrs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        metric_titles = [("asr", "ASR (%) theo epsilon", 100), ("f1", "F1-score sau tấn công", 1),
                         ("bleu", "BLEU theo epsilon", 1), ("bertscore_approx", "\"BERTScore\" xấp xỉ", 1)]
        for (key, title, scale), ax in zip(metric_titles, [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]):
            for color, label in zip(["r", "g"], labels):
                ax.plot(EPSILONS, [v * scale for v in all_results[snr][label][key]],
                        marker="o", color=color, label=label)
            ax.set_title(title); ax.set_xlabel("epsilon"); ax.grid(True)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Trước/sau Adversarial Training, FGSM qua kênh {args.channel}, SNR={snr}dB")
        fig.tight_layout()
        fname = f"defense_comparison_{tag}_snr{int(snr)}.png"
        fig.savefig(fname)
        print(f"Đã lưu {fname}")
    print(f"Đã lưu results_defense_{tag}.json")


if __name__ == "__main__":
    main()
