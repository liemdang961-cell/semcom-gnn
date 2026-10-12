"""
Tuan 9 (tiep) - So sanh mo hinh KHONG phong ve (checkpoint_week6.pt) voi
mo hinh CO phong ve Adversarial Training (checkpoint_week9.pt), cung bi
tan cong FGSM o cung 5 muc epsilon va cung SNR=10dB nhu Bang 3.6 cu, de
dien vao Bang 3.7 khoa luan ("hieu qua cua co che phong ve").

Dung LAI dung cac ham tu attack.py (fgsm_perturb_z, run_pass, compute_bleu,
semantic_similarity...) de dam bao THUAT TOAN TAN CONG dung y het Tuan 7-8,
chi khac o checkpoint nao duoc dem ra thu.

Chay: python compare_defense.py
"""

import torch
import matplotlib.pyplot as plt

from semantic_encoder import SemanticEncoder
from decoder import SemanticDecoder
from graph_builder import WebNLGGraphBuilder
from attack import run_pass, build_graphs, EPSILONS

SNR_DB_FIXED = 10  # giong Bang 3.6 cu, chi 1 muc SNR on dinh de so sanh de doc
CHECKPOINTS = {
    "Khong phong ve (Tuan 6)": "checkpoints/checkpoint_week6.pt",
    "Co phong ve - Adversarial Training (Tuan 9)": "checkpoints/checkpoint_week9.pt",
}


def load_model(path, device):
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
        raise RuntimeError(f"{path} khong co 'test_triples'.")
    return encoder, decoder, entity_vocab, relation_vocab, idx_to_entity, idx_to_relation, checkpoint["test_triples"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dang chay tren: {device}\n")

    builder = WebNLGGraphBuilder(device=str(device))
    all_results = {}

    for label, path in CHECKPOINTS.items():
        print(f"=== {label} ({path}) ===")
        encoder, decoder, entity_vocab, relation_vocab, idx_to_entity, idx_to_relation, test_triples = \
            load_model(path, device)
        test_graphs = build_graphs(builder, test_triples)
        print(f"  Tap test: {len(test_graphs)} do thi hop le.")

        clean = run_pass(encoder, decoder, builder, test_graphs, entity_vocab, relation_vocab,
                          idx_to_entity, idx_to_relation, device, epsilon=None, snr_db=SNR_DB_FIXED)
        acc_clean = clean["accuracy"]
        print(f"  Acc_clean = {acc_clean * 100:.2f}% | F1 = {clean['f1']:.4f}")

        results = {"asr": [], "accuracy": [], "f1": [], "bleu": [], "bertscore_approx": []}
        for eps in EPSILONS:
            r = run_pass(encoder, decoder, builder, test_graphs, entity_vocab, relation_vocab,
                          idx_to_entity, idx_to_relation, device, epsilon=eps, snr_db=SNR_DB_FIXED)
            asr = max(0.0, (acc_clean - r["accuracy"]) / acc_clean) if acc_clean > 0 else 0.0
            results["asr"].append(asr)
            results["accuracy"].append(r["accuracy"])
            results["f1"].append(r["f1"])
            results["bleu"].append(r["bleu"])
            results["bertscore_approx"].append(r["bertscore_approx"])
            print(f"    epsilon={eps:.2f} | ASR: {asr * 100:5.2f}% | Accuracy: {r['accuracy'] * 100:5.2f}% | "
                  f"F1: {r['f1']:.4f} | BLEU: {r['bleu']:.4f} | BERTScore: {r['bertscore_approx']:.4f}")

        all_results[label] = {"acc_clean": acc_clean, **results}
        print()

    # ==================== Bang 3.7 (dien vao khoa luan) ====================
    print("\n==== Bang 3.7 - So sanh ASR(%) truoc/sau phong ve, theo epsilon (SNR=10dB) ====")
    labels = list(CHECKPOINTS.keys())
    header = f"{'epsilon':>8} | " + " | ".join(f"{l[:30]:>30}" for l in labels)
    print(header)
    for i, eps in enumerate(EPSILONS):
        row = f"{eps:>8.2f} | " + " | ".join(
            f"{all_results[l]['asr'][i] * 100:29.2f}%" for l in labels
        )
        print(row)

    # ==================== Bieu do so sanh ====================
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metric_titles = [
        ("asr", "ASR (%) theo epsilon", 100),
        ("f1", "F1-score (sau tan cong) theo epsilon", 1),
        ("bleu", "BLEU theo epsilon", 1),
        ("bertscore_approx", "\"BERTScore\" xap xi theo epsilon", 1),
    ]
    flat_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    colors = ["r", "g"]
    for (key, title, scale), ax in zip(metric_titles, flat_axes):
        for color, label in zip(colors, labels):
            ys = [v * scale for v in all_results[label][key]]
            ax.plot(EPSILONS, ys, marker="o", color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("epsilon")
        ax.grid(True)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"So sanh truoc/sau Adversarial Training, tan cong FGSM tai SNR={SNR_DB_FIXED}dB - Tuan 9")
    fig.tight_layout()
    fig.savefig("defense_comparison_plot.png")
    print("\nDa luu bieu do so sanh vao 'defense_comparison_plot.png'.")


if __name__ == "__main__":
    main()
