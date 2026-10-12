"""
So sanh GNN "chua SeCom" va "co SeCom" (yeu cau rieng cua co, ngoai Bang
3.5/3.6/3.7 da co san) - dung CHINH checkpoint_week6.pt, KHONG train lai.

Dinh nghia (da xac nhan voi nguoi dung):
  - "Chua SeCom" = z (dau ra Encoder) di THANG vao Decoder, KHONG qua
    kenh truyen AWGN - tuc la khong co he thong truyen thong ngu nghia
    nao chen vao, chi thuan GNN ma hoa roi giai ma lai ngay (truong hop
    ly tuong, khong nhieu).
  - "Co SeCom" = dung pipeline SeCom thuc su: Encoder -> Channel AWGN ->
    Decoder, dung GIONG HET nhu da danh gia trong evaluate.py (Bang 3.5),
    o nhieu muc SNR khac nhau.

Muc dich: cho thay GNN tu no doan rat tot (gan nhu hoan hao) khi KHONG
phai lo ve kenh truyen, nhung khi phai truyen qua kenh thuc (co SeCom)
thi do chinh xac giam dan theo SNR - chung minh kenh truyen (khong phai
GNN yeu) la nguyen nhan gay mat mat thong tin, va la ly do can nghien
cuu he thong SeCom chiu nhieu tot (cac tuan sau).

Tai su dung CHINH cac ham tu evaluate.py (build_graphs, make_node_targets,
evaluate_snr) de dam bao nhat quan voi Bang 3.5 da co.

Chay: python compare_secom.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from semantic_encoder import SemanticEncoder
from decoder import SemanticDecoder
from graph_builder import WebNLGGraphBuilder
from evaluate import CHECKPOINT_PATH, SNR_LEVELS, build_graphs, make_node_targets, evaluate_snr


def evaluate_no_channel(encoder, decoder, graphs_with_triples, entity_vocab, device):
    """"Chua SeCom": z di thang vao Decoder, khong qua awgn_channel()."""
    encoder.eval()
    decoder.eval()
    total_correct, total_nodes = 0, 0
    tp, fp, fn = {}, {}, {}

    with torch.no_grad():
        for graph, _triples in graphs_with_triples:
            graph = graph.to(device)
            node_targets = make_node_targets(graph, entity_vocab, device)

            z = encoder(graph)
            node_preds, _ = decoder(z)  # KHONG qua kenh - day la diem khac duy nhat

            pred_labels = node_preds.argmax(dim=-1)
            total_correct += (pred_labels == node_targets).sum().item()
            total_nodes += node_targets.numel()

            for true_c, pred_c in zip(node_targets.tolist(), pred_labels.tolist()):
                if true_c == pred_c:
                    tp[true_c] = tp.get(true_c, 0) + 1
                else:
                    fn[true_c] = fn.get(true_c, 0) + 1
                    fp[pred_c] = fp.get(pred_c, 0) + 1

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

    return {"accuracy": accuracy, "f1": macro_f1}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dang chay tren: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    entity_vocab = checkpoint["entity_vocab"]
    relation_vocab = checkpoint["relation_vocab"]

    encoder = SemanticEncoder().to(device)
    decoder = SemanticDecoder(
        embed_dim=128, num_entities=len(entity_vocab), num_relations=len(relation_vocab)
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    print(f"Da load checkpoint '{checkpoint.get('label', CHECKPOINT_PATH)}'.")

    test_triples = checkpoint["test_triples"]
    builder = WebNLGGraphBuilder(device=str(device))
    test_graphs = build_graphs(builder, test_triples)
    print(f"Tap test: {len(test_graphs)} do thi hop le.\n")

    print("=== Danh gia 'CHUA SeCom' (z di thang vao Decoder, khong qua kenh) ===")
    no_channel = evaluate_no_channel(encoder, decoder, test_graphs, entity_vocab, device)
    print(f"  Accuracy = {no_channel['accuracy'] * 100:.2f}% | F1 = {no_channel['f1']:.4f}\n")

    print("=== Danh gia 'CO SeCom' (Encoder -> Channel AWGN -> Decoder), qua nhieu muc SNR ===")
    with_channel = evaluate_snr(encoder, decoder, test_graphs, entity_vocab, SNR_LEVELS, device)

    print("\n==== Bang so sanh GNN: Chua SeCom vs Co SeCom (theo SNR) ====")
    print(f"{'Truong hop':>28} | {'Accuracy':>9} | {'F1-score':>9}")
    print(f"{'Chua SeCom (khong qua kenh)':>28} | {no_channel['accuracy'] * 100:8.2f}% | {no_channel['f1']:9.4f}")
    for i, snr in enumerate(SNR_LEVELS):
        label = f"Co SeCom (SNR={snr}dB)"
        print(f"{label:>28} | {with_channel['accuracy'][i] * 100:8.2f}% | {with_channel['f1'][i]:9.4f}")

    # ==================== Bieu do ====================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].axhline(no_channel["accuracy"] * 100, color="k", linestyle="--", label="Chua SeCom (khong qua kenh)")
    axes[0].plot(SNR_LEVELS, [a * 100 for a in with_channel["accuracy"]], marker="o", color="m", label="Co SeCom")
    axes[0].set_title("Accuracy: Chua SeCom vs Co SeCom (theo SNR)")
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True); axes[0].legend()

    axes[1].axhline(no_channel["f1"], color="k", linestyle="--", label="Chua SeCom (khong qua kenh)")
    axes[1].plot(SNR_LEVELS, with_channel["f1"], marker="o", color="c", label="Co SeCom")
    axes[1].set_title("F1-score: Chua SeCom vs Co SeCom (theo SNR)")
    axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("F1")
    axes[1].grid(True); axes[1].legend()

    fig.suptitle("So sanh GNN: chua SeCom (ly tuong) vs co SeCom (qua kenh AWGN thuc)")
    fig.tight_layout()
    fig.savefig("secom_comparison_plot.png")
    print("\nDa luu bieu do vao 'secom_comparison_plot.png'.")


if __name__ == "__main__":
    main()
