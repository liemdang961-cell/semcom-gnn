"""
Tuan 6 - Danh gia he thong (Encoder -> Channel AWGN -> Decoder) qua nhieu
muc SNR, dung CHINH BO CHECKPOINT da huan luyen o train.py (Tuan 6).

So voi ban cu cua Kien, ban nay sua 4 diem:

1. KHONG gan cung num_entities=283, num_relations=50 nua - doc THANG tu
   checkpoint['entity_vocab'] / checkpoint['relation_vocab'], dam bao
   decoder duoc khoi tao dung kich thuoc da hoc (du checkpoint sau nay
   duoc train lai voi vocab khac, code van chay dung, khong phai sua tay).

2. KHONG tai lai 300 cau rieng tu WebNLG "test" split nua - dung dung
   checkpoint['test_triples'], la 10% da duoc GIU RIENG tu luc train.py
   chia 80/10/10 (Bang 3.1 khoa luan). Ly do: bo "test" chinh thuc cua
   WebNLG la mot nguon du lieu khac, vocab (ten thuc the/quan he) co the
   lech voi tu dien ma model da hoc, khong khop voi con so 80/10/10 da
   cong bo trong khoa luan.

3. Them tinh Accuracy va F1 (macro) cho Node Classifier - ban cu chi
   tinh MSE (do lech vat ly do kenh AWGN gay ra), chua danh gia model co
   doan DUNG thuc the hay khong.

4. Them tinh PSNR va SSIM tren vector embedding (Muc 3.2.2 khoa luan),
   va doi muc SNR khao sat thanh [-6,-3,0,3,6,9,12,15,18] (buoc 3dB,
   dung nhu khoang SNR da cong bo o Bang 3.5 khoa luan va o slider
   SNR trong gradio_app.py), thay vi [-5,0,5,10,15,20] nhu ban cu.

LUU Y: PSNR/SSIM o day tinh tren ma tran embedding [N, 128] cua tung do
thi (coi nhu 1 "tin hieu" can danh gia do meo), KHONG phai PSNR/SSIM
tren anh pixel - vi day la he thong Semantic Communication truyen vector
ngu nghia, khong truyen anh. Day la cach don gian hoa hop ly cho pham vi
khoa luan (ghi ro trong Muc 3.2.2), SSIM tinh toan cuc tren ca ma tran
(khong truot qua tung o nho/window nhu SSIM anh goc).

Chay: python evaluate.py
"""

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from semantic_encoder import SemanticEncoder
from decoder import SemanticDecoder
from channel import awgn_channel
from graph_builder import WebNLGGraphBuilder

CHECKPOINT_PATH = "checkpoints/checkpoint_week6.pt"
# Muc SNR khao sat (dB), buoc 3dB - dung theo Bang 3.5 khoa luan / slider gradio_app.py
SNR_LEVELS = [-6, -3, 0, 3, 6, 9, 12, 15, 18]


def compute_psnr(original: torch.Tensor, noisy: torch.Tensor) -> float:
    """PSNR = 10*log10(MAX^2 / MSE), MAX la bien do tin hieu goc (max-min)."""
    mse = F.mse_loss(noisy, original).item()
    if mse == 0:
        return float("inf")
    data_range = (original.max() - original.min()).item()
    if data_range == 0:
        return float("inf")
    return 10 * math.log10((data_range ** 2) / mse)


def compute_ssim(original: torch.Tensor, noisy: torch.Tensor) -> float:
    """SSIM (Wang et al.) tinh TOAN CUC tren ca ma tran embedding cua 1 do
    thi (khong chia o nho nhu SSIM anh goc) - don gian hoa hop ly vi day
    la vector ngu nghia, khong phai anh (Muc 3.2.2 khoa luan)."""
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
    """Doi tung bo ba thanh do thi, bo qua cau nao loi (giong train.py)."""
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


def evaluate_snr(encoder, decoder, graphs_with_triples, entity_vocab, snr_levels, device):
    encoder.eval()
    decoder.eval()

    results = {"mse": [], "psnr": [], "ssim": [], "accuracy": [], "f1": []}

    with torch.no_grad():
        for snr in snr_levels:
            total_mse = total_psnr = total_ssim = 0.0
            total_correct = total_nodes = 0
            n_graphs = 0
            # Tich luy TP/FP/FN theo tung lop thuc the de tinh F1-macro
            # TREN CAC LOP THUC SU XUAT HIEN trong luot danh gia nay
            # (vocab day du co the toi vai nghin lop, da so khong xuat
            # hien trong 10% test nen khong tinh vao F1 de tranh sai lech).
            tp, fp, fn = {}, {}, {}

            for graph, _triples in graphs_with_triples:
                graph = graph.to(device)
                node_targets = make_node_targets(graph, entity_vocab, device)

                z = encoder(graph)
                z_noisy = awgn_channel(z, snr_db=snr)
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

            results["mse"].append(avg_mse)
            results["psnr"].append(avg_psnr)
            results["ssim"].append(avg_ssim)
            results["accuracy"].append(accuracy)
            results["f1"].append(macro_f1)

            print(
                f"SNR: {snr:3d} dB | MSE: {avg_mse:.4f} | PSNR: {avg_psnr:6.2f} dB | "
                f"SSIM: {avg_ssim:.4f} | Accuracy: {accuracy * 100:5.2f}% | F1(macro): {macro_f1:.4f}"
            )

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dang chay tren: {device}")

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    except Exception as e:
        print(f"Loi khi nap checkpoint: {e}")
        return

    entity_vocab = checkpoint["entity_vocab"]
    relation_vocab = checkpoint["relation_vocab"]

    # Khoi tao dung kich thuoc tu dien DA HOC - khong gan cung so nhu ban cu
    encoder = SemanticEncoder().to(device)
    decoder = SemanticDecoder(
        embed_dim=128, num_entities=len(entity_vocab), num_relations=len(relation_vocab)
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    print(
        f"Da load checkpoint '{checkpoint.get('label', CHECKPOINT_PATH)}' thanh cong! "
        f"({len(entity_vocab)} thuc the, {len(relation_vocab)} quan he)"
    )

    if "test_triples" not in checkpoint:
        print(
            "Checkpoint nay CHUA co 'test_triples' (co the la checkpoint cu hon, "
            "vd checkpoint_week5.pt). Can checkpoint duoc luu boi train.py ban moi "
            "(Tuan 6) de co du lieu test dung theo ty le 80/10/10."
        )
        return

    test_triples = checkpoint["test_triples"]
    print(
        f"\nDung tap test da duoc GIU RIENG tu luc huan luyen "
        f"({len(test_triples)} cau, dung 10% theo Bang 3.1 khoa luan - "
        f"KHONG tai lai tu WebNLG nhu ban cu)."
    )

    builder = WebNLGGraphBuilder(device=str(device))
    test_graphs = build_graphs(builder, test_triples)
    print(f"Da dung duoc {len(test_graphs)} do thi hop le tu tap test.")

    print("\nBat dau danh gia qua cac muc SNR...")
    results = evaluate_snr(encoder, decoder, test_graphs, entity_vocab, SNR_LEVELS, device)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(SNR_LEVELS, results["mse"], marker="o", color="r")
    axes[0, 0].set_title("MSE theo SNR")
    axes[0, 0].set_xlabel("SNR (dB)")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(True)

    axes[0, 1].plot(SNR_LEVELS, results["psnr"], marker="o", color="b")
    axes[0, 1].set_title("PSNR theo SNR")
    axes[0, 1].set_xlabel("SNR (dB)")
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].grid(True)

    axes[1, 0].plot(SNR_LEVELS, results["ssim"], marker="o", color="g")
    axes[1, 0].set_title("SSIM theo SNR")
    axes[1, 0].set_xlabel("SNR (dB)")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].grid(True)

    axes[1, 1].plot(SNR_LEVELS, results["accuracy"], marker="o", color="m", label="Accuracy")
    axes[1, 1].plot(SNR_LEVELS, results["f1"], marker="s", color="c", label="F1 (macro)")
    axes[1, 1].set_title("Accuracy & F1 (Node Classifier) theo SNR")
    axes[1, 1].set_xlabel("SNR (dB)")
    axes[1, 1].set_ylabel("Ty le")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    fig.suptitle("Danh gia chat luong Semantic Communication theo SNR (Tuan 6 - du lieu full)")
    fig.tight_layout()
    fig.savefig("snr_evaluation_plot.png")
    print("\nDa chay xong! Da luu bieu do vao file 'snr_evaluation_plot.png'.")

    print("\n==== Bang tong hop (dien vao Bang 3.5 khoa luan) ====")
    print(f"{'SNR(dB)':>8} | {'Accuracy':>9} | {'F1-score':>9} | {'PSNR':>9} | {'SSIM':>8}")
    for i, snr in enumerate(SNR_LEVELS):
        print(
            f"{snr:>8} | {results['accuracy'][i] * 100:8.2f}% | {results['f1'][i]:9.4f} | "
            f"{results['psnr'][i]:7.2f}dB | {results['ssim'][i]:8.4f}"
        )


if __name__ == "__main__":
    main()
