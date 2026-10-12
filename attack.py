"""
Tuan 7-8 - Mo hinh de doa (Threat Model) va tan cong doi khang FGSM, DUNG
KHOP voi Muc 2.4 / 2.5 / 3.4 da viet san trong khoa luan.

CAP NHAT TUAN 8 (dung nguyen van de cuong): "Thuc nghiem tan cong voi
nhieu muc cuong do khac nhau; danh gia anh huong cua adversarial attack
theo SNR, BLEU, BERTScore/Semantic Similarity va Attack Success Rate."
-> Tuan 7 da lam xong phan "nhieu muc cuong do" (5 muc epsilon) nhung
CHI o 1 muc SNR co dinh (10dB). Tuan 8 mo rong: chay LAI toan bo phep thu
o CA 9 muc SNR (giong Bang 3.5, tu -6dB den 18dB) nhan voi 5 muc epsilon
-> tao ra 1 bang/ma tran 2 chieu (SNR x epsilon), xem tan cong FGSM gay
hai nhieu hon hay it hon khi kenh truyen tot/xau khac nhau.

===================== MO HINH DE DOA (THREAT MODEL, Muc 2.4.1) =========
Ke tan cong dong vai 1 thuc the "dung giua" (Man-in-the-Middle) tren KENH
TRUYEN VO TUYEN: chan vector ngu nghia z (dau ra cua Semantic Encoder,
tuc la cai THUC SU duoc truyen di) va chen them nhieu doi khang delta
TRUOC KHI tin hieu toi duoc bo nhan hop phap.

QUAN TRONG: nhieu doi khang duoc chen vao z (dau ra Encoder, tren duong
truyen) - KHONG PHAI vao dac trung tho 384 chieu truoc Encoder (do la noi
ke tan cong KHONG co quyen truy cap, vi no nam ben trong may phat, truoc
khi du lieu duoc dua len kenh truyen). Day la diem de nham lan, ban dau
mot phien ban cua file nay da code nham vao dac trung tho - da sua lai
cho dung voi vi du so minh hoa z=[0.9,1.7] -> z'=[0.8,1.8] o Muc 2.4.2.

Kich ban white-box (Muc 2.4.1): ke tan cong biet het kien truc + trong so
cua Encoder/Decoder, nen tinh duoc chinh xac gradient cua loss de tan
cong hieu qua nhat co the (truong hop nguy hiem nhat, lam co so danh gia
chat che nhat do ben vung cua he thong).

===================== THUAT TOAN FGSM (Muc 2.4.2) =======================
    z' = z + epsilon * sign( d(Loss)/dz )

trong do Loss la Cross-Entropy cua bo giai ma (Decoder), epsilon la ngan
sach nhieu (cang lon cang de bi phat hien nhung cang gay hai). Day la tan
cong 1-buoc (single-step gradient ascent), khac han nhieu AWGN ngau nhien
vi no duoc tinh toan CO CHU DICH theo dung huong lam loss tang nhanh nhat.

Kich ban danh gia (Muc 3.4.2): tai 1 muc SNR vat ly on dinh (SNR=10dB,
dung chung voi SNR_DB da dung luc train o Tuan 6), so sanh 2 truong hop:
  - "Sach": z chi di qua kenh AWGN binh thuong (SNR=10dB), KHONG bi tan cong.
  - "Bi tan cong": z BI CHEN THEM nhieu FGSM (z -> z') roi MOI di qua
    cung kenh AWGN (SNR=10dB) do - vi ke tan cong chi "chen them" nhieu
    doi khang vao kenh truyen vat ly van dang ton tai san, khong thay the
    no (Muc 3.4: "kenh truyen bi chen vector tan cong doi khang").

===================== CHI SO DO (Bang 3.6 khoa luan) =====================
  - ASR (Attack Success Rate), dung DUNG cong thuc da ghi o Muc 3.2.2:
        ASR = max(0, (Acc_clean - Acc_adv) / Acc_clean)
    voi Acc_clean la do chinh xac Node Classifier khi KHONG bi tan cong
    (epsilon = 0, chi co AWGN), Acc_adv la do chinh xac khi BI tan cong
    FGSM voi ngan sach epsilon tuong ung. Day la cong thuc muc GIAM do
    chinh xac TOAN CUC, khac voi kieu "dem tung node doi dung->sai" ma
    ban dau file nay dung nham - da sua lai cho dung.
  - F1-score: F1 (macro) cua Node Classifier SAU khi bi tan cong.
  - BLEU: so sanh van ban "tai dung lai" tu bo ba goc va tu bo ba model
    doan duoc SAU tan cong (tu viet BLEU n-gram don gian, khong can cai
    them thu vien ngoai).
  - "BERTScore": khoa luan goi la BERTScore nhung de khong phai cai them
    goi bert-score rieng (phai tai model BERT-base khac, ton dung luong),
    nhom TAI SU DUNG chinh model MiniLM da co san (dung de nhung cau
    trong graph_builder.py) de tinh do tuong dong cosine giua 2 cau - la
    1 PHIEN BAN DON GIAN HOA, can ghi ro trong Muc 3.2.2 khi bao ve.

Chay: python attack.py
"""

import math
from collections import Counter

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from semantic_encoder import SemanticEncoder
from decoder import SemanticDecoder
from channel import awgn_channel
from graph_builder import WebNLGGraphBuilder

CHECKPOINT_PATH = "checkpoints/checkpoint_week6.pt"
SNR_DB = 10  # muc SNR mac dinh (Bang 3.6 cu, giu lai de tham khao/doi chieu)
SNR_LEVELS = [-6, -3, 0, 3, 6, 9, 12, 15, 18]  # Tuan 8: dung chung voi Bang 3.5
EPSILONS = [0.01, 0.05, 0.1, 0.15, 0.2]  # ngan sach nhieu doi khang (Bang 3.6/3.8 khoa luan)


def make_targets(graph, triples, entity_vocab, relation_vocab, device):
    """Giong make_targets() trong train.py - tao nhan dung cho 1 do thi."""
    idx_map = {name: i for i, name in enumerate(graph.entities)}
    node_targets = torch.tensor(
        [entity_vocab[name] for name in graph.entities], dtype=torch.long, device=device
    )
    n = len(graph.entities)
    none_idx = len(relation_vocab)
    relation_targets = torch.full((n, n), none_idx, dtype=torch.long, device=device)
    for s, p, o in triples:
        si = idx_map[WebNLGGraphBuilder._clean(s)]
        oi = idx_map[WebNLGGraphBuilder._clean(o)]
        relation_targets[si, oi] = relation_vocab[WebNLGGraphBuilder._clean(p)]
    return node_targets, relation_targets


def fgsm_perturb_z(decoder, z, node_targets, relation_targets, epsilon):
    """FGSM dung DUNG Muc 2.4.2: chen nhieu vao z (dau ra Encoder, vector
    THUC SU duoc truyen tren kenh), khong phai vao dac trung tho truoc
    Encoder. z phai la tensor da tach khoi do thi tinh toan cua Encoder
    (de khong vo tinh lan truyen nguoc ca vao trong so Encoder)."""
    z = z.detach().clone().requires_grad_(True)
    node_logits, relation_logits = decoder(z)

    loss = F.cross_entropy(node_logits, node_targets) + F.cross_entropy(
        relation_logits.reshape(-1, relation_logits.shape[-1]), relation_targets.reshape(-1)
    )
    decoder.zero_grad(set_to_none=True)
    loss.backward()

    z_adv = (z + epsilon * z.grad.sign()).detach()
    return z_adv


def reconstruct_triples(node_preds, relation_logits, idx_to_entity, idx_to_relation, none_idx):
    n = node_preds.shape[0]
    pred_entities = [idx_to_entity.get(i, "?") for i in node_preds.tolist()]
    relation_pred = relation_logits.argmax(dim=-1)
    triples = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_idx = relation_pred[i, j].item()
            if r_idx != none_idx:
                triples.append((pred_entities[i], idx_to_relation.get(r_idx, "?"), pred_entities[j]))
    return triples


def triples_to_text(triples):
    return " . ".join(f"{s} {p} {o}" for s, p, o in triples)


def compute_bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """BLEU n-gram don gian (toi da 4-gram), tu viet - khong can cai nltk."""
    ref_tokens, cand_tokens = reference.split(), candidate.split()
    if not ref_tokens or not cand_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
        cand_ngrams = Counter(tuple(cand_tokens[i:i + n]) for i in range(len(cand_tokens) - n + 1))
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum(min(c, ref_ngrams.get(g, 0)) for g, c in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        precisions.append(overlap / total if total > 0 else 0.0)

    if min(precisions) == 0:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(cand_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
    return bp * geo_mean


def semantic_similarity(embedder, text_a: str, text_b: str) -> float:
    """"BERTScore" xap xi: do tuong dong cosine giua 2 cau, dung CHINH
    model MiniLM da co san (xem ghi chu dau file)."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    emb = embedder.encode([text_a, text_b], convert_to_tensor=True)
    sim = torch.nn.functional.cosine_similarity(emb[0:1], emb[1:2]).item()
    return sim


def build_graphs(builder, triples_list):
    graphs = []
    for triples in triples_list:
        try:
            g = builder.triples_to_graph(triples)
        except ValueError:
            continue
        graphs.append((g, triples))
    return graphs


def run_pass(encoder, decoder, builder, graphs_with_triples, entity_vocab, relation_vocab,
             idx_to_entity, idx_to_relation, device, epsilon=None, snr_db=SNR_DB):
    """Chay 1 luot danh gia tren toan bo tap test.

    epsilon=None  -> truong hop "sach": z chi qua kenh AWGN, khong tan cong.
    epsilon=X     -> truong hop "bi tan cong": z bi chen nhieu FGSM (ngan
                     sach X) TRUOC KHI qua kenh AWGN (Muc 3.4.2).
    snr_db        -> muc SNR vat ly cua kenh truyen AWGN luc do (Tuan 8:
                     quet qua nhieu muc, giong Bang 3.5, thay vi co dinh).
    """
    none_idx = len(relation_vocab)
    total_correct, total_nodes = 0, 0
    tp, fp, fn = {}, {}, {}
    bleu_scores, sim_scores = [], []

    for graph, triples in graphs_with_triples:
        graph = graph.to(device)
        node_targets, relation_targets = make_targets(graph, triples, entity_vocab, relation_vocab, device)

        with torch.no_grad():
            z = encoder(graph)

        if epsilon is not None:
            # Ke tan cong chen nhieu FGSM vao z (vector dang truyen), roi
            # KENH VAT LY (AWGN) van tiep tuc tac dong len z' nhu binh thuong.
            z = fgsm_perturb_z(decoder, z, node_targets, relation_targets, epsilon)

        with torch.no_grad():
            z_channel = awgn_channel(z, snr_db)
            node_logits, relation_logits = decoder(z_channel)

        pred = node_logits.argmax(dim=-1)
        total_correct += (pred == node_targets).sum().item()
        total_nodes += node_targets.numel()

        for true_c, pred_c in zip(node_targets.tolist(), pred.tolist()):
            if true_c == pred_c:
                tp[true_c] = tp.get(true_c, 0) + 1
            else:
                fn[true_c] = fn.get(true_c, 0) + 1
                fp[pred_c] = fp.get(pred_c, 0) + 1

        sentence_gt = triples_to_text(triples)
        pred_triples = reconstruct_triples(pred, relation_logits, idx_to_entity, idx_to_relation, none_idx)
        sentence_pred = triples_to_text(pred_triples)
        bleu_scores.append(compute_bleu(sentence_gt, sentence_pred))
        sim_scores.append(semantic_similarity(builder.embedder, sentence_gt, sentence_pred))

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

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0

    return {"accuracy": accuracy, "f1": macro_f1, "bleu": avg_bleu, "bertscore_approx": avg_sim}


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
    idx_to_entity = {i: n for n, i in entity_vocab.items()}
    idx_to_relation = {i: n for n, i in relation_vocab.items()}

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
        print("Checkpoint nay chua co 'test_triples' (co the la checkpoint cu). Can checkpoint_week6.pt.")
        return

    test_triples = checkpoint["test_triples"]
    builder = WebNLGGraphBuilder(device=str(device))
    test_graphs = build_graphs(builder, test_triples)
    print(f"Dung tap test giu rieng tu luc train: {len(test_graphs)} do thi hop le.\n")

    # Tuan 8: quet CA 9 muc SNR (giong Bang 3.5) NHAN voi 5 muc epsilon.
    # Voi MOI muc SNR: chay "sach" 1 lan de lay Acc_clean RIENG cho muc SNR
    # do (vi kenh xau/tot khac nhau thi Acc_clean cung khac nhau), roi moi
    # tinh ASR cua tung epsilon SO VOI dung Acc_clean cua muc SNR do.
    # -> ma tran ket qua 2 chieu: SNR_LEVELS x EPSILONS.
    results = {}  # results[snr][eps] = {"asr", "accuracy", "f1", "bleu", "bertscore_approx"}
    acc_clean_per_snr = {}

    for snr in SNR_LEVELS:
        print(f"\n=== SNR = {snr} dB ===")
        clean = run_pass(encoder, decoder, builder, test_graphs, entity_vocab, relation_vocab,
                          idx_to_entity, idx_to_relation, device, epsilon=None, snr_db=snr)
        acc_clean = clean["accuracy"]
        acc_clean_per_snr[snr] = acc_clean
        print(f"  Acc_clean(SNR={snr}dB) = {acc_clean * 100:.2f}% | F1 = {clean['f1']:.4f} | "
              f"BLEU = {clean['bleu']:.4f} | BERTScore(xap xi) = {clean['bertscore_approx']:.4f}")

        results[snr] = {}
        for eps in EPSILONS:
            r = run_pass(encoder, decoder, builder, test_graphs, entity_vocab, relation_vocab,
                          idx_to_entity, idx_to_relation, device, epsilon=eps, snr_db=snr)
            # ASR = max(0, (Acc_clean - Acc_adv) / Acc_clean) - Muc 3.2.2, nhung
            # gio Acc_clean la cua DUNG muc SNR nay, khong phai muc SNR=10dB co dinh.
            asr = max(0.0, (acc_clean - r["accuracy"]) / acc_clean) if acc_clean > 0 else 0.0
            results[snr][eps] = {
                "asr": asr, "accuracy": r["accuracy"], "f1": r["f1"],
                "bleu": r["bleu"], "bertscore_approx": r["bertscore_approx"],
            }
            print(
                f"    epsilon={eps:.2f} | ASR: {asr * 100:5.2f}% | "
                f"Accuracy: {r['accuracy'] * 100:5.2f}% | F1: {r['f1']:.4f} | "
                f"BLEU: {r['bleu']:.4f} | BERTScore(xap xi): {r['bertscore_approx']:.4f}"
            )

    # ==================== Bang tong hop dang ma tran (Tuan 8) ====================
    print("\n\n==== Bang 3.8 (de xuat) - ASR(%) theo SNR (hang) x epsilon (cot) ====")
    header = "SNR\\eps".rjust(8) + "".join(f"{eps:>9.2f}" for eps in EPSILONS)
    print(header)
    for snr in SNR_LEVELS:
        row = f"{snr:>8}" + "".join(f"{results[snr][eps]['asr'] * 100:9.2f}" for eps in EPSILONS)
        print(row)

    print("\n==== Bang chi tiet day du (ASR/Accuracy/F1/BLEU/BERTScore moi cap SNR-epsilon) ====")
    print(f"{'SNR':>5} | {'epsilon':>7} | {'ASR':>7} | {'Accuracy':>9} | {'F1':>7} | {'BLEU':>7} | {'BERTScore':>10}")
    for snr in SNR_LEVELS:
        for eps in EPSILONS:
            r = results[snr][eps]
            print(
                f"{snr:>5} | {eps:>7.2f} | {r['asr'] * 100:6.2f}% | {r['accuracy'] * 100:8.2f}% | "
                f"{r['f1']:7.4f} | {r['bleu']:7.4f} | {r['bertscore_approx']:10.4f}"
            )

    # ==================== Bieu do ====================
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(SNR_LEVELS) - 1, 1)) for i in range(len(SNR_LEVELS))]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    metric_axes = [
        ("asr", axes[0, 0], "ASR (Ty le tan cong thanh cong) theo epsilon, tung muc SNR", "ASR (%)", 100),
        ("f1", axes[0, 1], "F1-score (sau tan cong) theo epsilon, tung muc SNR", "F1", 1),
        ("bleu", axes[1, 0], "BLEU theo epsilon, tung muc SNR", "BLEU", 1),
        ("bertscore_approx", axes[1, 1], "\"BERTScore\" xap xi theo epsilon, tung muc SNR", "Do tuong dong", 1),
    ]
    for key, ax, title, ylabel, scale in metric_axes:
        for i, snr in enumerate(SNR_LEVELS):
            ys = [results[snr][eps][key] * scale for eps in EPSILONS]
            ax.plot(EPSILONS, ys, marker="o", color=colors[i], label=f"{snr}dB")
        ax.set_title(title)
        ax.set_xlabel("epsilon")
        ax.set_ylabel(ylabel)
        ax.grid(True)
    axes[0, 0].legend(title="SNR", fontsize=7, ncol=2)
    fig.suptitle("Tan cong doi khang FGSM vao z, quet qua nhieu muc SNR - Tuan 8")
    fig.tight_layout()
    fig.savefig("fgsm_attack_plot.png")
    print("\nDa luu bieu do duong (4 chi so x nhieu muc SNR) vao 'fgsm_attack_plot.png'.")

    # Heatmap rieng cho ASR - de nhin ngay vung "nguy hiem nhat" (SNR thap + epsilon cao)
    asr_matrix = [[results[snr][eps]["asr"] * 100 for eps in EPSILONS] for snr in SNR_LEVELS]
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im = ax2.imshow(asr_matrix, cmap="Reds", aspect="auto")
    ax2.set_xticks(range(len(EPSILONS))); ax2.set_xticklabels([f"{e:.2f}" for e in EPSILONS])
    ax2.set_yticks(range(len(SNR_LEVELS))); ax2.set_yticklabels([f"{s}dB" for s in SNR_LEVELS])
    ax2.set_xlabel("epsilon"); ax2.set_ylabel("SNR")
    ax2.set_title("ASR (%) theo SNR x epsilon")
    for i in range(len(SNR_LEVELS)):
        for j in range(len(EPSILONS)):
            ax2.text(j, i, f"{asr_matrix[i][j]:.1f}", ha="center", va="center", fontsize=8)
    fig2.colorbar(im, ax=ax2, label="ASR (%)")
    fig2.tight_layout()
    fig2.savefig("fgsm_attack_heatmap.png")
    print("Da luu heatmap ASR (SNR x epsilon) vao 'fgsm_attack_heatmap.png'.")


if __name__ == "__main__":
    main()
