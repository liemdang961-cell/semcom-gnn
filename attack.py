"""
Tuan 7 - Mo hinh de doa (Threat Model) va tan cong doi khang FGSM len
Semantic Encoder (Muc 2.4 / 3.2.3 khoa luan).

===================== MO HINH DE DOA (THREAT MODEL) =====================
- Attacker la ai: 1 thuc the co the can thiep tren DUONG TRUYEN, truoc khi
  vector ngu nghia di vao kenh AWGN (vd: chen nhieu/sai lech vao chinh
  dac trung dau vao ma Encoder nhan duoc, hoac biet truoc kien truc +
  trong so cua Encoder/Decoder - kich ban "white-box").
- Attacker lam gi: thay vi nhieu NGAU NHIEN nhu kenh AWGN (Tuan 5-6),
  attacker tinh toan 1 nhieu CO CHU DICH, nho nhat co the (gioi han boi
  ngan sach epsilon) nhung lam LOSS cua mo hinh tang len nhieu nhat -
  tuc la co y lam Decoder doan sai thuc the/quan he.
- Attacker KHONG the lam gi: khong thay doi duoc trong so cua Encoder/
  Decoder da huan luyen (khong tan cong vao qua trinh train), chi tac
  dong vao DU LIEU DAU VAO/duong truyen tai thoi diem suy luan (inference).
- Phuong phap tan cong trien khai: FGSM (Fast Gradient Sign Method -
  Goodfellow et al., 2015), 1 phuong phap tan cong doi khang pho bien,
  dung dao ham (gradient) cua loss theo dau vao de tao nhieu:

      x_adv = x + epsilon * sign( d(Loss)/dx )

  voi x la dac trung node tho (384 chieu tu MiniLM) TRUOC khi vao
  Semantic Encoder, epsilon la "ngan sach" nhieu (cang lon thi nhieu
  cang manh nhung cang de bi phat hien). Day la tan cong 1-buoc (single-
  step), phu hop lam baseline danh gia muc do de bi tan cong cua he
  thong, truoc khi xet toi phong thu (Adversarial Training - de o Tuan
  sau neu can).
===========================================================================

4 chi so do cho moi muc epsilon (Bang 3.6 khoa luan):
  - ASR (Attack Success Rate): trong so NHUNG NODE ma truoc tan cong model
    dang doan DUNG, co bao nhieu % sau khi bi tan cong lai doan SAI - do
    dung muc do "tan cong thanh cong lam hong ket qua dung".
  - F1-score: F1 (macro) cua Node Classifier SAU khi bi tan cong.
  - BLEU: so sanh chuoi van ban "tai dung lai" tu bo ba goc va tu bo ba
    model doan duoc SAU tan cong (tu viet BLEU n-gram don gian, khong
    can cai them thu vien ngoai).
  - "BERTScore": do tuong dong ngu nghia can 1 mo hinh ngon ngu rieng;
    de khong phai cai them goi bert-score (ton dung luong, phai tai model
    rieng), nhom TAI SU DUNG chinh model MiniLM da co san (dung de nhung
    cau trong graph_builder.py) de tinh do tuong dong cosine giua 2 cau -
    ban chat la 1 PHIEN BAN DON GIAN HOA cua BERTScore, ghi ro trong
    Muc 3.2.2 khoa luan (khong phai BERTScore "chuan" dung BERT-base).

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
SNR_DB = 10  # muc SNR co dinh khi tan cong, dung chung muc da dung luc train (Muc 2.2.1)
EPSILONS = [0.01, 0.05, 0.1, 0.15, 0.2]  # ngan sach nhieu doi khang (Bang 3.6 khoa luan)


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


def fgsm_perturb(encoder, decoder, graph, node_targets, relation_targets, epsilon):
    """Tao nhieu doi khang FGSM tren dac trung node tho (graph.x), truoc
    khi vao Encoder. Tra ve x_adv (da tach khoi do thi tinh toan)."""
    original_x = graph.x
    x = original_x.clone().detach().requires_grad_(True)
    graph.x = x

    z = encoder(graph)
    z_noisy = awgn_channel(z, SNR_DB)
    node_logits, relation_logits = decoder(z_noisy)

    loss = F.cross_entropy(node_logits, node_targets) + F.cross_entropy(
        relation_logits.reshape(-1, relation_logits.shape[-1]), relation_targets.reshape(-1)
    )
    encoder.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)
    loss.backward()

    x_adv = (x + epsilon * x.grad.sign()).detach()
    graph.x = original_x  # khoi phuc do thi goc, tranh lam hong du lieu cho vong lap sau
    return x_adv


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


def evaluate_epsilon(encoder, decoder, builder, graphs_with_triples, entity_vocab, relation_vocab,
                      idx_to_entity, idx_to_relation, epsilon, device):
    none_idx = len(relation_vocab)

    total_correct_before = 0
    total_attack_success = 0
    total_correct_after = 0
    total_nodes = 0
    tp, fp, fn = {}, {}, {}
    bleu_scores, sim_scores = [], []

    for graph, triples in graphs_with_triples:
        graph = graph.to(device)
        node_targets, relation_targets = make_targets(graph, triples, entity_vocab, relation_vocab, device)

        # 1. Du doan TRUOC khi tan cong (lam co so tinh ASR)
        with torch.no_grad():
            z_clean = encoder(graph)
            node_logits_clean, _ = decoder(awgn_channel(z_clean, SNR_DB))
        pred_clean = node_logits_clean.argmax(dim=-1)

        # 2. Tao nhieu doi khang FGSM, roi du doan LAI voi dau vao da bi tan cong
        x_adv = fgsm_perturb(encoder, decoder, graph, node_targets, relation_targets, epsilon)
        original_x = graph.x
        graph.x = x_adv
        with torch.no_grad():
            z_adv = encoder(graph)
            node_logits_adv, relation_logits_adv = decoder(awgn_channel(z_adv, SNR_DB))
        graph.x = original_x
        pred_adv = node_logits_adv.argmax(dim=-1)

        # ASR: trong so node TRUOC DOAN DUNG, sau tan cong co bao nhieu % doan SAI
        correct_before = (pred_clean == node_targets)
        became_wrong = correct_before & (pred_adv != node_targets)
        total_correct_before += correct_before.sum().item()
        total_attack_success += became_wrong.sum().item()

        # Accuracy/F1 SAU tan cong
        correct_after = (pred_adv == node_targets)
        total_correct_after += correct_after.sum().item()
        total_nodes += node_targets.numel()
        for true_c, pred_c in zip(node_targets.tolist(), pred_adv.tolist()):
            if true_c == pred_c:
                tp[true_c] = tp.get(true_c, 0) + 1
            else:
                fn[true_c] = fn.get(true_c, 0) + 1
                fp[pred_c] = fp.get(pred_c, 0) + 1

        # BLEU / "BERTScore" tren van ban tai dung tu bo ba
        sentence_gt = triples_to_text(triples)
        pred_triples = reconstruct_triples(pred_adv, relation_logits_adv, idx_to_entity, idx_to_relation, none_idx)
        sentence_adv = triples_to_text(pred_triples)
        bleu_scores.append(compute_bleu(sentence_gt, sentence_adv))
        sim_scores.append(semantic_similarity(builder.embedder, sentence_gt, sentence_adv))

    asr = total_attack_success / max(total_correct_before, 1)
    accuracy_after = total_correct_after / max(total_nodes, 1)

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

    return {
        "asr": asr, "accuracy_after": accuracy_after, "f1": macro_f1,
        "bleu": avg_bleu, "bertscore_approx": avg_sim,
    }


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

    print("Bat dau tan cong FGSM qua cac muc epsilon...")
    all_results = {"asr": [], "accuracy_after": [], "f1": [], "bleu": [], "bertscore_approx": []}
    for eps in EPSILONS:
        r = evaluate_epsilon(
            encoder, decoder, builder, test_graphs, entity_vocab, relation_vocab,
            idx_to_entity, idx_to_relation, eps, device,
        )
        for k in all_results:
            all_results[k].append(r[k])
        print(
            f"epsilon = {eps:.2f} | ASR: {r['asr'] * 100:5.2f}% | "
            f"Accuracy(sau tan cong): {r['accuracy_after'] * 100:5.2f}% | F1: {r['f1']:.4f} | "
            f"BLEU: {r['bleu']:.4f} | BERTScore(xap xi): {r['bertscore_approx']:.4f}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(EPSILONS, [a * 100 for a in all_results["asr"]], marker="o", color="r")
    axes[0, 0].set_title("ASR (Ty le tan cong thanh cong) theo epsilon")
    axes[0, 0].set_xlabel("epsilon"); axes[0, 0].set_ylabel("ASR (%)"); axes[0, 0].grid(True)

    axes[0, 1].plot(EPSILONS, all_results["f1"], marker="o", color="m")
    axes[0, 1].set_title("F1-score (sau tan cong) theo epsilon")
    axes[0, 1].set_xlabel("epsilon"); axes[0, 1].set_ylabel("F1"); axes[0, 1].grid(True)

    axes[1, 0].plot(EPSILONS, all_results["bleu"], marker="o", color="b")
    axes[1, 0].set_title("BLEU theo epsilon")
    axes[1, 0].set_xlabel("epsilon"); axes[1, 0].set_ylabel("BLEU"); axes[1, 0].grid(True)

    axes[1, 1].plot(EPSILONS, all_results["bertscore_approx"], marker="o", color="g")
    axes[1, 1].set_title("\"BERTScore\" xap xi theo epsilon")
    axes[1, 1].set_xlabel("epsilon"); axes[1, 1].set_ylabel("Do tuong dong"); axes[1, 1].grid(True)

    fig.suptitle("Tan cong doi khang FGSM len Semantic Encoder (Tuan 7)")
    fig.tight_layout()
    fig.savefig("fgsm_attack_plot.png")
    print("\nDa chay xong! Da luu bieu do vao file 'fgsm_attack_plot.png'.")

    print("\n==== Bang tong hop (dien vao Bang 3.6 khoa luan) ====")
    print(f"{'epsilon':>8} | {'ASR':>8} | {'BLEU':>8} | {'BERTScore':>10} | {'F1-score':>9}")
    for i, eps in enumerate(EPSILONS):
        print(
            f"{eps:>8.2f} | {all_results['asr'][i] * 100:6.2f}% | "
            f"{all_results['bleu'][i]:8.4f} | {all_results['bertscore_approx'][i]:10.4f} | "
            f"{all_results['f1'][i]:9.4f}"
        )


if __name__ == "__main__":
    main()
