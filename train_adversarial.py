"""
Tuan 9 - Co che phong ve: Adversarial Training (Muc 2.5 khoa luan), nham
tang kha nang chong chiu cua he thong truoc tan cong FGSM (Tuan 7-8).

Y TUONG (giong DUNG tinh than FGSM adversarial training cua Goodfellow et
al.): trong luc huan luyen, voi MOI cau, ngoai lan "sach" nhu train.py
(Tuan 6), them 1 lan nua: chen nhieu FGSM vao z (dung CHINH cong thuc
z' = z + epsilon * sign(dL/dz) nhu attack.py), roi cho ca 2 phien ban
(sach + bi tan cong) cung di qua Decoder, cong 2 loss lai de lan truyen
nguoc. Decoder vi vay "thay" ca truong hop binh thuong va truong hop bi
tan cong ngay trong luc hoc, nen se bot nhay cam voi FGSM hon so voi
checkpoint_week6.pt (chi hoc tren du lieu sach).

KHAC VOI attack.py: o day KHONG dung decoder.zero_grad() de lay gradient
(lam vay se xoa mat gradient dang duoc TICH LUY cho ACCUM_STEPS, gay loi
huan luyen). Thay vao do dung torch.autograd.grad() de lay rieng gradient
theo z (khong dung vao .grad cua tham so mo hinh), giu nguyen dung cong
thuc FGSM nhung khong lam hong vong lap tich luy gradient hien co.

LUU Y THIET KE (ghi ro de bao ve khi duoc hoi): z_adv duoc tao tu z da
.detach() -> khi lan truyen nguoc loss_adv, CHI Decoder duoc cap nhat tu
nhanh nay (Encoder khong nhan gradient tu nhanh adversarial, chi nhan tu
nhanh "sach" nhu binh thuong). Day la don gian hoa hop ly, chua phai
"full" adversarial training lan toi ca Encoder, nhung van dung dung cong
thuc FGSM va dung muc tieu: Decoder hoc duoc cach "chiu dung" z bi nhieu.

EPSILON_TRAIN = 0.1 co dinh (muc giua trong 5 muc da thu o Bang 3.6/3.8),
khong random moi cau - de de giai thich/doi chieu voi co.

Chay: python train_adversarial.py
Sau khi xong, dung lai attack.py (doi CHECKPOINT_PATH sang checkpoint_week9.pt)
de so sanh ASR truoc/sau phong ve -> dien vao Bang 3.7 khoa luan.
"""

import os
import random
import torch
import torch.nn as nn

from graph_builder import WebNLGGraphBuilder, load_webnlg_triples
from semantic_encoder import SemanticEncoder
from channel import awgn_channel
from decoder import SemanticDecoder

# ----------------- Cau hinh -----------------
MAX_SAMPLES = None
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SNR_DB = 10                # giong train.py Tuan 6, de so sanh cong bang
EPSILON_TRAIN = 0.1        # muc nhieu FGSM chen vao luc train (co dinh)
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LR = 1e-3
ACCUM_STEPS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_LABEL = "Tuan 9 - Adversarial Training (FGSM, epsilon=0.1 luc train)"
CHECKPOINT_PATH = "checkpoints/checkpoint_week9.pt"


def build_vocab(all_triples):
    entity_vocab, relation_vocab = {}, {}
    for triples in all_triples:
        for s, p, o in triples:
            for e in (s, o):
                ce = WebNLGGraphBuilder._clean(e)
                if ce not in entity_vocab:
                    entity_vocab[ce] = len(entity_vocab)
            cp = WebNLGGraphBuilder._clean(p)
            if cp not in relation_vocab:
                relation_vocab[cp] = len(relation_vocab)
    return entity_vocab, relation_vocab


def make_targets(graph, triples, entity_vocab, relation_vocab, device):
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


def fgsm_perturb_z_for_training(decoder, z, node_targets, relation_targets, node_loss_fn,
                                 relation_loss_fn, epsilon):
    """FGSM dung CHINH cong thuc z' = z + epsilon*sign(dL/dz) nhu attack.py,
    nhung dung torch.autograd.grad() de KHONG dong vao .grad cua Decoder
    (tranh pha hong gradient dang tich luy cho ACCUM_STEPS trong vong lap
    train chinh)."""
    z_det = z.detach().clone().requires_grad_(True)
    node_logits, relation_logits = decoder(z_det)
    loss = node_loss_fn(node_logits, node_targets) + relation_loss_fn(
        relation_logits.reshape(-1, relation_logits.shape[-1]), relation_targets.reshape(-1)
    )
    grad_z = torch.autograd.grad(loss, z_det, retain_graph=False, create_graph=False)[0]
    z_adv = (z_det + epsilon * grad_z.sign()).detach()
    return z_adv


def run_epoch(graphs_with_triples, encoder, decoder, optimizer, node_loss_fn, relation_loss_fn,
              entity_vocab, relation_vocab, train: bool):
    encoder.train(mode=train)
    decoder.train(mode=train)
    total_loss, count = 0.0, 0
    total_steps = len(graphs_with_triples)

    if train:
        optimizer.zero_grad()

    for step, (graph, triples) in enumerate(graphs_with_triples):
        if train and total_steps > 1000 and (step + 1) % 500 == 0:
            print(f"    ... da xu ly {step + 1}/{total_steps} cau trong epoch nay")
        graph = graph.to(DEVICE)
        node_targets, relation_targets = make_targets(graph, triples, entity_vocab, relation_vocab, DEVICE)

        with torch.set_grad_enabled(train):
            z = encoder(graph)
            z_noisy = awgn_channel(z, SNR_DB)
            node_logits, relation_logits = decoder(z_noisy)
            loss_clean = node_loss_fn(node_logits, node_targets) + relation_loss_fn(
                relation_logits.reshape(-1, relation_logits.shape[-1]), relation_targets.reshape(-1)
            )

            if train:
                # Chi tao mau doi khang luc TRAIN (khong lam luc validation,
                # de val_loss van do dung "hieu nang sach" nhu checkpoint cu,
                # dam bao Early Stopping so sanh cong bang voi Tuan 6).
                z_adv = fgsm_perturb_z_for_training(
                    decoder, z, node_targets, relation_targets, node_loss_fn, relation_loss_fn, EPSILON_TRAIN
                )
                z_adv_noisy = awgn_channel(z_adv, SNR_DB)
                node_logits_adv, relation_logits_adv = decoder(z_adv_noisy)
                loss_adv = node_loss_fn(node_logits_adv, node_targets) + relation_loss_fn(
                    relation_logits_adv.reshape(-1, relation_logits_adv.shape[-1]), relation_targets.reshape(-1)
                )
                loss = loss_clean + loss_adv
            else:
                loss = loss_clean

        if train:
            (loss / ACCUM_STEPS).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        count += 1

    if train and count % ACCUM_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(count, 1)


def main():
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    print(f"Dang chay tren: {DEVICE}")
    print(f"Adversarial Training: FGSM epsilon={EPSILON_TRAIN} (co dinh), SNR_DB={SNR_DB}dB\n")
    print("Dang tai du lieu WebNLG (can mang internet)...")
    all_triples = load_webnlg_triples(split="train", max_samples=MAX_SAMPLES)
    if not all_triples:
        raise RuntimeError("Khong tai duoc du lieu. Kiem tra ket noi mang / thu vien 'datasets'.")
    print(f"Da tai {len(all_triples)} cau.")

    entity_vocab, relation_vocab = build_vocab(all_triples)
    print(f"So thuc the trong tu dien: {len(entity_vocab)} | So quan he: {len(relation_vocab)}")

    builder = WebNLGGraphBuilder(device=DEVICE)
    graphs_with_triples = []
    for triples in all_triples:
        try:
            g = builder.triples_to_graph(triples)
        except ValueError:
            continue
        graphs_with_triples.append((g, triples))

    # QUAN TRONG: cung seed=42 nhu train.py (Tuan 6) -> chia 80/10/10 ra
    # DUNG CUNG tap test nhu checkpoint_week6.pt, dam bao so sanh cong bang
    # khi dem 2 checkpoint ra chay attack.py de dien Bang 3.7.
    random.seed(42)
    random.shuffle(graphs_with_triples)
    n_total = len(graphs_with_triples)
    split_train = int(n_total * TRAIN_RATIO)
    split_val = int(n_total * (TRAIN_RATIO + VAL_RATIO))
    train_set = graphs_with_triples[:split_train]
    val_set = graphs_with_triples[split_train:split_val]
    test_set = graphs_with_triples[split_val:]
    print(f"Tap huan luyen: {len(train_set)} cau | Tap kiem dinh: {len(val_set)} cau | "
          f"Tap kiem thu (giu rieng, danh cho attack.py): {len(test_set)} cau")

    encoder = SemanticEncoder(in_dim=builder.embed_dim).to(DEVICE)
    decoder = SemanticDecoder(
        embed_dim=128, num_entities=len(entity_vocab), num_relations=len(relation_vocab)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    node_loss_fn = nn.CrossEntropyLoss()
    relation_loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(
            train_set, encoder, decoder, optimizer, node_loss_fn, relation_loss_fn,
            entity_vocab, relation_vocab, train=True,
        )
        val_loss = run_epoch(
            val_set, encoder, decoder, optimizer, node_loss_fn, relation_loss_fn,
            entity_vocab, relation_vocab, train=False,
        )
        print(f"[Epoch {epoch}/{MAX_EPOCHS}] train_loss(sach+doi khang) = {train_loss:.4f} | "
              f"val_loss(sach) = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "label": CHECKPOINT_LABEL,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "entity_vocab": entity_vocab,
                "relation_vocab": relation_vocab,
                "snr_db_trained": SNR_DB,
                "epsilon_adv_trained": EPSILON_TRAIN,
                "num_samples_trained": len(all_triples),
                "test_triples": [triples for _, triples in test_set],
            }, CHECKPOINT_PATH)
            print(f"  -> val_loss giam, da luu checkpoint vao {CHECKPOINT_PATH}")
        else:
            epochs_no_improve += 1
            print(f"  -> val_loss khong giam ({epochs_no_improve}/{EARLY_STOP_PATIENCE} epoch lien tiep)")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early Stopping: val_loss khong giam sau {EARLY_STOP_PATIENCE} epoch, dung tai epoch {epoch}.")
                break

    print(f"Huan luyen xong. Checkpoint tot nhat: val_loss = {best_val_loss:.4f} -> {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
