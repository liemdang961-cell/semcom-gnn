"""
Tuan 6 - Vong lap huan luyen: noi Encoder -> Channel -> Decoder
thanh 1 he thong hoc duoc, huan luyen tren bo ba WebNLG.

Khac voi ban Tuan 5 (300 cau, chi chia 90/10 train/val): ban nay huan
luyen tren TOAN BO du lieu WebNLG tai duoc (MAX_SAMPLES = None), chia
dung ty le 80/10/10 (train/val/test) nhu Bang 3.1 khoa luan.

LUU Y QUAN TRONG (da ghi trong build_vocab()): voi full du lieu, so
thuc the trong tu dien se tang len rat nhieu (toi da ~17.000-18.000 cau
hop le, uoc tinh vai chuc nghin thuc the), nen so lop cuoi cua Node
Classifier se rat lon. Dieu nay khong lam crash chuong trinh (PyTorch
xu ly lop Linear vai chuc nghin class binh thuong) nhung co the khien
mo hinh kho hoc chinh xac tung thuc the hiem gap - day la gioi han can
neu ro trong phan "Han che" cua khoa luan, khong phai loi code.

Vong lap train hien tai xu ly TUNG DO THI MOT (khong gop batch thuc su
qua DataLoader), nen voi full du lieu moi epoch se cham hon nhieu so
voi ban 300 cau - nen chay tren GPU (Colab T4 tro len) va kien nhan
cho; Early Stopping (patience=15) se tu dung som neu val_loss khong
con cai thien, khong nhat thiet chay het MAX_EPOCHS=100.

Chay: python train.py
"""

import os
import random
import torch
import torch.nn as nn

from graph_builder import WebNLGGraphBuilder, load_webnlg_triples
from semantic_encoder import SemanticEncoder
from channel import awgn_channel
from decoder import SemanticDecoder

# ----------------- Cau hinh (se tinh chinh lai o Tuan 10 - "toi uu mo hinh") -----------------
MAX_SAMPLES = None      # None = lay TOAN BO du lieu train cua WebNLG (full data, Bang 3.1)
TRAIN_RATIO = 0.8       # 80% train
VAL_RATIO = 0.1         # 10% validation, phan con lai (10%) la test (Bang 3.1)
SNR_DB = 10             # tam thoi huan luyen o 1 muc SNR co dinh (giong Hello et al. huan luyen o 14dB)
MAX_EPOCHS = 100        # gioi han TREN, it khi chay het vi Early Stopping se tu dung som hon
EARLY_STOP_PATIENCE = 15  # neu val_loss khong giam sau 15 epoch lien tiep -> tu dung (Muc 2.2.1)
LR = 1e-3
ACCUM_STEPS = 8         # gom 8 cau moi lan cap nhat trong so, gia lap "batch size"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_LABEL = "Tuan 6 - full du lieu (80/10/10)"  # doi tiep khi chay lai o tuan sau
CHECKPOINT_PATH = "checkpoints/checkpoint_week6.pt"    # doi ten file tuong ung, KHONG de trung ten cu


def build_vocab(all_triples):
    """Xay tu dien thuc the va quan he TREN TOAN BO du lieu huan luyen
    (khac voi tu dien nho trong gradio_app.py chi lay tu 1 cau nhap vao).

    LUU Y: khi tang MAX_SAMPLES len toi 20.000 (Bang 3.1), so thuc the se
    tang len rat nhieu (co the vai chuc nghin). Phan loai node tren mot
    tu dien qua lon se rat kho hoc va ton bo nho - day la diem can quyet
    dinh them khi mo rong quy mo (co the phai gioi han tu dien hoac doi
    chien luoc decode), ghi chu lai de ban bao cho nhom biet truoc.
    """
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
    """Tao nhan dung (ground truth) cho 1 do thi: nhan node va ma tran quan he N x N."""
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


def run_epoch(graphs_with_triples, encoder, decoder, optimizer, node_loss_fn, relation_loss_fn,
              entity_vocab, relation_vocab, train: bool):
    encoder.train(mode=train)
    decoder.train(mode=train)
    total_loss, count = 0.0, 0
    total_steps = len(graphs_with_triples)

    if train:
        optimizer.zero_grad()

    for step, (graph, triples) in enumerate(graphs_with_triples):
        # Full du lieu chay lau, in tien do moi 500 cau de biet chuong
        # trinh van dang chay chu khong bi treo (bo qua neu tap rat nho).
        if train and total_steps > 1000 and (step + 1) % 500 == 0:
            print(f"    ... da xu ly {step + 1}/{total_steps} cau trong epoch nay")
        graph = graph.to(DEVICE)
        node_targets, relation_targets = make_targets(graph, triples, entity_vocab, relation_vocab, DEVICE)

        with torch.set_grad_enabled(train):
            z = encoder(graph)
            z_noisy = awgn_channel(z, SNR_DB)
            node_logits, relation_logits = decoder(z_noisy)

            loss_node = node_loss_fn(node_logits, node_targets)
            loss_rel = relation_loss_fn(
                relation_logits.reshape(-1, relation_logits.shape[-1]), relation_targets.reshape(-1)
            )
            loss = loss_node + loss_rel

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

    random.seed(42)
    random.shuffle(graphs_with_triples)
    n_total = len(graphs_with_triples)
    split_train = int(n_total * TRAIN_RATIO)
    split_val = int(n_total * (TRAIN_RATIO + VAL_RATIO))
    train_set = graphs_with_triples[:split_train]
    val_set = graphs_with_triples[split_train:split_val]
    test_set = graphs_with_triples[split_val:]
    print(f"Tap huan luyen: {len(train_set)} cau | Tap kiem dinh: {len(val_set)} cau | "
          f"Tap kiem thu (giu rieng, danh cho evaluate.py): {len(test_set)} cau")

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
        print(f"[Epoch {epoch}/{MAX_EPOCHS}] train_loss = {train_loss:.4f} | val_loss = {val_loss:.4f}")

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
                "num_samples_trained": len(all_triples),
                # Luu dung tap test (10%) da tach o tren, de evaluate.py dung
                # LAI CHINH XAC cung 1 tap nay khi danh gia - dam bao khop
                # voi ty le 80/10/10 da cong bo o Bang 3.1 khoa luan, thay
                # vi dung tap "test" rieng cua WebNLG (vocab co the lech).
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
