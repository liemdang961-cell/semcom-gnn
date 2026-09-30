"""
Tuan 5 (tiep) - Vong lap huan luyen: noi Encoder -> Channel -> Decoder
thanh 1 he thong hoc duoc, huan luyen tren bo ba WebNLG.

Day la LAN CHAY THU DAU TIEN de kiem tra ca pipeline hoc duoc that (loss
giam dan), nen mac dinh chi lay MAX_SAMPLES cau (khong phai full 20.000
mau nhu Bang 3.1). Khi chay on dinh, tang MAX_SAMPLES dan len.

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
MAX_SAMPLES = 300       # so cau lay de chay thu; tang dan len khi da on dinh (toi da ~17.668 cau)
TRAIN_RATIO = 0.9
SNR_DB = 10             # tam thoi huan luyen o 1 muc SNR co dinh (giong Hello et al. huan luyen o 14dB)
MAX_EPOCHS = 100        # gioi han TREN, it khi chay het vi Early Stopping se tu dung som hon
EARLY_STOP_PATIENCE = 15  # neu val_loss khong giam sau 15 epoch lien tiep -> tu dung (Muc 2.2.1)
LR = 1e-3
ACCUM_STEPS = 8         # gom 8 cau moi lan cap nhat trong so, gia lap "batch size"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_LABEL = "Tuan 5 - 300 cau"  # doi thanh "Tuan 6 - ..." khi chay lai o tuan 6, v.v.
CHECKPOINT_PATH = "checkpoints/checkpoint_week5.pt"  # doi ten file tuong ung, KHONG de trung ten cu


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

    if train:
        optimizer.zero_grad()

    for step, (graph, triples) in enumerate(graphs_with_triples):
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
    split = int(len(graphs_with_triples) * TRAIN_RATIO)
    train_set, val_set = graphs_with_triples[:split], graphs_with_triples[split:]
    print(f"Tap huan luyen: {len(train_set)} cau | Tap kiem dinh: {len(val_set)} cau")

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
