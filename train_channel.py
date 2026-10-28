"""
Huấn luyện hệ thống Encoder -> Kênh -> Decoder với KÊNH TÙY CHỌN, dùng để tạo
2 checkpoint mới cho phần so sánh theo yêu cầu của cô:

  1) GNN + SeCom trên kênh Rayleigh (huấn luyện QUA kênh Rayleigh):
         python train_channel.py --channel rayleigh
         -> checkpoints/checkpoint_rayleigh.pt

  2) "GNN thường" (huấn luyện SẠCH, không qua kênh nhiễu; sau đó đem đi qua kênh
     lúc đánh giá để xem độ chính xác tụt thế nào):
         python train_channel.py --channel none
         -> checkpoints/checkpoint_gnn_clean.pt

Dùng lại build_vocab / make_targets / cấu hình từ train.py (KHÔNG sửa train.py),
cùng seed 42 và cùng cách chia 80/10/10 => tập test GIỐNG HỆT checkpoint_week6.pt,
nên so sánh giữa các checkpoint là công bằng.

Chạy thử nhanh (chỉ để kiểm tra code chạy được, KHÔNG dùng số liệu này):
         python train_channel.py --channel rayleigh --max-samples 300 --max-epochs 2 --out checkpoints/test.pt
"""
import argparse
import os
import random

import torch
import torch.nn as nn

from graph_builder import WebNLGGraphBuilder, load_webnlg_triples
from semantic_encoder import SemanticEncoder
from decoder import SemanticDecoder
from channel import get_channel, CHANNEL_DISPLAY
import train as base  # dùng lại build_vocab, make_targets và các hằng số cấu hình

DEVICE = base.DEVICE
VAL_SEED = 1234  # cố định nhiễu ở pha validation để early stopping ổn định


def run_epoch(graphs_with_triples, encoder, decoder, optimizer, node_loss_fn, relation_loss_fn,
              entity_vocab, relation_vocab, channel_fn, snr_db, train: bool):
    encoder.train(mode=train)
    decoder.train(mode=train)
    total_loss, count = 0.0, 0
    total_steps = len(graphs_with_triples)

    if not train:
        torch.manual_seed(VAL_SEED)
    if train:
        optimizer.zero_grad()

    for step, (graph, triples) in enumerate(graphs_with_triples):
        if train and total_steps > 1000 and (step + 1) % 500 == 0:
            print(f"    ... đã xử lý {step + 1}/{total_steps} câu trong epoch này")
        graph = graph.to(DEVICE)
        node_targets, relation_targets = base.make_targets(
            graph, triples, entity_vocab, relation_vocab, DEVICE)

        with torch.set_grad_enabled(train):
            z = encoder(graph)
            z_noisy = channel_fn(z, snr_db)
            node_logits, relation_logits = decoder(z_noisy)

            loss_node = node_loss_fn(node_logits, node_targets)
            loss_rel = relation_loss_fn(
                relation_logits.reshape(-1, relation_logits.shape[-1]), relation_targets.reshape(-1)
            )
            loss = loss_node + loss_rel

        if train:
            (loss / base.ACCUM_STEPS).backward()
            if (step + 1) % base.ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        count += 1

    if train and count % base.ACCUM_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(count, 1)


def default_out(channel):
    return "checkpoints/checkpoint_gnn_clean.pt" if channel == "none" else f"checkpoints/checkpoint_{channel}.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", choices=["awgn", "rayleigh", "none"], required=True)
    parser.add_argument("--snr", type=float, default=base.SNR_DB,
                        help="SNR (dB) dùng khi huấn luyện, mặc định giống train.py (10 dB)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-samples", type=int, default=base.MAX_SAMPLES)
    parser.add_argument("--max-epochs", type=int, default=base.MAX_EPOCHS)
    args = parser.parse_args()

    out_path = args.out or default_out(args.channel)
    channel_fn = get_channel(args.channel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.manual_seed(42)

    label = (f"Huấn luyện qua kênh {CHANNEL_DISPLAY[args.channel]}"
             + ("" if args.channel == "none" else f", SNR={args.snr}dB") + " (80/10/10)")
    print(f"Đang chạy trên: {DEVICE}")
    print(f"Kênh huấn luyện: {CHANNEL_DISPLAY[args.channel]} | SNR={args.snr}dB | lưu vào {out_path}")
    print("Đang tải dữ liệu WebNLG (cần mạng internet)...")
    all_triples = load_webnlg_triples(split="train", max_samples=args.max_samples)
    if not all_triples:
        raise RuntimeError("Không tải được dữ liệu. Kiểm tra kết nối mạng / thư viện 'datasets'.")
    print(f"Đã tải {len(all_triples)} câu.")

    entity_vocab, relation_vocab = base.build_vocab(all_triples)
    print(f"Số thực thể: {len(entity_vocab)} | Số quan hệ: {len(relation_vocab)}")

    builder = WebNLGGraphBuilder(device=DEVICE)
    graphs_with_triples = []
    for triples in all_triples:
        try:
            g = builder.triples_to_graph(triples)
        except ValueError:
            continue
        graphs_with_triples.append((g, triples))

    # Cùng seed và cùng thứ tự với train.py -> cùng tập test
    random.seed(42)
    random.shuffle(graphs_with_triples)
    n_total = len(graphs_with_triples)
    split_train = int(n_total * base.TRAIN_RATIO)
    split_val = int(n_total * (base.TRAIN_RATIO + base.VAL_RATIO))
    train_set = graphs_with_triples[:split_train]
    val_set = graphs_with_triples[split_train:split_val]
    test_set = graphs_with_triples[split_val:]
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test (giữ riêng): {len(test_set)}")

    encoder = SemanticEncoder(in_dim=builder.embed_dim).to(DEVICE)
    decoder = SemanticDecoder(
        embed_dim=128, num_entities=len(entity_vocab), num_relations=len(relation_vocab)
    ).to(DEVICE)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=base.LR)
    node_loss_fn = nn.CrossEntropyLoss()
    relation_loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.max_epochs + 1):
        train_loss = run_epoch(train_set, encoder, decoder, optimizer, node_loss_fn, relation_loss_fn,
                               entity_vocab, relation_vocab, channel_fn, args.snr, train=True)
        val_loss = run_epoch(val_set, encoder, decoder, optimizer, node_loss_fn, relation_loss_fn,
                             entity_vocab, relation_vocab, channel_fn, args.snr, train=False)
        print(f"[Epoch {epoch}/{args.max_epochs}] train_loss = {train_loss:.4f} | val_loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "label": label,
                "channel": args.channel,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "entity_vocab": entity_vocab,
                "relation_vocab": relation_vocab,
                "snr_db_trained": args.snr,
                "num_samples_trained": len(all_triples),
                "test_triples": [triples for _, triples in test_set],
            }, out_path)
            print(f"  -> val_loss giảm, đã lưu checkpoint vào {out_path}")
        else:
            epochs_no_improve += 1
            print(f"  -> val_loss không giảm ({epochs_no_improve}/{base.EARLY_STOP_PATIENCE})")
            if epochs_no_improve >= base.EARLY_STOP_PATIENCE:
                print(f"Early Stopping tại epoch {epoch}.")
                break

    print(f"Huấn luyện xong. val_loss tốt nhất = {best_val_loss:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
