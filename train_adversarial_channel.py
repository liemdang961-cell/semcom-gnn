"""
Huấn luyện đối kháng (Adversarial Training, FGSM) với KÊNH TÙY CHỌN - bản của
train_adversarial.py (Tuần 9) nhưng cho phép huấn luyện qua kênh Rayleigh.

Giữ NGUYÊN thuật toán của train_adversarial.py (dùng lại chính hàm
fgsm_perturb_z_for_training, build_vocab, make_targets, EPSILON_TRAIN=0.1,
cùng seed 42/chia 80/10/10, huấn luyện từ đầu như bản AWGN). Chỉ khác: cả nhánh
"sạch" lẫn nhánh "bị tấn công" đều đi qua kênh được chọn thay vì luôn là AWGN.
Không ghi đè train_adversarial.py và checkpoint_week9.pt.

Chạy:
    python train_adversarial_channel.py --channel rayleigh
    -> checkpoints/checkpoint_rayleigh_adv.pt

Sau đó chạy:  python compare_defense_channel.py --channel rayleigh
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
import train_adversarial as base  # dùng lại hàm FGSM, vocab, targets, hằng số

DEVICE = base.DEVICE
VAL_SEED = 1234


def run_epoch(graphs_with_triples, encoder, decoder, optimizer, node_loss_fn, relation_loss_fn,
              entity_vocab, relation_vocab, channel_fn, snr_db, epsilon, train: bool):
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
            loss_clean = node_loss_fn(node_logits, node_targets) + relation_loss_fn(
                relation_logits.reshape(-1, relation_logits.shape[-1]), relation_targets.reshape(-1)
            )

            if train:
                # Kẻ tấn công chèn FGSM vào z (trước kênh), rồi kênh vật lý vẫn tác động như bình thường
                z_adv = base.fgsm_perturb_z_for_training(
                    decoder, z, node_targets, relation_targets, node_loss_fn, relation_loss_fn, epsilon
                )
                z_adv_noisy = channel_fn(z_adv, snr_db)
                node_logits_adv, relation_logits_adv = decoder(z_adv_noisy)
                loss_adv = node_loss_fn(node_logits_adv, node_targets) + relation_loss_fn(
                    relation_logits_adv.reshape(-1, relation_logits_adv.shape[-1]),
                    relation_targets.reshape(-1)
                )
                loss = loss_clean + loss_adv
            else:
                loss = loss_clean  # val_loss chỉ đo hiệu năng sạch, như bản AWGN

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", choices=["awgn", "rayleigh"], default="rayleigh")
    parser.add_argument("--snr", type=float, default=base.SNR_DB)
    parser.add_argument("--epsilon", type=float, default=base.EPSILON_TRAIN)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-samples", type=int, default=base.MAX_SAMPLES)
    parser.add_argument("--max-epochs", type=int, default=base.MAX_EPOCHS)
    args = parser.parse_args()

    out_path = args.out or f"checkpoints/checkpoint_{args.channel}_adv.pt"
    channel_fn = get_channel(args.channel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.manual_seed(42)

    label = (f"Adversarial Training (FGSM, epsilon={args.epsilon} lúc train) "
             f"qua kênh {CHANNEL_DISPLAY[args.channel]}, SNR={args.snr}dB")
    print(f"Đang chạy trên: {DEVICE}")
    print(f"{label} | lưu vào {out_path}\n")
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

    # Cùng seed 42 như train.py / train_adversarial.py -> cùng tập test
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
                               entity_vocab, relation_vocab, channel_fn, args.snr, args.epsilon, train=True)
        val_loss = run_epoch(val_set, encoder, decoder, optimizer, node_loss_fn, relation_loss_fn,
                             entity_vocab, relation_vocab, channel_fn, args.snr, args.epsilon, train=False)
        print(f"[Epoch {epoch}/{args.max_epochs}] train_loss(sạch+đối kháng) = {train_loss:.4f} | "
              f"val_loss(sạch) = {val_loss:.4f}")

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
                "epsilon_adv_trained": args.epsilon,
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
