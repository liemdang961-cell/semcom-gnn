"""
Giao dien demo - tich hop Encoder + Kenh AWGN + Decoder.

Co them 1 o tick "Su dung AI da huan luyen de khoi phuc":
  - TAT (mac dinh): chi chay Encoder -> Channel, dung lai o do. Hien dung
    128 so da bi nhieu, KHONG co gang doan lai gi ca - minh hoa dung y
    "du lieu qua kenh nhieu that su bi hong, khong dung AI thi vo nghia".
  - BAT: nap checkpoint_week5.pt (Encoder + Decoder DA HUAN LUYEN tren 300
    cau) va cho no thu khoi phuc lai do thi.

LUU Y QUAN TRONG: checkpoint chi hoc tu 300 cau, nen Decoder chi "biet"
dung 283 thuc the va 50 quan he nam trong 300 cau do. Neu ban tu go ten
nam ngoai pham vi nay, ket qua doan se sai vi AI chua tung thay ten do
bao gio - day khong phai loi code.

Chay: python gradio_app.py
(Truoc khi chay, upload checkpoint_week5.pt vao cung thu muc qua khung
Tep ben trai Colab, neu muon dung o tick "Su dung AI".)
"""

import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
import gradio as gr

from graph_builder import WebNLGGraphBuilder
from semantic_encoder import SemanticEncoder
from channel import awgn_channel
from decoder import SemanticDecoder

# Tim checkpoint o 2 noi: thu muc hien tai (upload tay) HOAC trong Google
# Drive da duoc gan vao Colab (drive.mount). Dung file dau tien tim thay duoc.
CHECKPOINT_CANDIDATES = [
    "/content/drive/MyDrive/KhoaLuan_SemCom/checkpoint_week5.pt",
    "checkpoint_week5.pt",
    "/content/drive/MyDrive/checkpoint_week5.pt",
]
CHECKPOINT_PATH = next((p for p in CHECKPOINT_CANDIDATES if os.path.exists(p)), CHECKPOINT_CANDIDATES[0])

print("Dang nap model MiniLM va khoi tao Semantic Encoder...")
builder = WebNLGGraphBuilder(device="cpu")
untrained_encoder = SemanticEncoder(in_dim=builder.embed_dim)
untrained_encoder.eval()
print("San sang.")

# ----- Thu nap checkpoint da huan luyen (neu co) -----
trained_encoder = None
trained_decoder = None
entity_vocab, relation_vocab = None, None
idx_to_entity, idx_to_relation = None, None
checkpoint_loaded = False

if os.path.exists(CHECKPOINT_PATH):
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        entity_vocab = ckpt["entity_vocab"]
        relation_vocab = ckpt["relation_vocab"]
        idx_to_entity = {i: name for name, i in entity_vocab.items()}
        idx_to_relation = {i: name for name, i in relation_vocab.items()}

        trained_encoder = SemanticEncoder(in_dim=builder.embed_dim)
        trained_encoder.load_state_dict(ckpt["encoder"])
        trained_encoder.eval()

        trained_decoder = SemanticDecoder(
            embed_dim=128, num_entities=len(entity_vocab), num_relations=len(relation_vocab)
        )
        trained_decoder.load_state_dict(ckpt["decoder"])
        trained_decoder.eval()

        checkpoint_loaded = True
        print(f"Da nap checkpoint: {len(entity_vocab)} thuc the, {len(relation_vocab)} quan he "
              f"(huan luyen o SNR = {ckpt.get('snr_db_trained', '?')} dB).")
    except Exception as e:  # phong truong hop file loi/khong tuong thich
        print(f"Khong nap duoc checkpoint ({e}). O tick AI se khong dung duoc.")
else:
    print(f"Chua thay {CHECKPOINT_PATH} trong thu muc hien tai. O tick AI se khong dung duoc "
          f"cho den khi ban upload file nay qua khung Tep ben trai Colab.")


def parse_triples(text: str):
    triples = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def draw_graph(entities, triples, title):
    G = nx.DiGraph()
    for name in entities:
        G.add_node(name)
    for s, p, o in triples:
        G.add_edge(s, o, label=p)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "(khong co node nao)", ha="center", va="center")
        ax.axis("off")
        return fig
    pos = nx.spring_layout(G, seed=42, k=1.2)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="#8ecae6",
             node_size=1700, font_size=8, arrows=True, arrowsize=16)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def blank_fig(message: str):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=10)
    ax.axis("off")
    return fig


def run_demo(triples_text: str, snr_db: float, use_ai: bool):
    triples = parse_triples(triples_text)
    if not triples:
        empty = plt.figure()
        return empty, empty, "Chua nhap duoc bo ba nao hop le. Dinh dang: chu_the | quan_he | khach_the (moi dong 1 bo ba)."

    clean_triples = [(builder._clean(s), builder._clean(p), builder._clean(o)) for s, p, o in triples]
    graph = builder.triples_to_graph(triples)
    entities = graph.entities

    active_encoder = trained_encoder if checkpoint_loaded else untrained_encoder

    with torch.no_grad():
        z = active_encoder(graph)
        z_noisy = awgn_channel(z, snr_db)

    fig_original = draw_graph(entities, clean_triples, f"Do thi GOC ({len(entities)} node, {len(clean_triples)} canh)")
    n = len(entities)

    if not use_ai:
        # AI CHUA huan luyen: dung 1 decoder khoi tao ngau nhien (tu dien
        # thu nho, chi lay tu chinh cau nhap vao) de minh hoa AI "doan lui"
        # khi chua duoc hoc - thuong gop nhieu node lai thanh 1-2 node sai.
        local_relations = sorted(set(p for _, p, _ in clean_triples))
        random_decoder = SemanticDecoder(
            embed_dim=z_noisy.shape[1], num_entities=n, num_relations=len(local_relations)
        )
        random_decoder.eval()
        with torch.no_grad():
            node_logits, relation_logits = random_decoder(z_noisy)

        pred_entities = [entities[i] for i in node_logits.argmax(dim=-1).tolist()]
        relation_pred = relation_logits.argmax(dim=-1)
        none_idx = len(local_relations)
        reconstructed_triples = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r_idx = relation_pred[i, j].item()
                if r_idx != none_idx:
                    reconstructed_triples.append((pred_entities[i], local_relations[r_idx], pred_entities[j]))

        fig_recon = draw_graph(
            list(dict.fromkeys(pred_entities)), reconstructed_triples,
            f"Do thi KHOI PHUC - AI CHUA huan luyen (SNR = {snr_db} dB)",
        )
        summary = (
            f"SNR da chon: {snr_db} dB\n"
            f"So bo ba goc: {len(clean_triples)} | So bo ba doan duoc: {len(reconstructed_triples)}\n\n"
            "----\n"
            "AI o day CHUA duoc huan luyen (trong so ngau nhien), nen doan lung\n"
            "tung, thuong gop nhieu thuc the lai thanh 1-2 thuc the sai. Tich\n"
            "chon 'Su dung AI da huan luyen' de xem ket qua sau khi AI da hoc."
        )
        return fig_original, fig_recon, summary

    if not checkpoint_loaded:
        fig_recon = blank_fig(f"Chua tim thay {CHECKPOINT_PATH}.\nHay upload file nay vao Colab\nqua khung Tep ben trai.")
        summary = f"Khong the dung AI khoi phuc vi chua nap duoc {CHECKPOINT_PATH}."
        return fig_original, fig_recon, summary

    with torch.no_grad():
        node_logits, relation_logits = trained_decoder(z_noisy)

    pred_entity_idx = node_logits.argmax(dim=-1).tolist()
    pred_entities = [idx_to_entity.get(i, "?") for i in pred_entity_idx]

    relation_pred = relation_logits.argmax(dim=-1)
    none_idx = len(relation_vocab)
    reconstructed_triples = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_idx = relation_pred[i, j].item()
            if r_idx != none_idx:
                reconstructed_triples.append((pred_entities[i], idx_to_relation.get(r_idx, "?"), pred_entities[j]))

    fig_recon = draw_graph(
        list(dict.fromkeys(pred_entities)), reconstructed_triples,
        f"Do thi KHOI PHUC boi AI da huan luyen (SNR = {snr_db} dB)",
    )

    unknown_entities = [e for e in entities if e not in entity_vocab]
    caveat = ""
    if unknown_entities:
        caveat = (
            f"\nLUU Y: {len(unknown_entities)} thuc the ban nhap ({', '.join(unknown_entities[:3])}...) "
            f"KHONG nam trong 283 thuc the AI da hoc, nen AI chac chan doan sai cho cac thuc the nay.\n"
        )

    summary = (
        f"SNR da chon: {snr_db} dB | Checkpoint hoc tu 300 cau "
        f"({len(entity_vocab)} thuc the, {len(relation_vocab)} quan he)\n"
        f"So bo ba goc: {len(clean_triples)} | So bo ba AI doan duoc: {len(reconstructed_triples)}\n"
        f"{caveat}\n"
        "----\n"
        "Day la ket qua tu AI DA HUAN LUYEN (chi 300 cau, 5 epoch - con rat\n"
        "so khai). Ket qua co the van sai nhieu, nhung day la lan dau tien\n"
        "he thong thuc su 'hoc' de khoi phuc, khac voi doan ngau nhien ben tren."
    )
    return fig_original, fig_recon, summary


with gr.Blocks(title="Demo Semantic Communication") as demo:
    gr.Markdown(
        "## Demo Semantic Communication: Encoder + Kenh AWGN + AI khoi phuc\n"
        "Nhap bo ba (dinh dang `chu_the | quan_he | khach_the`, moi dong 1 bo ba), chon SNR.\n\n"
        + ("**Da nap AI huan luyen tu 300 cau WebNLG.**" if checkpoint_loaded
           else "**Chua nap duoc checkpoint - upload `checkpoint_week5.pt` vao Colab de bat AI khoi phuc.**")
    )
    inp = gr.Textbox(
        lines=5, label="Nhap bo ba",
        placeholder="Alan_Bean | was a crew member of | Apollo_12\nApollo_12 | operator | NASA\nAlan_Bean | was selected by | NASA",
    )
    snr_slider = gr.Slider(minimum=-6, maximum=18, step=3, value=10, label="SNR (dB) - muc nhieu kenh truyen")
    use_ai_checkbox = gr.Checkbox(label="Su dung AI da huan luyen de khoi phuc", value=False)
    btn = gr.Button("Chay thu", variant="primary")

    with gr.Row():
        out_plot_original = gr.Plot(label="Do thi goc")
        out_plot_recon = gr.Plot(label="Ket qua sau kenh nhieu")
    out_text = gr.Textbox(label="Tom tat", lines=10)

    btn.click(fn=run_demo, inputs=[inp, snr_slider, use_ai_checkbox], outputs=[out_plot_original, out_plot_recon, out_text])

    gr.Examples(
        examples=[
            ["Alan_Bean | was a crew member of | Apollo_12\nApollo_12 | operator | NASA\nAlan_Bean | was selected by | NASA", 10, False],
            ["Alan_Bean | was a crew member of | Apollo_12\nApollo_12 | operator | NASA\nAlan_Bean | was selected by | NASA", 10, True],
        ],
        inputs=[inp, snr_slider, use_ai_checkbox],
    )

if __name__ == "__main__":
    demo.launch(share=True)
