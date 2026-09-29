"""
Giao dien demo - tich hop Encoder (tuan 4) + Kenh AWGN + Decoder (tuan 5).

Luong chay: nhap bo ba -> graph_builder -> semantic_encoder -> channel (AWGN)
-> decoder -> hien do thi goc va do thi "khoi phuc duoc".

LUU Y QUAN TRONG: encoder VA decoder o day CHUA duoc huan luyen (huan luyen
la buoc rieng, se lam sau khi 2 khoi nay chay dung). Vi vay do thi khoi phuc
gan nhu chac chan SAI - dung de kiem tra pipeline noi dung nhau khong bi loi
shape/kieu du lieu, KHONG phai ket qua khoa hoc.

Chay: python gradio_app.py
"""

import torch
import networkx as nx
import matplotlib.pyplot as plt
import gradio as gr

from graph_builder import WebNLGGraphBuilder
from semantic_encoder import SemanticEncoder
from channel import awgn_channel
from decoder import SemanticDecoder

print("Dang nap model MiniLM va khoi tao Semantic Encoder...")
builder = WebNLGGraphBuilder(device="cpu")
encoder = SemanticEncoder(in_dim=builder.embed_dim)
encoder.eval()
print("San sang.")


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


def run_demo(triples_text: str, snr_db: float):
    triples = parse_triples(triples_text)
    if not triples:
        empty = plt.figure()
        return empty, empty, "Chua nhap duoc bo ba nao hop le. Dinh dang: chu_the | quan_he | khach_the (moi dong 1 bo ba)."

    # ---- Buoc 1: Encoder (tuan 4) ----
    graph = builder.triples_to_graph(triples)
    entities = graph.entities
    relations_vocab = sorted(set(builder._clean(p) for _, p, _ in triples))
    clean_triples = [(builder._clean(s), builder._clean(p), builder._clean(o)) for s, p, o in triples]

    with torch.no_grad():
        z = encoder(graph)  # vector ngu nghia goc [N, 128]

        # ---- Buoc 2: Kenh AWGN (tuan 5) ----
        z_noisy = awgn_channel(z, snr_db)

        # ---- Buoc 3: Decoder (tuan 5, kien truc, CHUA huan luyen) ----
        num_entities = len(entities)
        num_relations = len(relations_vocab)
        decoder = SemanticDecoder(
            embed_dim=z.shape[1], num_entities=num_entities, num_relations=num_relations
        )
        decoder.eval()
        node_logits, relation_logits = decoder(z_noisy)

        pred_entity_idx = node_logits.argmax(dim=-1).tolist()
        pred_entities = [entities[i] for i in pred_entity_idx]

        relation_pred = relation_logits.argmax(dim=-1)  # [N, N]
        none_idx = num_relations  # nhan "none" nam o vi tri cuoi
        reconstructed_triples = []
        for i in range(num_entities):
            for j in range(num_entities):
                if i == j:
                    continue
                r_idx = relation_pred[i, j].item()
                if r_idx != none_idx:
                    reconstructed_triples.append((pred_entities[i], relations_vocab[r_idx], pred_entities[j]))

    noise_mse = (z - z_noisy).pow(2).mean().item()

    fig_original = draw_graph(entities, clean_triples, f"Do thi GOC ({len(entities)} node, {len(clean_triples)} canh)")
    fig_recon = draw_graph(
        list(dict.fromkeys(pred_entities)), reconstructed_triples,
        f"Do thi KHOI PHUC sau kenh nhieu (SNR = {snr_db} dB)",
    )

    summary = (
        f"SNR da chon: {snr_db} dB | Sai lech trung binh do nhieu (MSE): {noise_mse:.4f}\n\n"
        f"So bo ba goc: {len(clean_triples)}\n"
        f"So bo ba decoder doan duoc: {len(reconstructed_triples)}\n\n"
        "----\n"
        "Luu y: Decoder CHUA duoc huan luyen, nen do thi khoi phuc o day "
        "gan nhu chac chan sai/ngau nhien. Buoc nay chi xac nhan Encoder -> "
        "Kenh AWGN -> Decoder noi voi nhau dung shape, chua phai ket qua "
        "thuc nghiem. Ket qua that se co sau khi huan luyen mo hinh."
    )
    return fig_original, fig_recon, summary


with gr.Blocks(title="Demo Semantic Communication - Tuan 4-5") as demo:
    gr.Markdown(
        "## Demo Semantic Communication: Encoder + Kenh AWGN + Decoder\n"
        "Nhap bo ba (dinh dang `chu_the | quan_he | khach_the`, moi dong 1 bo ba), "
        "chon muc SNR, xem do thi goc va do thi ma decoder khoi phuc duoc.\n\n"
        "**Encoder va Decoder deu CHUA duoc huan luyen** - day la buoc kiem tra "
        "kien truc va cach noi cac khoi, chua phai ket qua thuc nghiem cuoi cung."
    )
    inp = gr.Textbox(
        lines=5, label="Nhap bo ba",
        placeholder="Alan_Bean | was a crew member of | Apollo_12\nApollo_12 | operator | NASA\nAlan_Bean | was selected by | NASA",
    )
    snr_slider = gr.Slider(minimum=-6, maximum=18, step=3, value=10, label="SNR (dB) - theo dai da chon o Muc 3.3")
    btn = gr.Button("Chay Encoder -> Channel -> Decoder", variant="primary")

    with gr.Row():
        out_plot_original = gr.Plot(label="Do thi goc")
        out_plot_recon = gr.Plot(label="Do thi khoi phuc (chua huan luyen)")
    out_text = gr.Textbox(label="Tom tat", lines=10)

    btn.click(fn=run_demo, inputs=[inp, snr_slider], outputs=[out_plot_original, out_plot_recon, out_text])

    gr.Examples(
        examples=[
            ["Alan_Bean | was a crew member of | Apollo_12\nApollo_12 | operator | NASA\nAlan_Bean | was selected by | NASA", 10],
            ["Amsterdam_Airport_Schiphol | location | Netherlands\nNetherlands | leader | Mark_Rutte", -6],
        ],
        inputs=[inp, snr_slider],
    )

if __name__ == "__main__":
    demo.launch(share=True)
