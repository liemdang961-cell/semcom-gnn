"""
Tuan 4 - Giao dien demo cho Semantic Encoder (chua huan luyen).

Muc dich: nhap cac bo ba (dinh dang WebNLG) -> xem do thi tri thuc duoc
tao ra -> xem vector ngu nghia 128 chieu do GAT encoder sinh ra.

LUU Y QUAN TRONG: encoder trong file nay CHUA duoc huan luyen (do la
viec cua cac tuan sau, khi co Channel + Decoder + vong lap train). Vi
vay cac con so trong vector chi de kiem tra pipeline chay dung, KHONG
phai ket qua khoa hoc that. Ket qua that se co sau khi huan luyen.

Chay: python gradio_app.py
"""

import torch
import networkx as nx
import matplotlib.pyplot as plt
import gradio as gr

from graph_builder import WebNLGGraphBuilder
from semantic_encoder import SemanticEncoder

# ----- Nap model 1 lan duy nhat khi khoi dong app (khong nap lai moi lan bam nut) -----
print("Dang nap model MiniLM va khoi tao Semantic Encoder...")
builder = WebNLGGraphBuilder(device="cpu")
encoder = SemanticEncoder(in_dim=builder.embed_dim)
encoder.eval()  # chi demo forward pass, chua huan luyen
print("San sang.")


def parse_triples(text: str):
    """Moi dong 1 bo ba, cach nhau boi dau '|'. Vd:
    Alan_Bean | was a crew member of | Apollo_12
    """
    triples = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def run_demo(triples_text: str):
    triples = parse_triples(triples_text)
    if not triples:
        return None, "Chua nhap duoc bo ba nao hop le. Dinh dang: chu_the | quan_he | khach_the (moi dong 1 bo ba)."

    graph = builder.triples_to_graph(triples)
    with torch.no_grad():
        z = encoder(graph)

    # ---- Ve do thi tri thuc bang networkx ----
    G = nx.DiGraph()
    for name in graph.entities:
        G.add_node(name)
    for s, p, o in triples:
        G.add_edge(builder._clean(s), builder._clean(o), label=builder._clean(p))

    fig, ax = plt.subplots(figsize=(6, 4.2))
    pos = nx.spring_layout(G, seed=42, k=1.2)
    nx.draw(
        G, pos, ax=ax, with_labels=True, node_color="#8ecae6",
        node_size=1900, font_size=9, arrows=True, arrowsize=18,
    )
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)
    ax.set_title(f"Do thi tri thuc: {len(graph.entities)} node, {len(triples)} canh")
    fig.tight_layout()

    # ---- Tom tat ket qua encoder ----
    first_entity = graph.entities[0]
    summary = (
        f"So node trong do thi: {z.shape[0]}\n"
        f"So chieu vector ngu nghia moi node: {z.shape[1]}\n\n"
        f"Vi du vector cua node '{first_entity}' (5 gia tri dau):\n"
        f"{[round(v, 4) for v in z[0][:5].tolist()]}\n\n"
        "----\n"
        "Luu y: Encoder o day CHUA duoc huan luyen (thuoc pham vi tuan 4). "
        "Cac con so tren chi xac nhan pipeline chay dung dinh dang va shape, "
        "chua phai ket qua khoa hoc. Sau khi them Channel + Decoder va huan luyen "
        "o cac tuan sau, vector nay se thuc su mang y nghia ngu nghia cua cau."
    )
    return fig, summary


with gr.Blocks(title="Demo Semantic Encoder - Tuan 4") as demo:
    gr.Markdown(
        "## Demo Semantic Encoder dung GAT (Tuan 4)\n"
        "Nhap cac bo ba theo dinh dang WebNLG (moi dong: `chu_the | quan_he | khach_the`), "
        "he thong se dung thanh do thi tri thuc va chay qua Semantic Encoder.\n\n"
        "**Encoder chua duoc huan luyen** - day chi la buoc kiem tra kien truc, "
        "chua phai ket qua thuc nghiem cuoi cung."
    )
    with gr.Row():
        inp = gr.Textbox(
            lines=6,
            label="Nhap bo ba",
            placeholder="Alan_Bean | was a crew member of | Apollo_12\nApollo_12 | operator | NASA\nAlan_Bean | was selected by | NASA",
        )
    btn = gr.Button("Chay Semantic Encoder", variant="primary")
    with gr.Row():
        out_plot = gr.Plot(label="Do thi tri thuc duoc tao ra")
        out_text = gr.Textbox(label="Ket qua Semantic Encoder", lines=12)

    btn.click(fn=run_demo, inputs=inp, outputs=[out_plot, out_text])

    gr.Examples(
        examples=[
            "Alan_Bean | was a crew member of | Apollo_12\nApollo_12 | operator | NASA\nAlan_Bean | was selected by | NASA",
            "Amsterdam_Airport_Schiphol | location | Netherlands\nNetherlands | leader | Mark_Rutte",
        ],
        inputs=inp,
    )

if __name__ == "__main__":
    # share=True de Colab tao 1 link cong khai tam thoi, mo duoc tren dien thoai/may khac
    demo.launch(share=True)
