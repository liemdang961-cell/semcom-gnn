"""
Giao dien demo - tich hop Encoder + Kenh AWGN + Decoder.

Thay vi 1 o tick, giao dien co 1 O CHON (dropdown) liet ke moi checkpoint
da huan luyen nam trong thu muc checkpoints/ (vd checkpoint_week5.pt,
checkpoint_week6.pt...). Cac file nay nam SAN trong repo GitHub, nen chi
can "git clone" la co du, khong can Google Drive hay upload tay.

- Chon "Khong dung AI (ngau nhien)": dung 1 decoder khoi tao ngau nhien
  (tu dien nho, lay tu chinh cau nhap vao) - minh hoa AI khi CHUA hoc.
- Chon 1 checkpoint that: dung dung Encoder + Decoder da hoc tu file do.

Chay: python gradio_app.py
"""

import glob
import torch
import networkx as nx
import matplotlib.pyplot as plt
import gradio as gr

from graph_builder import WebNLGGraphBuilder
from semantic_encoder import SemanticEncoder
from channel import awgn_channel
from decoder import SemanticDecoder

CHECKPOINTS_DIR = "checkpoints"
NO_AI_OPTION = "Khong dung AI (ngau nhien, chua huan luyen)"

print("Dang nap model MiniLM va khoi tao Semantic Encoder...")
builder = WebNLGGraphBuilder(device="cpu")
print("San sang.")

# ----- Quet thu muc checkpoints/, nap moi file tim thay -----
loaded_models = {}  # ten hien thi -> dict{encoder, decoder, entity_vocab, relation_vocab, idx_to_entity, idx_to_relation}

for path in sorted(glob.glob(f"{CHECKPOINTS_DIR}/*.pt")):
    try:
        ckpt = torch.load(path, map_location="cpu")
        entity_vocab = ckpt["entity_vocab"]
        relation_vocab = ckpt["relation_vocab"]

        enc = SemanticEncoder(in_dim=builder.embed_dim)
        enc.load_state_dict(ckpt["encoder"])
        enc.eval()

        dec = SemanticDecoder(embed_dim=128, num_entities=len(entity_vocab), num_relations=len(relation_vocab))
        dec.load_state_dict(ckpt["decoder"])
        dec.eval()

        label = ckpt.get("label", path)
        num_samples = ckpt.get("num_samples_trained", "?")
        display_name = f"{label} ({num_samples} cau, {len(entity_vocab)} thuc the)"

        loaded_models[display_name] = {
            "encoder": enc, "decoder": dec,
            "entity_vocab": entity_vocab, "relation_vocab": relation_vocab,
            "idx_to_entity": {i: n for n, i in entity_vocab.items()},
            "idx_to_relation": {i: n for n, i in relation_vocab.items()},
        }
        print(f"Da nap checkpoint: {path} -> '{display_name}'")
    except Exception as e:
        print(f"Bo qua {path} (loi khi nap: {e})")

if not loaded_models:
    print(f"Chua tim thay checkpoint nao trong '{CHECKPOINTS_DIR}/'. Chi dung duoc tuy chon '{NO_AI_OPTION}'.")

DROPDOWN_CHOICES = [NO_AI_OPTION] + list(loaded_models.keys())


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


def decode_and_draw(z_noisy, entities, decoder, idx_to_entity, relation_vocab_size, idx_to_relation, title):
    n = len(entities)
    with torch.no_grad():
        node_logits, relation_logits = decoder(z_noisy)

    pred_entities = [idx_to_entity.get(i, "?") for i in node_logits.argmax(dim=-1).tolist()]
    relation_pred = relation_logits.argmax(dim=-1)
    none_idx = relation_vocab_size
    reconstructed_triples = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_idx = relation_pred[i, j].item()
            if r_idx != none_idx:
                reconstructed_triples.append((pred_entities[i], idx_to_relation.get(r_idx, "?"), pred_entities[j]))

    fig = draw_graph(list(dict.fromkeys(pred_entities)), reconstructed_triples, title)
    return fig, reconstructed_triples


def run_demo(triples_text: str, snr_db: float, choice: str):
    triples = parse_triples(triples_text)
    if not triples:
        empty = plt.figure()
        return empty, empty, "Chua nhap duoc bo ba nao hop le. Dinh dang: chu_the | quan_he | khach_the (moi dong 1 bo ba)."

    clean_triples = [(builder._clean(s), builder._clean(p), builder._clean(o)) for s, p, o in triples]
    graph = builder.triples_to_graph(triples)
    entities = graph.entities
    n = len(entities)

    use_trained = choice in loaded_models
    active_encoder = loaded_models[choice]["encoder"] if use_trained else SemanticEncoder(in_dim=builder.embed_dim).eval()

    with torch.no_grad():
        z = active_encoder(graph)
        z_noisy = awgn_channel(z, snr_db)

    fig_original = draw_graph(entities, clean_triples, f"Do thi GOC ({n} node, {len(clean_triples)} canh)")

    if not use_trained:
        local_relations = sorted(set(p for _, p, _ in clean_triples))
        random_decoder = SemanticDecoder(embed_dim=z_noisy.shape[1], num_entities=n, num_relations=len(local_relations))
        random_decoder.eval()
        idx_to_entity_local = {i: name for i, name in enumerate(entities)}
        idx_to_relation_local = {i: r for i, r in enumerate(local_relations)}
        fig_recon, reconstructed = decode_and_draw(
            z_noisy, entities, random_decoder, idx_to_entity_local, len(local_relations), idx_to_relation_local,
            f"Do thi KHOI PHUC - AI CHUA huan luyen (SNR = {snr_db} dB)",
        )
        summary = (
            f"SNR da chon: {snr_db} dB | Che do: {NO_AI_OPTION}\n"
            f"So bo ba goc: {len(clean_triples)} | So bo ba doan duoc: {len(reconstructed)}\n\n"
            "----\n"
            "AI o day CHUA duoc huan luyen (trong so ngau nhien), nen doan lung\n"
            "tung, thuong gop nhieu thuc the lai thanh 1-2 thuc the sai."
        )
        return fig_original, fig_recon, summary

    model = loaded_models[choice]
    fig_recon, reconstructed = decode_and_draw(
        z_noisy, entities, model["decoder"], model["idx_to_entity"], len(model["relation_vocab"]),
        model["idx_to_relation"], f"Do thi KHOI PHUC boi '{choice}' (SNR = {snr_db} dB)",
    )

    unknown_entities = [e for e in entities if e not in model["entity_vocab"]]
    caveat = ""
    if unknown_entities:
        caveat = (
            f"\nLUU Y: {len(unknown_entities)} thuc the ban nhap ({', '.join(unknown_entities[:3])}...) "
            f"KHONG nam trong tu dien AI da hoc, nen AI chac chan doan sai cho cac thuc the nay.\n"
        )

    summary = (
        f"SNR da chon: {snr_db} dB | Mo hinh: {choice}\n"
        f"So bo ba goc: {len(clean_triples)} | So bo ba AI doan duoc: {len(reconstructed)}\n"
        f"{caveat}\n"
        "----\n"
        "Ket qua tu AI DA HUAN LUYEN. Neu con moi, hay so sanh voi cac\n"
        "checkpoint khac (neu co) hoac voi 'Khong dung AI' de thay su khac biet."
    )
    return fig_original, fig_recon, summary


with gr.Blocks(title="Demo Semantic Communication") as demo:
    gr.Markdown(
        "## Demo Semantic Communication: Encoder + Kenh AWGN + AI khoi phuc\n"
        "Nhap bo ba (dinh dang `chu_the | quan_he | khach_the`, moi dong 1 bo ba), chon SNR va chon mo hinh.\n\n"
        f"**Da tim thay {len(loaded_models)} checkpoint da huan luyen** trong repo."
        if loaded_models else
        "**Chua co checkpoint nao trong thu muc `checkpoints/` cua repo.** Chi dung duoc tuy chon 'Khong dung AI'."
    )
    inp = gr.Textbox(
        lines=5, label="Nhap bo ba",
        placeholder="Alan_Bean | was a crew member of | Apollo_12\nApollo_12 | operator | NASA\nAlan_Bean | was selected by | NASA",
    )
    snr_slider = gr.Slider(minimum=-6, maximum=18, step=3, value=10, label="SNR (dB) - muc nhieu kenh truyen")
    model_choice = gr.Dropdown(choices=DROPDOWN_CHOICES, value=DROPDOWN_CHOICES[0], label="Chon mo hinh")
    btn = gr.Button("Chay thu", variant="primary")

    with gr.Row():
        out_plot_original = gr.Plot(label="Do thi goc")
        out_plot_recon = gr.Plot(label="Ket qua sau kenh nhieu")
    out_text = gr.Textbox(label="Tom tat", lines=10)

    btn.click(fn=run_demo, inputs=[inp, snr_slider, model_choice], outputs=[out_plot_original, out_plot_recon, out_text])

    gr.Examples(
        examples=[
            ["Alan_Bean | was a crew member of | Apollo_12\nApollo_12 | operator | NASA\nAlan_Bean | was selected by | NASA", 10, DROPDOWN_CHOICES[0]],
        ],
        inputs=[inp, snr_slider, model_choice],
    )

if __name__ == "__main__":
    demo.launch(share=True)
