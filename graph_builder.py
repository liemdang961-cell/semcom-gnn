"""
Tuan 4 - Buoc 1: Chuyen doi bo ba (subject, predicate, object) thanh do thi
PyTorch Geometric, dung cho Semantic Encoder (Muc 2.2, 2.3.1 khoa luan).

Moi cau WebNLG duoc bieu dien thanh MOT do thi rieng:
  - node  = thuc the (subject / object), nhung bang all-MiniLM-L6-v2 (384 chieu)
  - edge  = quan he (predicate), cung nhung bang MiniLM (384 chieu)

Dung all-MiniLM-L6-v2 thay vi BERT-base vi hai bai [4] (Hello et al.) va
[5] (Fan et al.) trong Chuong 1 deu dung mo hinh nay tren chinh WebNLG,
nen ket qua cua nhom se de doi chieu voi hai bai do.
"""

from typing import List, Tuple
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

Triple = Tuple[str, str, str]  # (subject, predicate, object)


class WebNLGGraphBuilder:
    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self.embedder = SentenceTransformer(embed_model_name, device=device)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()  # 384

    @staticmethod
    def _clean(text: str) -> str:
        """Chuan hoa nhe: bo gach duoi, ngoac kep thuong thay trong WebNLG."""
        return text.replace("_", " ").strip().strip('"')

    def triples_to_graph(self, triples: List[Triple]) -> Data:
        """Chuyen 1 danh sach bo ba thanh 1 Data object.

        - Thuc the trung nhau (vd cung la subject o nhieu bo ba) chi tao 1 node.
        - Neu do thi qua nho (1 node, khong co canh) van tra ve binh thuong;
          Semantic Encoder o buoc sau se tu xu ly truong hop nay.
        """
        entities: List[str] = []
        entity_to_idx = {}

        def get_idx(name: str) -> int:
            name = self._clean(name)
            if name not in entity_to_idx:
                entity_to_idx[name] = len(entities)
                entities.append(name)
            return entity_to_idx[name]

        src_list, dst_list, relation_texts = [], [], []
        for s, p, o in triples:
            si, oi = get_idx(s), get_idx(o)
            src_list.append(si)
            dst_list.append(oi)
            relation_texts.append(self._clean(p))

        if len(entities) == 0:
            raise ValueError("Danh sach triples rong, khong the tao do thi.")

        # Nhung tat ca thuc the va quan he trong 1 lan goi cho nhanh
        node_emb = self.embedder.encode(entities, convert_to_tensor=True, device=self.device)
        if relation_texts:
            edge_emb = self.embedder.encode(relation_texts, convert_to_tensor=True, device=self.device)
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            # Do thi chi co 1 node, khong co canh nao (hiem gap voi WebNLG)
            edge_emb = torch.empty((0, self.embed_dim))
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(
            x=node_emb.float(),
            edge_index=edge_index,
            edge_attr=edge_emb.float(),
        )
        data.entities = entities          # giu lai ten that de doi chieu/debug
        data.relations = relation_texts
        return data

    def build_dataset(self, list_of_triples: List[List[Triple]]) -> List[Data]:
        """Ap dung triples_to_graph cho toan bo tap du lieu (vd 20.000 mau train)."""
        graphs = []
        for triples in list_of_triples:
            try:
                graphs.append(self.triples_to_graph(triples))
            except ValueError:
                continue  # bo qua mau rong, khong lam gay ca pipeline
        return graphs


def load_webnlg_triples(split: str = "train", max_samples: int | None = None) -> List[List[Triple]]:
    """Tai bo ba tu bo du lieu WebNLG qua thu vien 'datasets' (can mang internet).

    Neu nhom da co san file WebNLG rieng (json/csv da tien xu ly o tuan 3),
    thay ham nay bang ham doc file cua nhom - miem sao tra ve dung dinh dang
    List[List[Triple]] la dung duoc voi WebNLGGraphBuilder.
    """
    from datasets import load_dataset

    ds = load_dataset("web_nlg", "release_v3.0_en", split=split, trust_remote_code=True)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    all_triples: List[List[Triple]] = []
    for row in ds:
        triples = []
        for t in row["modified_triple_sets"]["mtriple_set"][0]:
            # dinh dang goc: "subject | predicate | object"
            parts = [p.strip() for p in t.split("|")]
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
        if triples:
            all_triples.append(triples)
    return all_triples


if __name__ == "__main__":
    # Chay thu nhanh voi vai cau tay, khong can mang / khong can tai WebNLG that
    sample_triples = [
        ("Alan_Bean", "was a crew member of", "Apollo_12"),
        ("Apollo_12", "operator", "NASA"),
        ("Alan_Bean", "was selected by", "NASA"),
    ]
    builder = WebNLGGraphBuilder()
    g = builder.triples_to_graph(sample_triples)
    print("So node:", g.x.shape[0], "| So canh:", g.edge_index.shape[1])
    print("Thuc the:", g.entities)
    print("Quan he :", g.relations)
