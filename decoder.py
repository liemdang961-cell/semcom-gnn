"""
Tuan 5 (lam truoc) - Semantic Decoder (Muc 2.3.3 khoa luan).

Gom 2 khoi, dung cong thuc tuong tu (4)-(7) trong bai Hello va cong su [5]:
  - Node Classifier : MLP co ket noi tat (skip connection), doan lai
                       thuc the ban dau tu vector nhan duoc.
  - Relation Classifier : dung Self-Attention (Transformer Encoder) de
                       tinh chinh embedding truoc khi phan loai quan he
                       giua tung cap node (bao gom ca nhan "none" - khong
                       co quan he truc tiep).

LUU Y: decoder o day CHUA duoc huan luyen. So lop entity/relation (vocab)
trong ban demo chi lay tu chinh cau nhap vao (closed-world nho), khac voi
huan luyen that se dung vocab xay tu toan bo 20.000 mau WebNLG train.
"""

import torch
import torch.nn as nn


class NodeClassifier(nn.Module):
    """MLP voi ket noi tat, phan loai node ve lai 1 trong so cac thuc the
    da biet (Muc 2.3.3 - Bo phan loai nut)."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(embed_dim, hidden_dim)  # chieu input de cong truc tiep (skip connection)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.act(self.fc1(y)))
        h = self.act(self.fc2(h) + self.skip(y))  # ket noi tat: cong thang tu input ban dau
        return self.out(h)  # [N, num_classes]


class RelationClassifier(nn.Module):
    """Self-Attention (Transformer Encoder) tinh chinh embedding cua toan
    bo node, sau do phan loai quan he cho tung cap (i, j) (Muc 2.3.3 -
    Bo phan loai quan he). Co them 1 nhan "none" cho cap khong co quan he."""

    def __init__(self, embed_dim: int, num_relations: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.self_attn = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Linear(embed_dim * 2, num_relations + 1)  # +1 la nhan "none"

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [N, embed_dim]
        n = y.shape[0]
        refined = self.self_attn(y.unsqueeze(0)).squeeze(0)          # [N, embed_dim]
        yi = refined.unsqueeze(1).expand(n, n, -1)                   # [N, N, d]
        yj = refined.unsqueeze(0).expand(n, n, -1)                   # [N, N, d]
        pair = torch.cat([yi, yj], dim=-1)                           # [N, N, 2d]
        return self.classifier(pair)                                 # [N, N, num_relations+1]


class SemanticDecoder(nn.Module):
    """Ghep 2 khoi tren thanh 1 decoder hoan chinh."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128,
                 num_entities: int = 100, num_relations: int = 50):
        super().__init__()
        self.node_classifier = NodeClassifier(embed_dim, hidden_dim, num_entities)
        self.relation_classifier = RelationClassifier(embed_dim, num_relations)

    def forward(self, y: torch.Tensor):
        node_logits = self.node_classifier(y)            # [N, num_entities]
        relation_logits = self.relation_classifier(y)     # [N, N, num_relations+1]
        return node_logits, relation_logits


if __name__ == "__main__":
    y = torch.randn(4, 128)  # gia lap 4 node, 128 chieu (dau ra tu channel.py)
    decoder = SemanticDecoder(embed_dim=128, num_entities=4, num_relations=3)
    node_logits, relation_logits = decoder(y)
    print("Node logits shape:", node_logits.shape)          # [4, 4]
    print("Relation logits shape:", relation_logits.shape)  # [4, 4, 4]
