"""
Tuan 4 - Buoc 2: Semantic Encoder dung GNN (Muc 2.3.1 khoa luan).

Ly do tu viet EdgeGATLayer thay vi dung thang torch_geometric.nn.GATConv:
GATConv goc CHI dung dac trung node (x) de tinh attention, KHONG nhan
edge_attr. Nhung trong do thi tri thuc, canh (quan he) mang y nghia
quan trong khong kem node, nen encoder phai "dung hop dac trung cua
cac nut va canh" nhu mo ta o 2.3.1. Vi vay lop ben duoi ghep them
edge_attr vao cong thuc tinh he so chu y (attention coefficient),
cung tinh than voi co che cua Yu va cong su [9] va Hello va cong su [4].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class EdgeGATLayer(MessagePassing):
    """1 lop GAT co ket hop edge_attr khi tinh attention va khi tong hop tin.

    Cong thuc (tuong tu GAT goc, co them e_ij):
        a_ij = LeakyReLU( W_a [ W_x*x_i || W_x*x_j || W_e*e_ij ] )
        alpha_ij = softmax_j(a_ij)                     # chuan hoa theo tung node dich i
        x_i' = sigma( sum_j alpha_ij * (W_x*x_j + W_e*e_ij) )
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__(aggr="add", node_dim=0)
        self.heads = heads
        self.out_per_head = out_dim // heads
        assert out_dim % heads == 0, "out_dim phai chia het cho so head"

        self.lin_x = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_e = nn.Linear(edge_dim, out_dim, bias=False)
        # attention: nhan vao [x_i || x_j || e_ij] da chia theo head
        self.attn = nn.Parameter(torch.empty(heads, 3 * self.out_per_head))
        nn.init.xavier_uniform_(self.attn)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin_x(x).view(-1, self.heads, self.out_per_head)          # [N, H, d]
        e = self.lin_e(edge_attr).view(-1, self.heads, self.out_per_head)  # [E, H, d]
        out = self.propagate(edge_index, x=x, edge_attr=e)
        return out.view(-1, self.heads * self.out_per_head)                # [N, out_dim]

    def message(self, x_i, x_j, edge_attr, index, size_i):
        # x_i, x_j: [E, H, d] (dau va cuoi canh), edge_attr: [E, H, d]
        cat = torch.cat([x_i, x_j, edge_attr], dim=-1)          # [E, H, 3d]
        e = self.leaky_relu((cat * self.attn).sum(dim=-1))      # [E, H]
        alpha = softmax(e, index, num_nodes=size_i)             # chuan hoa theo node dich i
        alpha = self.dropout(alpha)
        return (x_j + edge_attr) * alpha.unsqueeze(-1)          # [E, H, d]


class SemanticEncoder(nn.Module):
    """Bo ma hoa ngu nghia: dua dac trung tho (384 chieu tu MiniLM) qua 2 lop
    EdgeGATLayer, roi chieu xuong embed_dim=128 (dung Bang 3.1 khoa luan).
    """

    def __init__(self, in_dim: int = 384, hidden_dim: int = 128, embed_dim: int = 128,
                 heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.node_in = nn.Linear(in_dim, hidden_dim)
        self.edge_in = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EdgeGATLayer(hidden_dim, hidden_dim, hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = F.relu(self.node_in(data.x))
        if data.edge_attr.numel() > 0:
            edge_attr = F.relu(self.edge_in(data.edge_attr))
            edge_index = data.edge_index
        else:
            # do thi 1-node khong co canh: tao 1 self-loop de lop GAT khong bi loi
            edge_index = torch.zeros((2, 1), dtype=torch.long, device=x.device)
            edge_attr = torch.zeros((1, x.shape[-1]), device=x.device)

        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, edge_index, edge_attr)
            x = norm(x + residual)          # residual + LayerNorm cho on dinh khi huan luyen
            x = self.dropout(F.relu(x))

        z = self.out_proj(x)                # [N, embed_dim] - vector ngu nghia cuoi cung
        return z


if __name__ == "__main__":
    # Test nhanh voi tensor gia, khong can du lieu that
    from torch_geometric.data import Data

    x = torch.randn(4, 384)                       # 4 node, 384 chieu (nhu MiniLM)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.randn(3, 384)
    fake_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    encoder = SemanticEncoder()
    z = encoder(fake_graph)
    print("Output shape:", z.shape)  # ky vong: torch.Size([4, 128])
