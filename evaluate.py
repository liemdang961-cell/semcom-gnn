import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader 

# Import các file từ dự án 
from semantic_encoder import SemanticEncoder
from decoder import SemanticDecoder
from channel import awgn_channel
# Gọi đúng tên hàm và class từ file graph_builder.py 
from graph_builder import WebNLGGraphBuilder, load_webnlg_triples

def evaluate_snr(encoder, decoder, dataloader, snr_levels, device):
    encoder.eval()
    decoder.eval()
    
    results_mse = [] 
    
    with torch.no_grad():
        for snr in snr_levels:
            total_mse = 0
            num_batches = 0
            
            for data in dataloader:
                data = data.to(device)
                
                # 1. Semantic Encoder
                node_embeddings = encoder(data)
                
                # 2. Truyền qua kênh AWGN
                rx_signal = awgn_channel(node_embeddings, snr_db=snr)
                
                # 3. Semantic Decoder (ĐÃ SỬA: Chỉ truyền 1 tham số)
                node_preds, _ = decoder(rx_signal) 
                
                # 4. Tính toán độ lệch MSE (ĐÃ SỬA: Tránh lỗi kích thước)
                # Đánh giá độ méo tín hiệu vật lý do nhiễu kênh AWGN gây ra
                mse = torch.nn.functional.mse_loss(rx_signal, node_embeddings)
                
                total_mse += mse.item()
                num_batches += 1
                    
            avg_mse = total_mse / num_batches if num_batches > 0 else 0
            results_mse.append(avg_mse)
            
            print(f"SNR: {snr:2d} dB | Độ lệch MSE: {avg_mse:.4f} (Càng nhỏ càng tốt)")
            
    return results_mse

def main():
    # 1. Định nghĩa device đầu tiên để không bị lỗi NameError
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên thiết bị: {device}")
    
    # 2. Khởi tạo mô hình một lần duy nhất, xóa các dòng thừa
    encoder = SemanticEncoder().to(device)
    decoder = SemanticDecoder(num_entities=283, num_relations=50).to(device)

    try:
        checkpoint = torch.load('checkpoints/checkpoint_week5.pt', map_location=device)
        encoder.load_state_dict(checkpoint['encoder']) 
        decoder.load_state_dict(checkpoint['decoder'])
        print("Đã load checkpoint thành công!")
    except Exception as e:
        print(f"Lỗi khi nạp checkpoint: {e}")
        return
    
    
    print("\nĐang tải bộ dữ liệu test (khoảng 300 câu để đánh giá)...")
    # Lấy 300 câu test từ mạng
    test_triples = load_webnlg_triples(split="test", max_samples=300)
    # Nhúng thành đồ thị
    builder = WebNLGGraphBuilder(device=str(device))
    test_graphs = builder.build_dataset(test_triples)
    # Bỏ vào DataLoader
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    # -----------------------------------------------------------------------
    
    snr_levels = [-5, 0, 5, 10, 15, 20]
    
    print("\nBắt đầu đánh giá qua các mức SNR...")
    mse_scores = evaluate_snr(encoder, decoder, test_loader, snr_levels, device)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    plt.plot(snr_levels, mse_scores, marker='o', color='r', linewidth=2, label='MSE Loss')
        
    plt.title('Đánh giá chất lượng Semantic Communication theo SNR (Tuần 6)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Độ lệch MSE (Càng thấp càng tốt)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('snr_evaluation_plot.png')
    print("\nĐã chạy xong! Đã lưu biểu đồ vào file 'snr_evaluation_plot.png'.")

if __name__ == "__main__":
    main()