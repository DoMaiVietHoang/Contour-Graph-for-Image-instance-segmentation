from GCN import ContourMerge
from dataset import TreeCrownGraphDataset
from torch_geometric.data import DataLoader
import torch
Adjacebt_dir = './data/Adjacent'
LPIPS_dir = './data/Lpips'
Shape_dir = './data/Shape_features'
dataset = TreeCrownGraphDataset(Adjacebt_dir, LPIPS_dir, Shape_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
model = ContourMerge(
    in_channels = 5,
    hidden_channels = 64,
    out_channels = 32,
    num_layers = 3
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
class_weight = torch.tensor([20.0, 1.0]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        positive_y = torch.ones(batch.edge_index.shape[1], dtype=torch.long).to(device)
        
        # Negative sampling: chọn ngẫu nhiên các cặp không có trong edge_index
        num_nodes = batch.x.shape[0]
        all_pairs = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes))
        edge_set = set(tuple(e) for e in batch.edge_index.t().cpu().numpy())
        neg_pairs = [p for p in all_pairs if tuple(p.numpy()) not in edge_set]
        neg_pairs = torch.stack(neg_pairs)[:batch.edge_index.shape[1]]  # Lấy số lượng bằng positive
        neg_edge_index = neg_pairs.t().to(device)
        negative_y = torch.zeros(neg_pairs.shape[0], dtype=torch.long).to(device)
        # Kết hợp positive và negative
        full_edge_index = torch.cat([batch.edge_index, neg_edge_index], dim=1)
        Y = torch.cat([positive_y, negative_y])
        optimizer.zero_grad()
        logits = model(batch.x, full_edge_index)
        loss = criterion(logits, Y)
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    avg_loss = total_loss / len(dataset)
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
# Lưu mô hình
torch.save(model.state_dict(), 'contour_merger.pth')