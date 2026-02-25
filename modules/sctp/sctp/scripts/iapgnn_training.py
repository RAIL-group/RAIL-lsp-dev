import pickle
import gzip
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sctp.learning.iap_gnn import BipartiteEdgeRegressor
from torch.utils.data import Dataset
from torch_geometric.data import Data

class GzipGNNDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = glob.glob(os.path.join(folder_path, '*.pgz'))
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with gzip.open(self.file_paths[idx], 'rb') as f:
            # This returns your 'GraphData' object
            raw_data = pickle.load(f)
        data = Data(
            x=torch.from_numpy(raw_data.x).float(),
            edge_index=torch.from_numpy(raw_data.edge_index).long(),
            edge_attr=torch.from_numpy(raw_data.edge_attr).float(),
            y=torch.from_numpy(raw_data.y).float(),
        )
        if hasattr(raw_data, 'graph_metadata'):
            data.metadata = raw_data.graph_metadata
        return data
        


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 2. Forward Pass
        # Pass the batch data and the specific node attributes
        output = model(
            x= batch.x, 
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
        )
        
        # 3. Compute Loss
        # Ensure output and target have the same shape [Batch, 1]
        # loss = criterion(output.view(-1), batch.y.view(-1))
        loss = criterion(output, batch.y)  # MSE Loss between predicted and true edge values
        
        # 4. Backward Pass
        loss.backward()
        optimizer.step()
        
        # total_loss += loss.item() * batch.num_graphs
        total_loss += loss.item()
        
    return total_loss / len(loader)


if __name__ == "__main__":
    # Load your data
    dataset = GzipGNNDataset('data/sctp/graph_data/pickles/')
    # with open('data/sctp/graph_data/pickles/data_1000_0.pgz', 'rb') as f:
    #     dataset = pickle.load(f)

    # Use a DataLoader for batching
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 1. Setup Device & Model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # Force CPU for debugging
    NODE_IN = 2
    EDGE_IN = 2
    HIDDEN = 32
    model = BipartiteEdgeRegressor(node_in_dim=NODE_IN, edge_in_dim=EDGE_IN, hidden_dim=HIDDEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 5. Execute Training
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, MSE Loss: {loss:.4f}')