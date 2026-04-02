import sys, os
import pickle, gzip
import glob
# import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sctp.learning.iap_gnn import BipartiteEdgeRegressor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
import sctp.scripts.data_gen as data_gen
import argparse


sys.modules['__main__'].GraphData = data_gen.GraphData

class GzipGNNDataset(Dataset):
    def __init__(self, file_list):
        # self.file_paths = glob.glob(os.path.join(folder_path, '*.pgz'))
        self.file_paths = file_list  # Expecting a list of file paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with gzip.open(self.file_paths[idx], 'rb') as f:
            # This returns your 'GraphData' object
            raw_data = pickle.load(f)
            data = Data(
                x=torch.from_numpy(raw_data.x).float(),
                # edge_index=torch.from_numpy(raw_data.edge_index).long().t().contiguous(),
                edge_index=torch.from_numpy(raw_data.edge_index).long().contiguous(),
                edge_attr=torch.from_numpy(raw_data.edge_attr).float(),
                y=torch.from_numpy(raw_data.y).float(),
            )
            if hasattr(raw_data, 'graph_metadata'):
                data.metadata = raw_data.graph_metadata
        return data

# def custom_loss_function(preds, targets):

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # 2. Forward Pass
        # Pass the batch data and the specific node attributes
        output, masks = model(
            x= batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
        )
        # loss = model.loss(output, batch.y, masks)
        loss = model.ig_regression_loss(output, batch.y, batch.edge_attr, uncertain_weight=1.0, certain_weight=0.1)
        # 4. Backward Pass
        loss.backward()
        optimizer.step()

        # total_loss += loss.item() * batch.num_graphs
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, masks = model(batch.x, batch.edge_index, batch.edge_attr)
            # loss = model.loss(out, batch.y, masks)
            loss = model.ig_regression_loss(out, batch.y, batch.edge_attr, uncertain_weight=1.0, certain_weight=0.1)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_type', type=str, default='bridges')
    args = parser.parse_args()
    
    
    # --- 1. Data Preparation & Splitting ---
    data_dir = 'data/sctp/graph_data/pickles_'+ args.graph_type+'/'
    print(f"Loading data from {data_dir}...")
    # exit(0)
    all_files = glob.glob(os.path.join(data_dir, '*.pgz'))
    
    # 80/20 Split
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    # Load your data
    train_dataset = GzipGNNDataset(train_files) # Modify Dataset to take a list of files
    test_dataset = GzipGNNDataset(test_files)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 1. Setup Device & Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')  # Force CPU for debugging
    # NODE_IN = 2
    # EDGE_IN = 2
    # HIDDEN_S = 64
    # HIDDEN_M = 128
    learning_rate = 0.0005
    # model = BipartiteEdgeRegressor(node_in_dim=NODE_IN, edge_in_dim=EDGE_IN, hidden_dim=HIDDEN_M).to(device)
    model = BipartiteEdgeRegressor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='data/sctp/training/iap_gnn_trainning_'+args.graph_type+'_logs')

    # 5. Execute Training
    num_epochs = 180
    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        # loss = train_epoch(model, train_loader, optimizer, device)
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluation Step
        test_loss = evaluate(model, test_loader, device)
        
        # Logging to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        
        print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f}")

        # --- 4. Save Model ---
        # Saving every epoch or just the last one
        if epoch % 10 == 0 and epoch > 80:
            torch.save(model.state_dict(), f'data/sctp/training/iap_gnn_{args.graph_type}_epoch_{epoch}.pt')

    writer.close()
        
    