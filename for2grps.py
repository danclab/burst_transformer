

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load and Preprocess Data
variables = [
    "reach_dur",
    "reach_rt",
    "reach_vis_err",
    "reach_vis_abs_err",
    "aim_vis_err",
    "aim_vis_abs_err"
]
# ---- Create PDF ----
pdf_path = "/Users/tanisha/Desktop/implicit.pdf"
pdf = PdfPages(pdf_path)
    
for var in variables:
    print(f"\n===== Processing: {var} =====")
    
    X = np.load("/Users/tanisha/Desktop/CNRS_DATA/aligned_data_v2/implicit/X_time_binned_sensored.npy")
    y = np.load(f"/Users/tanisha/Desktop/CNRS_DATA/aligned_data_v2/implicit/{var}_time_binned_sensored.npy")
    
    # Title Page
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.5, 0.5, var, fontsize=36, ha='center', va='center', weight='bold')
    pdf.savefig()
    plt.close()
    
    print("Loaded X (burst) shape:", X.shape)
    print(f"Loaded y ({var}) shape:", y.shape)
    # Remove invalid entries
    mask = np.all(np.isfinite(X), axis=(1, 2)) & np.isfinite(y)
    X, y = X[mask], y[mask]
    # Normalize X (per feature)
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = ((X_reshaped - X_reshaped.mean(axis=0)) / X_reshaped.std(axis=0)).reshape(X.shape)
    # Normalize y
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    # Dataset
    class TimeBinnedBurstDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]
    # Define the datasets
    train_dataset = TimeBinnedBurstDataset(X_train, y_train)
    test_dataset = TimeBinnedBurstDataset(X_test, y_test)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # Model Definitions
    class CustomTransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dropout=0.1):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.linear1 = nn.Linear(d_model, d_model * 4)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_model * 4, d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.activation = nn.ReLU()
            self.attn_weights = None
        def forward(self, src):
            attn_output, attn_weights = self.self_attn(src, src, src)
            self.attn_weights = attn_weights
            src = src + self.dropout1(attn_output)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            return self.norm2(src)
    class TransformerRegressor(nn.Module):
        def __init__(self, input_dim, seq_len, hidden_dim=64, n_heads=4, num_layers=4):
            super().__init__()
            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.dropout1 = nn.Dropout(p=0.2)
            self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
            self.transformer_layers = nn.ModuleList([
                CustomTransformerEncoderLayer(hidden_dim, n_heads, dropout=0.2) for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout2 = nn.Dropout(p=0.2)
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Dropout(0.4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        def forward(self, x):
            x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
            x = self.dropout1(x)
            for layer in self.transformer_layers:
                x = layer(x)
            x = self.norm(x)
            x = self.dropout2(x)
            return self.regressor(x.mean(dim=1)).squeeze(1)
        def get_attention_weights(self, x):
            x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
            for layer in self.transformer_layers:
                x = layer(x)
            return [layer.attn_weights.detach().cpu() for layer in self.transformer_layers]
    # Train Model
    device = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerRegressor(input_dim=156, seq_len=55).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    n_epochs = 1000
    n_epochs_chk = n_epochs - 1
    for epoch in range(n_epochs):
        model.train()
        total_loss, all_train_preds, all_train_targets = 0, [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_train_preds.extend(pred.detach().cpu().numpy())
            all_train_targets.extend(yb.cpu().numpy())
        model.eval()
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb.to(device)).cpu()
                all_val_preds.extend(pred.numpy())
                all_val_targets.extend(yb.numpy())
        print(f"Epoch {epoch+1}/{n_epochs} | Train MSE: {mean_squared_error(all_train_targets, all_train_preds):.4f} | Train R2: {r2_score(all_train_targets, all_train_preds):.4f} | Val MSE: {mean_squared_error(all_val_targets, all_val_preds):.4f} | Val R2: {r2_score(all_val_targets, all_val_preds):.4f}")
        if epoch == n_epochs_chk:
         targets_orig_t = scaler_y.inverse_transform(np.array(all_train_targets).reshape(-1, 1)).flatten()
         preds_orig_t = scaler_y.inverse_transform(np.array(all_train_preds).reshape(-1, 1)).flatten()
         targets_orig = scaler_y.inverse_transform(np.array(all_val_targets).reshape(-1, 1)).flatten()
         preds_orig = scaler_y.inverse_transform(np.array(all_val_preds).reshape(-1, 1)).flatten()
    
         print("Range of actual reach_dur:", y.min(), "to", y.max())
         print("Range of predicted reach_dur:", preds_orig.min(), "to", preds_orig.max())
    
         plt.figure(figsize=(12, 6))  # landscape 

         plt.subplot(1, 2, 1)
         plt.scatter(targets_orig_t, preds_orig_t, alpha=0.5)
         plt.plot([min(targets_orig_t), max(targets_orig_t)], [min(targets_orig_t), max(targets_orig_t)], 'r--')
         plt.xlabel(f"Train: Actual {var}")
         plt.ylabel(f"Train: Predicted {var}")
         plt.title(f"Train: Epoch {n_epochs}\nMSE: {mean_squared_error(all_train_targets, all_train_preds):.4f}, R2: {r2_score(all_train_targets, all_train_preds):.4f}")
         plt.grid(True)
        
         plt.subplot(1, 2, 2)
         plt.scatter(targets_orig, preds_orig, alpha=0.5)
         plt.plot([min(targets_orig), max(targets_orig)], [min(targets_orig), max(targets_orig)], 'r--')
         plt.xlabel(f"Test: Actual {var}")
         plt.ylabel(f"Test: Predicted {var}")
         plt.title(f"Test: Epoch {n_epochs}\nMSE: {mean_squared_error(all_val_targets, all_val_preds):.4f}, R2: {r2_score(all_val_targets, all_val_preds):.4f}")
         plt.grid(True)
            
         plt.tight_layout(); pdf.savefig(); plt.close()
         plt.show()
    # Visualizations & Analysis    
    model.eval()    
    # Step 0: Prepare test data
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_selected_np = X_test_tensor.cpu().numpy()
    # Step 1: Identify Top 15 Samples with Least MSE
    test_preds = np.array(all_val_preds)
    test_targets = np.array(all_val_targets)
    mse_per_sample = (test_preds - test_targets) ** 2
    top15_indices = np.argsort(mse_per_sample)[:15]
    X_top15 = X_test_tensor[top15_indices]
    # Step 2: Attention Weights from First Layer
    attn_weights = model.get_attention_weights(X_top15)
    layer1_attn_top15 = attn_weights[0]  # shape: [15, 55, 55]
    layer_avg_attn_top15 = layer1_attn_top15.mean(dim=1).cpu().numpy()  # [15, 55]
    
    bin_size = 0.05  # 50 ms
    waveform_len = 156
    fs_fft = waveform_len / bin_size
    freqs = np.fft.rfftfreq(waveform_len, d=1/fs_fft)

    # Step 3: Attention on All Test Samples
    attn_weights_all = model.get_attention_weights(X_test_tensor)
    layer1_attn_all = attn_weights_all[0]  # shape: [N, 55, 55]
    layer_avg_attn_all = layer1_attn_all.mean(dim=1).cpu().numpy()  # [N, 55]
    
    # Plot 2: Attention Weights Across Time Bins (All Test Samples)
    plt.figure(figsize=(8, 6))
    plt.imshow(layer_avg_attn_all, aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention weight')
    plt.xlabel('Time Bin (0–54)')
    plt.ylabel('Test Sample Index')
    plt.title('Attention Weights Across Time Bins (All Test Samples)')
    plt.tight_layout(); pdf.savefig(); plt.close()
    plt.show()
    
    # Plot 4: Waveform Shapes for Top Attention Bins from Plot 2
    bin_mean_all = layer_avg_attn_all.mean(axis=0)
    top_bins_all = np.argsort(bin_mean_all)[-5:]
    print("Top attention bins (All test samples):", top_bins_all)
    
    plt.figure(figsize=(12, 5))
    for bin_idx in top_bins_all:
        avg_waveform = X_selected_np[:, bin_idx, :].mean(axis=0)
        plt.plot(avg_waveform, label=f"Bin {bin_idx}")
    plt.xlabel("Burst Feature Index (1 to 156)")
    plt.ylabel("Amplitude")
    plt.title("Waveform Shapes for Top Attention Bins (All Test Samples)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout(); pdf.savefig(); plt.close()
    plt.show()
    
    # Plot 5: Average Burst Waveform from Top Bins (All Test Samples)
    avg_waveform_allbins = X_selected_np[:, top_bins_all, :].mean(axis=(0, 1))
    
    # Define time axis using bin_size
    t = np.arange(waveform_len) / fs_fft * 1000  # in ms
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, avg_waveform_allbins)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title("Average Burst Waveform from Top Attention Bins (All Test Samples)")
    plt.grid(True)
    plt.tight_layout(); pdf.savefig(); plt.close()
    plt.show()
    
    # Plot X: Mean and Median Attention Weights 
    mean_attn = layer_avg_attn_all.mean(axis=0)
    std_attn = layer_avg_attn_all.std(axis=0)
    sem_attn = std_attn / np.sqrt(layer_avg_attn_all.shape[0])  # Standard Error of the Mean
    
    # Convert bin indices to time (in milliseconds)
    bin_size_sec = 0.05  # 50 ms
    time_ms = np.arange(len(mean_attn)) * bin_size_sec * 1000  # Convert to ms
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_ms, mean_attn, label='Mean Attention', color='blue', linewidth=2)
    plt.fill_between(
        time_ms,
        mean_attn - sem_attn,
        mean_attn + sem_attn,
        color='blue',
        alpha=0.3,
        label='±1 SEM'
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Attention Weight")
    plt.title("Mean Attention Weights ± Standard Error (All Test Samples)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout(); pdf.savefig(); plt.close()
    plt.show()

    
    # Plot 7: FFT of Plot 4
    plt.figure(figsize=(12, 5))
    for bin_idx in top_bins_all:
        avg_waveform = X_selected_np[:, bin_idx, :].mean(axis=0)
        fft_vals = np.abs(np.fft.rfft(avg_waveform))
        plt.plot(freqs, fft_vals, label=f"Bin {bin_idx}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Spectrum (FFT) of Top Attention Bins (All Test Samples)")
    plt.xlim(0, 50)  # <-- Limit to 0–50 Hz
    plt.legend()
    plt.grid(True)
    plt.tight_layout(); pdf.savefig(); plt.close()
    plt.show()
    
# Finalize
pdf.close()
print("PDF saved to:", pdf_path)









