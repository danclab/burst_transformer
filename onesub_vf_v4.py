#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 12:22:40 2025

@author: tanisha
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import json

# Load and Preprocess Data
variables = [
    #"reach_dur",
    #"reach_rt",
    "reach_vis_err",
    #"reach_vis_abs_err",
    #"aim_vis_err",
    #"aim_vis_abs_err"
]

subs = [f"sub-{i}" for i in [101, 102, 103
    ,106, 107, 108, 109 ,110
   , 111, 113, 114, 119, 120,
   122, 123, 124, 126, 127, 129, 130, 
   131,133, 134, 135, 136, 138, 139, 142, 143, 144
]]

# Configuration
# subs = [f"sub-{i}" for i in [
#    101, 102, 103, 106, 107, 108, 109, 
#    110, 111, 112, 113, 114, 117, 119, 120,
#    122, 123, 124, 126, 127, 129, 130, 
#    131, 132,133, 134, 135, 136, 138, 139, 140,
#    141, 142, 143, 144, 145
# ]]

device = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")
for var in variables:
    # init subject-level collectors
    global_waveforms = []
    global_attn_weights = []
    global_attn_weights_per_subject = []
    
    print(f"\n===== Processing: {var} =====")
    
    # Create PDF 
    pdf_path = f"/Users/tanisha/Desktop/cnrs_finalcodes_results/analysis_plotsv2_{var}.pdf"
    pdf = PdfPages(pdf_path)
    
    # variable Page
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.5, 0.5, var, fontsize=36, ha='center', va='center', weight='bold')
    pdf.savefig()
    plt.close()
    global_waveforms = []
    global_attn_weights = []
    global_attn_weights_per_subject = []  # <== new
    all_subject_bursts = []  # Store each subject's X_selected_np


    for subject_id in subs:

        print(f"\n===== Processing: {subject_id} =====")
        
        #X = np.load("/Users/tanisha/Desktop/CNRS_DATA/aligned_data_v2/all_samples_omitrows/X_time_binned_sensored.npy")
        #y = np.load(f"/Users/tanisha/Desktop/CNRS_DATA/aligned_data_v2/all_samples_omitrows/{var}_time_binned_sensored.npy")
        #csv_path = f"/Users/tanisha/Desktop/CNRS_DATA/aligned_data_v2/merged_final_burst_err_{subject_id}.csv"
        csv_path = f"/Users/tanisha/Desktop/CNRS_DATA/aligned_data_v2/merged_final_burst_err_{subject_id}.csv"
       
        bin_size = 0.05
        waveform_len = 156
        target_var = var 
        
        # Load and preprocess
        df = pd.read_csv(csv_path)
        df = df[df['trial_target'].isin([1, 3])]
        
        # Remove NaNs and malformed data
        df = df[df[target_var].notna()]
        df = df[df['waveform'].notna()]
        df['waveform'] = df['waveform'].apply(json.loads)
        
        # Filtering based on behavioral criteria
        valid_mask = (
            df['reach_vis_err'].between(-90, 90) &
            df['aim_vis_err'].between(-90, 90) &
            df['reach_vis_abs_err'].between(0, 90) &
            df['aim_vis_abs_err'].between(0, 90)
        )
        df = df[valid_mask]
        
        # Peak time bins
        min_peak, max_peak = df['peak_time'].min(), df['peak_time'].max()
        bins = np.arange(min_peak, max_peak + bin_size, bin_size)
        num_bins = len(bins) - 1
        
        # Sensor codes
        sensor_codes = sorted(df['code'].unique())
        num_sensors = len(sensor_codes)
        
        # Mapping sensor codes to indices
        sensor_to_index = {code: i for i, code in enumerate(sensor_codes)}
        
        # Create trial-wise input matrix
        X_list, y_list = [], []
        grouped = df.groupby("trial_in_block")
        for trial_id, trial_df in grouped:
            burst_matrix = []
            for i in range(num_bins):
                t_start, t_end = bins[i], bins[i + 1]
                bin_slice = trial_df[(trial_df['peak_time'] >= t_start) & (trial_df['peak_time'] < t_end)]
                
                bin_sensor_waveforms = []
                for code in sensor_codes:
                    sensor_bursts = bin_slice[bin_slice['code'] == code]
                    if not sensor_bursts.empty:
                        waveforms = np.array(sensor_bursts['waveform'].tolist())
                        avg_waveform = np.mean(waveforms, axis=0)
                    else:
                        avg_waveform = [0.0] * waveform_len
                    bin_sensor_waveforms.append(avg_waveform)
                
                burst_matrix.append(bin_sensor_waveforms)
            
            if len(burst_matrix) == num_bins:
                trial_matrix = np.array(burst_matrix)  # shape: (num_bins, num_sensors, 156)
                # Reshape from (num_bins, num_sensors, 156) → (num_sensors, num_bins, 156)
                trial_matrix = trial_matrix.transpose(1, 0, 2)
                for sensor_matrix in trial_matrix:  # shape: (num_bins, 156)
                    X_list.append(sensor_matrix)
                    y_list.append(trial_df.iloc[0][target_var])
        
        X = np.array(X_list)  # shape: (num_trials × sensors, num_bins, 156)
        y = np.array(y_list)
        # Subject id Page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, subject_id, fontsize=36, ha='center', va='center', weight='bold')
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
        # Dataset
        class TimeBinnedBurstDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)
            def __len__(self): return len(self.X)
            def __getitem__(self, idx): return self.X[idx], self.y[idx]
        # ? Define the datasets
        train_dataset = TimeBinnedBurstDataset(X_train, y_train)
        test_dataset = TimeBinnedBurstDataset(X_test, y_test)
        # ? Create DataLoaders
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
        model = TransformerRegressor(input_dim=156, seq_len=55).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.MSELoss()
        n_epochs = 1000
        n_epochs_chk = n_epochs - 1
        max_val_r2 = 0
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
            train_mse = mean_squared_error(all_train_targets, all_train_preds)
            train_r2 = r2_score(all_train_targets, all_train_preds)
            val_mse = mean_squared_error(all_val_targets, all_val_preds)
            val_r2 = r2_score(all_val_targets, all_val_preds)
            
            if (val_r2 > max_val_r2):
                max_val_r2 = val_r2
                max_val_mse = val_mse
                max_train_r2 = train_r2
                max_train_mse = train_mse
                max_epoch = epoch+1
                print(f"Epoch {epoch+1}/{n_epochs} | Train MSE: {train_mse:.4f} | Train R2: {train_r2:.4f} | Val MSE: {val_mse:.4f} | Val R2: {val_r2:.4f}")

                # Save best model
                model_save_path = f"/Users/tanisha/Desktop/cnrs_finalcodes_results/models/best_model_{subject_id}_{var}.pt"
                torch.save({
                    'epoch': max_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_r2': max_train_r2,
                    'val_r2': max_val_r2,
                    'train_mse': max_train_mse,
                    'val_mse': max_val_mse,
                }, model_save_path)
                print(f"✅ Best model saved to {model_save_path}")
                
                targets_orig_t = scaler_y.inverse_transform(np.array(all_train_targets).reshape(-1, 1)).flatten()
                preds_orig_t = scaler_y.inverse_transform(np.array(all_train_preds).reshape(-1, 1)).flatten()
                targets_orig = scaler_y.inverse_transform(np.array(all_val_targets).reshape(-1, 1)).flatten()
                preds_orig = scaler_y.inverse_transform(np.array(all_val_preds).reshape(-1, 1)).flatten()
        
                # print("Range of actual reach_dur:", y.min(), "to", y.max())
                # print("Range of predicted reach_dur:", preds_orig.min(), "to", preds_orig.max())
        
        print(f"Max -- Epoch {max_epoch} | Train MSE: {max_train_mse:.4f} | Train R2: {max_train_r2:.4f} | Val MSE: {max_val_mse:.4f} | Val R2: {max_val_r2:.4f}")
        plt.figure(figsize=(12, 6))  # landscape size
    
        plt.subplot(1, 2, 1)
        plt.scatter(targets_orig_t, preds_orig_t, alpha=0.5)
        plt.plot([min(targets_orig_t), max(targets_orig_t)], [
            min(targets_orig_t), max(targets_orig_t)], 'r--')
        plt.xlabel(f"Train: Actual {var}")
        plt.ylabel(f"Train: Predicted {var}")
        plt.title(
            f"Train: Epoch {max_epoch}\nMSE: {max_train_mse:.4f}, R2: {max_train_r2:.4f}")
        plt.grid(True)
            
        plt.subplot(1, 2, 2)
        plt.scatter(targets_orig, preds_orig, alpha=0.5)
        plt.plot([min(targets_orig), max(targets_orig)], [
            min(targets_orig), max(targets_orig)], 'r--')
        plt.xlabel(f"Test: Actual {var}")
        plt.ylabel(f"Test: Predicted {var}")
        plt.title(
            f"Val: Epoch {max_epoch}\nMSE: {max_val_mse:.4f}, R2: {max_val_r2:.4f}")
        plt.grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.show()
        plt.close()

        # Visualizations & Analysis        
        model.eval()
        # Prepare test data
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        X_selected_np = X_test_tensor.cpu().numpy()
        all_subject_bursts.append(X_selected_np)
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
        # FFT of Plot 3
        bin_size = 0.05  # 50 ms
        waveform_len = 156
        fs_fft = waveform_len / bin_size
        freqs = np.fft.rfftfreq(waveform_len, d=1/fs_fft)
        # Attention on All Test Samples
        attn_weights_all = model.get_attention_weights(X_test_tensor)
        layer1_attn_all = attn_weights_all[0]  # shape: [N, 55, 55]
        layer_avg_attn_all = layer1_attn_all.mean(dim=1).cpu().numpy()  # [N, 55]
        
        # Attention Weights Across Time Bins (All Test Samples)
        plt.figure(figsize=(8, 6))
        plt.imshow(layer_avg_attn_all, aspect='auto', cmap='viridis')
        plt.colorbar(label='Attention weight')
        plt.xlabel('Time Bin (0–54)')
        plt.ylabel('Test Sample Index')
        plt.title('Attention Weights Across Time Bins (All Test Samples)')
        plt.tight_layout(); pdf.savefig(); plt.close()
        plt.show()
        
        # Waveform Shapes for Top Attention Bins from Plot 2
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
        
        # Plot X: Mean Attention Weights with Standard Error Shading
        mean_attn = layer_avg_attn_all.mean(axis=0)
        std_attn = layer_avg_attn_all.std(axis=0)
        sem_attn = std_attn / np.sqrt(layer_avg_attn_all.shape[0])  # Standard Error of the Mean
        median_attn = np.median(layer_avg_attn_all, axis=0)  # shape: (55,)

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
        plt.plot(time_ms, median_attn, label='Median Attention', color='orange', linestyle='--', linewidth=2)
        
        plt.xlabel("Time (ms)")
        plt.ylabel("Attention Weight")
        plt.title("Mean & Median Attention Weights ± SEM (This Subject)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout(); pdf.savefig(); plt.close()
        plt.show()
            
        global_waveforms.append(avg_waveform_allbins)
        global_attn_weights.append(layer_avg_attn_all)
        global_attn_weights_per_subject.append(layer_avg_attn_all)  # For subject-wise overlay
        # global_waveforms_by_var[var].append(avg_waveform_allbins)
        # global_attn_weights_by_var[var].append(layer_avg_attn_all)
        # global_attn_weights_per_subject_by_var[var].append(layer_avg_attn_all)


        # Plot 7: FFT of Plot 4 (limited to 0–50 Hz)
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

        
    # Global plots (after processing all subjects for this var)
    if global_waveforms and global_attn_weights:
        
        global_waveforms = np.stack(global_waveforms)  # shape: [num_subjects, 156]
        avg_waveform_across_subjects = global_waveforms.mean(axis=0)
    
        plt.figure(figsize=(10, 4))
        plt.plot(t, avg_waveform_across_subjects)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.title(f"Global Average Burst Waveform — {var}")
        plt.grid(True)
        plt.tight_layout(); pdf.savefig(); plt.close()
        plt.show()
    
        # Global Attention Plot with SEM
        global_attn_weights = np.concatenate(global_attn_weights, axis=0)
        mean_attn_global = global_attn_weights.mean(axis=0)
        sem_attn_global = global_attn_weights.std(axis=0) / np.sqrt(global_attn_weights.shape[0])
        median_attn_global = np.median(global_attn_weights, axis=0)
        time_ms = np.arange(len(mean_attn_global)) * bin_size * 1000
        
        plt.figure(figsize=(10, 5))
        plt.plot(time_ms, mean_attn_global, label='Mean Attention (Global)', color='darkgreen', linewidth=2)
        plt.fill_between(
            time_ms,
            mean_attn_global - sem_attn_global,
            mean_attn_global + sem_attn_global,
            color='darkgreen',
            alpha=0.3,
            label='±1 SEM (Global)'
        )
        plt.plot(time_ms, median_attn_global, label='Median Attention (Global)', color='orange', linestyle='--', linewidth=2)
        
        plt.xlabel("Time (ms)")
        plt.ylabel("Attention Weight")
        plt.title(f"Global Attention Weights ± SEM & Median — {var}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout(); pdf.savefig(); plt.close()
        plt.show()
        plt.figure(figsize=(14, 6))  # Make it wider and slightly taller

        for i, subj_waveform in enumerate(global_waveforms):
            plt.plot(t, subj_waveform, label=f'{subs[i]}')
        
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.title(f"Per-Subject Burst Waveforms (avg waveform across top 5 att. bins) — {var}")
        
        # Move legend outside
        plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='x-small', frameon=False, ncol=1)
        
        # Adjust layout to leave space for the legend
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Allocate 15% space for legend
        pdf.savefig(); plt.close()
        plt.show()
        # PLOT 1: Top Bin Per Subject
        top_bins = []
        top_times_ms = []
        plt.figure(figsize=(14, 6))  # Wider and taller

        for i, (attn, bursts) in enumerate(zip(global_attn_weights_per_subject, all_subject_bursts)):  
            top_bin = np.argmax(attn.mean(axis=0))  # subject-specific top bin
            top_time_ms = top_bin * bin_size * 1000
            avg_waveform = bursts[:, top_bin, :].mean(axis=0)
        
            label = f'{subs[i]} (Bin {top_bin} → {int(top_time_ms)}ms)'
            plt.plot(t, avg_waveform, label=label)
        
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.title("Per-Subject Burst Waveform from Top Attention Bin")
        
        # Move legend outside but more compact
        plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='x-small', frameon=False, ncol=1)
        
        # Adjust layout to use more space for plot
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Only 15% margin for legend
        pdf.savefig(); plt.close()
        plt.show()
        # PLOT 2: Global Peak Attention Bin
        global_top_bin = np.argmax(mean_attn_global)
        bin_time_start = global_top_bin * bin_size * 1000
        bin_time_end = (global_top_bin + 1) * bin_size * 1000
        
        plt.figure(figsize=(14, 6))  # Wider plot to accommodate external legend
        
        for i, bursts in enumerate(all_subject_bursts):
            avg_waveform = bursts[:, global_top_bin, :].mean(axis=0)
            plt.plot(t, avg_waveform, label=f'{subs[i]}')
        
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.title(f"Per-Subject Waveform from Global Top Attention Bin {global_top_bin} ({bin_time_start:.0f}–{bin_time_end:.0f} ms)")
        
        # External legend
        plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='x-small', frameon=False, ncol=1)
        
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve space on the right for legend
        pdf.savefig(); plt.close()
        plt.show()
                

    # plt.show()
    plt.figure(figsize=(14, 6))  # Make plot wider to accommodate legend

    for i, subj_attn in enumerate(global_attn_weights_per_subject):
        subj_mean = subj_attn.mean(axis=0)
        plt.plot(time_ms, subj_mean, label=f'{subs[i]}')
    
    plt.xlabel("Time (ms)")
    plt.ylabel("Attention Weight")
    plt.title(f"Per-Subject Attention Weights — {var}")
    
    # Move legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='x-small', frameon=False, ncol=1)
    
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right
    pdf.savefig(); plt.close()
    plt.show()


# Finalize
    pdf.close()
    print("PDF saved to:", pdf_path)









