#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 11:38:42 2025

@author: tanisha
"""

import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# Configuration
subject_ids = [f"sub-{i}" for i in [
    101, 102, 103, 106, 107, 108, 109, 110, 111, 112, 113, 114, 117, 118, 119, 120,
    122, 123, 124, 126, 127, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140,
    141, 142, 143, 144, 145
]]
base_path = "/Users/tanisha/Desktop/CNRS_DATA/aligned_data_v2/"
output_path = "/Users/tanisha/Desktop/CNRS_DATA/aligned_data_v2/all_samples_omitrows/"
bin_size = 0.05  # 50 ms bins
waveform_len = 156

all_trials_X = []
all_trials_y = []
all_codes = set()
all_peak_times = []

# Step 1: Collect global peak time range and sensor codes
print("Collecting global peak time range and sensor codes...")
for subj in subject_ids:
    file_path = os.path.join(base_path, f"merged_final_burst_err_{subj}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, usecols=[
            'trial_target', 'reach_rt', 'reach_dur',
            'reach_vis_err', 'aim_vis_abs_err', 'aim_vis_err', 'reach_vis_abs_err',
            'waveform', 'peak_time', 'trial_in_block', 'code'
        ])
        df = df[df['trial_target'].isin([1, 3])]
        df['waveform'] = df['waveform'].apply(json.loads)
        all_peak_times.extend(df['peak_time'].tolist())
        all_codes.update(df['code'].unique())

# Sensor and time bin setup
sensor_codes = sorted(list(all_codes))
sensor_to_index = {code: idx for idx, code in enumerate(sensor_codes)}
num_sensors = len(sensor_codes)
min_peak, max_peak = np.min(all_peak_times), np.max(all_peak_times)
bins = np.arange(min_peak, max_peak + bin_size, bin_size)
num_bins = len(bins) - 1

print(f"Discovered {num_sensors} unique sensors: {sensor_codes}")
print(f"Global peak_time range: {min_peak:.3f}s to {max_peak:.3f}s")
print(f"Number of time bins: {num_bins}")

# Initialize containers
all_trials_reach_rt = []
all_trials_reach_dur = []
all_reach_vis_err = []
all_aim_vis_abs_err = []
all_aim_vis_err = []
all_reach_vis_abs_err = []

valid_row_count = 0
invalid_row_count = 0

# Step 2: Process each subject and filter trials
print("Building trial-wise sensor-aware inputs...")
for subj in tqdm(subject_ids):
    file_path = os.path.join(base_path, f"merged_final_burst_err_{subj}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, usecols=[
            'trial_target', 'reach_rt', 'reach_dur',
            'reach_vis_err', 'aim_vis_abs_err', 'aim_vis_err', 'reach_vis_abs_err',
            'waveform', 'peak_time', 'trial_in_block', 'code'
        ])
        df = df[df['trial_target'].isin([1, 3])]
        df['waveform'] = df['waveform'].apply(json.loads)

        # Apply behavioral filters
        valid_mask = (
            df['reach_vis_err'].between(-90, 90) &
            df['aim_vis_err'].between(-90, 90) &
            df['reach_vis_abs_err'].between(0, 90) &
            df['aim_vis_abs_err'].between(0, 90)
        )
        invalid_row_count += (~valid_mask).sum()
        valid_row_count += valid_mask.sum()

        df = df[valid_mask]

        grouped = df.groupby('trial_in_block')
        for trial_id, trial_df in grouped:
            trial_target = trial_df['trial_target'].iloc[0]
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

            trial_matrix = np.array(burst_matrix)  # (num_bins, num_sensors, 156)
            all_trials_X.append(trial_matrix)
            all_trials_y.append(trial_target)
            all_trials_reach_rt.append(trial_df['reach_rt'].iloc[0])
            all_trials_reach_dur.append(trial_df['reach_dur'].iloc[0])
            all_reach_vis_err.append(trial_df['reach_vis_err'].iloc[0])
            all_aim_vis_abs_err.append(trial_df['aim_vis_abs_err'].iloc[0])
            all_aim_vis_err.append(trial_df['aim_vis_err'].iloc[0])
            all_reach_vis_abs_err.append(trial_df['reach_vis_abs_err'].iloc[0])

# Final Summary
print("\n=== Data Filtering Summary ===")
print(f" Valid rows used     : {valid_row_count}")
print(f" Invalid rows skipped: {invalid_row_count}")

# Reshape and save arrays
X = np.array(all_trials_X)  # (num_trials, num_bins, num_sensors, 156)
y = np.array(all_trials_y)
X_reshaped = X.transpose(0, 2, 1, 3).reshape(-1, num_bins, waveform_len)
y_reshaped = np.repeat(y, num_sensors)

reach_rt_reshaped = np.repeat(np.array(all_trials_reach_rt), num_sensors)
reach_dur_reshaped = np.repeat(np.array(all_trials_reach_dur), num_sensors)
reach_vis_err_reshaped = np.repeat(np.array(all_reach_vis_err), num_sensors)
aim_vis_abs_err_reshaped = np.repeat(np.array(all_aim_vis_abs_err), num_sensors)
aim_vis_err_reshaped = np.repeat(np.array(all_aim_vis_err), num_sensors)
reach_vis_abs_err_reshaped = np.repeat(np.array(all_reach_vis_abs_err), num_sensors)

np.save(os.path.join(output_path, "X_time_binned_sensored.npy"), X_reshaped)
np.save(os.path.join(output_path, "y_time_binned_sensored.npy"), y_reshaped)
np.save(os.path.join(output_path, "reach_rt_time_binned_sensored.npy"), reach_rt_reshaped)
np.save(os.path.join(output_path, "reach_dur_time_binned_sensored.npy"), reach_dur_reshaped)
np.save(os.path.join(output_path, "reach_vis_err_time_binned_sensored.npy"), reach_vis_err_reshaped)
np.save(os.path.join(output_path, "aim_vis_abs_err_time_binned_sensored.npy"), aim_vis_abs_err_reshaped)
np.save(os.path.join(output_path, "aim_vis_err_time_binned_sensored.npy"), aim_vis_err_reshaped)
np.save(os.path.join(output_path, "reach_vis_abs_err_time_binned_sensored.npy"), reach_vis_abs_err_reshaped)

# Print shapes
print("\nFinal shapes:")
print("X:", X_reshaped.shape)
print("y:", y_reshaped.shape)
print("reach_rt:", reach_rt_reshaped.shape)
print("reach_dur:", reach_dur_reshaped.shape)
print("reach_vis_err:", reach_vis_err_reshaped.shape)
print("aim_vis_abs_err:", aim_vis_abs_err_reshaped.shape)
print("aim_vis_err:", aim_vis_err_reshaped.shape)
print("reach_vis_abs_err:", reach_vis_abs_err_reshaped.shape)
print(f"\n All arrays saved to: {output_path}")
