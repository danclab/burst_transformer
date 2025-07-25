
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# Configuration
subject_ids = [f"sub-{i}" for i in [101, 102, 103, 106, 107, 108, 109]]
base_path = "/Users/tanisha/Desktop/CNRS_DATA/aligned_data/"
output_path = "/Users/tanisha/Desktop/CNRS_DATA/aligned_data/"
bin_size = 0.05  # 50 ms bins
waveform_len = 156

all_trials_X = []
all_trials_y = []
all_codes = set()

# Step 1: Collect global peak time range and all unique 'code' values
print("Collecting global peak time range and sensor codes...")
all_peak_times = []

for subj in subject_ids:
    file_path = os.path.join(base_path, f"merged_final_burst_beh_{subj}.csv")
    if os.path.exists(file_path):
        #df = pd.read_csv(file_path, usecols=['trial_target', 'waveform', 'peak_time', 'trial_in_block', 'code'])
        df = pd.read_csv(file_path, usecols=['trial_target', 'reach_rt', 'reach_dur', 'waveform', 'peak_time', 'trial_in_block', 'code'])
        df = df[df['trial_target'].isin([1, 3])]
        df['waveform'] = df['waveform'].apply(json.loads)
        all_peak_times.extend(df['peak_time'].tolist())
        all_codes.update(df['code'].unique())

# Map sensor code strings to indices
sensor_codes = sorted(list(all_codes))
sensor_to_index = {code: idx for idx, code in enumerate(sensor_codes)}
num_sensors = len(sensor_codes)
print(f"Discovered {num_sensors} unique sensors: {sensor_codes}")

min_peak, max_peak = np.min(all_peak_times), np.max(all_peak_times)
print(f"Global peak_time range: {min_peak:.3f}s to {max_peak:.3f}s")

# Define time bins
bins = np.arange(min_peak, max_peak + bin_size, bin_size)
num_bins = len(bins) - 1
print(f"Number of time bins: {num_bins}")
all_trials_reach_rt = []
all_trials_reach_dur = []

# Step 2: Process each trial into (num_bins, num_sensors, 156)
print("Building trial-wise sensor-aware inputs...")
for subj in tqdm(subject_ids):
    file_path = os.path.join(base_path, f"merged_final_burst_beh_{subj}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, usecols=['trial_target',  'reach_rt', 'reach_dur','waveform', 'peak_time', 'trial_in_block', 'code'])
        df = df[df['trial_target'].isin([1, 3])]
        df['waveform'] = df['waveform'].apply(json.loads)

        # Group by trial
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

            trial_matrix = np.array(burst_matrix)  # shape: (num_bins, num_sensors, 156)
            trial_reach_rt = trial_df['reach_rt'].iloc[0]
            all_trials_reach_rt.append(trial_reach_rt)
            trial_reach_dur = trial_df['reach_dur'].iloc[0]
            all_trials_reach_dur.append(trial_reach_dur)
            all_trials_X.append(trial_matrix)
            all_trials_y.append(trial_target)

# Convert to arrays
X = np.array(all_trials_X)  # shape: (num_trials, num_bins, num_sensors, 156)
y = np.array(all_trials_y)  # shape: (num_trials,)

# Reshape to (num_trials * num_sensors, num_bins, 156)
X_reshaped = X.transpose(0, 2, 1, 3).reshape(-1, num_bins, waveform_len)
y_reshaped = np.repeat(y, num_sensors)
reach_rt = np.array(all_trials_reach_rt)
reach_rt_reshaped = np.repeat(reach_rt, num_sensors)
reach_dur = np.array(all_trials_reach_dur)
reach_dur_reshaped = np.repeat(reach_dur, num_sensors)

# Save arrays
np.save(os.path.join(output_path, "X_time_binned_sensored.npy"), X_reshaped)
np.save(os.path.join(output_path, "y_time_binned_sensored.npy"), y_reshaped)
np.save(os.path.join(output_path, "reach_rt_time_binned_sensored.npy"), reach_rt_reshaped)
np.save(os.path.join(output_path, "reach_dur_time_binned_sensored.npy"), reach_dur_reshaped)


print("Final shapes:")
print("X:", X_reshaped.shape)
print("y:", y_reshaped.shape)
print("reach rt",reach_rt_reshaped.shape)
print("reach dur", reach_dur_reshaped.shape)

print(f"Saved to {output_path}")



