"""Compute correlation per frame.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from jaercon import procaedat

# data path
data_path = os.path.join(
    os.environ["HOME"], "data", "lipreading", "004", "2")
file_name_base = "bin_blue_with_L_9_again_8"
#  file_name_base = "bin_white_by_E_2_now_10"

davis_file_path = os.path.join(
    data_path, file_name_base+"_davis.aedat")
das_file_path = os.path.join(
    data_path, file_name_base+"_das.aedat")

# extract DAS events
ts_das, ch, _, _, _ = procaedat.load_and_decode_ams1c(
    das_file_path, return_type=False)
print("[MESSAGE] Loaded DAS data")

# extract DAVIS events
ts_dvs, x_addrs, y_addrs, _, ts_aps, aps_frames = \
    procaedat.load_and_decode_davis_rec(davis_file_path, verbose=False)

print("[MESSAGE] Loaded DAVIS data")

# sync
das_trigger = procaedat.find_trigger((ts_das*1e6).astype(np.uint32))
dvs_trigger = procaedat.find_trigger((ts_dvs*1e6).astype(np.uint32))
aps_trigger = procaedat.find_trigger((ts_aps*1e6).astype(np.uint32))

# filter events
ts_das = ts_das[das_trigger:]
ch = ch[das_trigger:]

ts_dvs = ts_dvs[dvs_trigger:]
x_addrs = x_addrs[dvs_trigger:]
y_addrs = y_addrs[dvs_trigger:]

# select frame
aps_frames = aps_frames[aps_trigger:]
ts_aps = ts_aps[aps_trigger:]
num_frames = aps_frames.shape[0]

# parameters to define resolution
window_size = 0.005
stride_size = 0.001
num_chunks = 9

# plot per frame correlation
for frame_idx in range(1, num_frames):
    time_pre = ts_aps[frame_idx-1]
    time_curr = ts_aps[frame_idx]

    # select das time between frame
    frame_ts_das_idx = np.logical_and(
        ts_das > time_pre,
        ts_das < time_curr)
    frame_ts_das = ts_das[frame_ts_das_idx]
    frame_das_chs = ch[frame_ts_das_idx]

    # select dvs time between frame
    frame_ts_dvs_idx = np.logical_and(
        ts_dvs > time_pre,
        ts_dvs < time_curr)
    frame_ts_dvs = ts_dvs[frame_ts_dvs_idx]
    frame_x_adds = x_addrs[frame_ts_dvs_idx]
    frame_y_adds = y_addrs[frame_ts_dvs_idx]

    # calculate correlation
    frame_num_windows, frame_max_time = procaedat.count_windows(
        frame_ts_das, window_size=window_size, stride_size=stride_size)
    frame_das_spike_rate = procaedat.compute_das_spikerate(
        frame_ts_das, frame_das_chs, frame_num_windows,
        window_size=window_size, stride_size=stride_size,
        channels_range=[20, 40])

    frame_corr_mat = procaedat.compute_events_corr(
        frame_ts_dvs, frame_x_adds, frame_y_adds,
        frame_das_spike_rate, frame_num_windows,
        frame_max_time, window_size=window_size,
        stride_size=stride_size, num_chunks=num_chunks)

    # plotting for correlation map and respective frame
    plt.figure()
    plt.subplot(121)
    sns.heatmap(np.flip(frame_corr_mat.mean(axis=0), axis=0), square=True)
    plt.subplot(122)
    plt.imshow(aps_frames[frame_idx], cmap="gray")
    plt.show()
