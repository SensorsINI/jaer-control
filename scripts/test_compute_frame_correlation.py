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
    os.environ["HOME"], "data", "lipreading", "004", "4")
#  file_name_base = "bin_blue_with_L_9_again_8"
#  file_name_base = "bin_white_by_E_2_now_10"
#  file_name_base = "bin_red_with_R_3_now_3"
file_name_base = "bin_blue_at_Q_6_again_16"

davis_file_path = os.path.join(
    data_path, file_name_base+"_davis.aedat")
das_file_path = os.path.join(
    data_path, file_name_base+"_das.aedat")

# extract DAS events
ts_das, ch, _, _, _ = procaedat.load_and_decode_ams1c(
    das_file_path, return_type=False)
print("[MESSAGE] Loaded DAS data")

# extract DAVIS events
ts_dvs, x_addrs, y_addrs, pol, ts_aps, aps_frames = \
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
pol = pol[dvs_trigger:]

# select frame
aps_frames = aps_frames[aps_trigger:]
ts_aps = ts_aps[aps_trigger:]
num_frames = aps_frames.shape[0]

# parameters to define resolution
window_size = 0.005
stride_size = 0.001
num_chunks = 9
clip_value = 5
histrange = [(0, v) for v in (180, 240)]

time_pre = ts_aps[0]
frame_counter = 0

# plot per frame correlation
for frame_idx in range(1, num_frames):
    # select time
    #  time_pre = ts_aps[frame_idx-1]
    time_curr = ts_aps[frame_idx]
    print("Time pre:", time_pre, "Time curr:", time_curr)

    # select das time between frame
    frame_ts_das_idx = np.logical_and(
        ts_das >= time_pre,
        ts_das <= time_curr)
    frame_ts_das = ts_das[frame_ts_das_idx]
    frame_das_chs = ch[frame_ts_das_idx]

    # select dvs time between frame
    frame_ts_dvs_idx = np.logical_and(
        ts_dvs >= time_pre,
        ts_dvs <= time_curr)
    frame_ts_dvs = ts_dvs[frame_ts_dvs_idx]
    frame_x_adds = x_addrs[frame_ts_dvs_idx]
    frame_y_adds = y_addrs[frame_ts_dvs_idx]
    frame_pol = pol[frame_ts_dvs_idx]

    # construct DVS histogram
    # TODO: to improve
    pol_on = (frame_pol == 1)
    pol_off = np.logical_not(pol_on)
    img_on, _, _ = np.histogram2d(
            180-frame_y_adds[pol_on], 240-frame_x_adds[pol_on],
            bins=(180, 240), range=histrange)
    img_off, _, _ = np.histogram2d(
            180-frame_x_adds[pol_off], 240-frame_x_adds[pol_off],
            bins=(180, 240), range=histrange)
    if clip_value is not None:
        integrated_img = np.clip(
            (img_on-img_off), -clip_value, clip_value)
    else:
        integrated_img = (img_on-img_off)
    img = integrated_img+clip_value
    dvs_img = img/float(clip_value*2)

    if frame_ts_das_idx.sum()*frame_ts_dvs_idx.sum() == 0 or \
            frame_ts_das_idx.sum() < 300:
        continue
    else:
        time_pre = ts_aps[frame_idx]

    # calculate correlation
    frame_num_windows, frame_min_time, frame_max_time = \
        procaedat.count_windows(
            frame_ts_das, window_size=window_size, stride_size=stride_size)

    frame_das_spike_rate = procaedat.compute_das_spikerate(
        frame_ts_das, frame_das_chs, frame_num_windows, frame_min_time,
        window_size=window_size, stride_size=stride_size,
        channels_range=[1, 64])

    # TODO: to investigate if the correlation calculation is correct.
    # TODO: why the axes are flipped.
    frame_corr_mat = procaedat.compute_events_corr(
        frame_ts_dvs, frame_x_adds, frame_y_adds,
        frame_das_spike_rate, frame_num_windows,
        frame_min_time, frame_max_time, window_size=window_size,
        stride_size=stride_size, num_chunks=num_chunks)

    # plotting for correlation map and respective frame
    #  frame_corr_mat = np.abs(frame_corr_mat)
    frame_corr_mat = np.flip(frame_corr_mat.mean(axis=0), axis=0)
    frame_corr_mat = np.flip(frame_corr_mat, axis=1)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    sns.heatmap(frame_corr_mat, square=True, cbar_kws={"shrink": .5})
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(aps_frames[frame_idx], cmap="gray")
    plt.axis("off")
    plt.title("Frame {}; Time {}s".format(frame_idx, time_curr))
    plt.subplot(133)
    plt.imshow(dvs_img, cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    vid_save_path = os.path.join(
        os.environ["HOME"], "data", "lipreading", "video_folder",
        file_name_base)
    if not os.path.isdir(vid_save_path):
        os.makedirs(vid_save_path)

    plt.savefig(
        os.path.join(
            vid_save_path, file_name_base+"_{}.jpeg".format(frame_counter)),
        dpi=300)

    plt.close()
    print("Frame {}/{}".format(frame_idx, num_frames))
    frame_counter += 1
