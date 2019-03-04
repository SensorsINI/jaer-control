# -*- coding: utf-8 -*-

from os import path
from cochelp import es_utils
import numpy as np
from jaercon.procaedat import load_and_decode_davis_rec, load_and_decode_ams1c
import seaborn as sns
import matplotlib.pyplot as plt

num_x = 240
num_y = 180


def find_trigger(x):
    dx = 1e-3
    dy = np.diff(x) / dx
    return np.where(dy > 1e10)[0][-1] + 1


def count_windows(das_timestamps, w=0.005, sd=0.001):
    min_time = np.min(das_timestamps)
    max_time = np.max(das_timestamps)
    num_windows = int(((max_time - min_time) // 0.001 + 1 - w * 1e3) // (sd * 1e3) + 1)

    return num_windows, max_time


def count_das_spikes(das_timestamps, channels, num_windows, w=0.005, sd=0.001, n_channels=64):
    """Spike count.

    Assume the resolution is 1ms

    # Argument
    timestamps: numpy.ndarray
        list of time stamps (stamps in secs)
    channels: numpy.ndarrary
        list of channel ids
    w: float
        sliding windows size, in (secs)
    sd: float
        stride size, in (secs)
    n_channels: int
        number of channels
    """

    das_spike_count = np.zeros((num_windows, n_channels))
    for t, c in zip(das_timestamps, channels):
        das_spike_count[int(max(0, (t-w)//sd+1)):int(t//sd)+1, c] += 1
    return das_spike_count.T


def calc_events_corr(davis_timestamps, x_addresses, y_addresses, das_spike_count, num_windows, max_time,
                     w=0.005, sd=0.001, num_chunks=9):

    y_chunk_size = num_y//num_chunks

    # select signal
    time_idx = (davis_timestamps <= max_time)
    davis_timestamps = davis_timestamps[time_idx]
    x_addresses = x_addresses[time_idx]
    y_addresses = y_addresses[time_idx]

    spike_chunk = np.zeros((num_windows, y_chunk_size, num_x))
    corr_mat = np.zeros((das_spike_count.shape[0], num_y, num_x))

    for chunk_id in range(num_chunks):
        # selecting right portion of data
        y_idx = np.logical_and((y_addresses >= chunk_id*y_chunk_size),
                               (y_addresses < (chunk_id+1)*y_chunk_size))
        temp_time = davis_timestamps[y_idx]
        temp_x_addrs = x_addresses[y_idx]
        temp_y_addrs = y_addresses[y_idx]

        # filling up this chunk
        for t, x, y in zip(temp_time, temp_x_addrs, temp_y_addrs):
            spike_chunk[int(max(0, (t-w)//sd+1)):int(t//sd)+1, int(y-chunk_id*y_chunk_size), x] += 1

        # computing correlation
        das_spike_count_zeromean = das_spike_count-np.mean(das_spike_count, axis=1).reshape(-1, 1)
        spike_chunk_2d = spike_chunk.reshape((spike_chunk.shape[0], -1))
        spike_chunk_2d_zeromean = spike_chunk_2d-np.mean(spike_chunk_2d, axis=0).T

        cov_chunk = np.dot(das_spike_count_zeromean, spike_chunk_2d_zeromean).reshape(
                (das_spike_count.shape[0], y_chunk_size, num_x))
        std_chunk = np.dot(np.sqrt(np.sum(das_spike_count_zeromean**2, axis=1)).reshape(-1, 1),
                           np.sqrt(np.sum(spike_chunk_2d_zeromean**2, axis=0)).reshape(1, -1)).reshape(
            (das_spike_count.shape[0], y_chunk_size, num_x)) + 1e-4
        corr_chunk = cov_chunk/std_chunk

        if np.any(np.logical_and(corr_chunk > 1, corr_chunk < -1)):
            print("Wrong correlation calculation in chunk {}".format(chunk_id))

        corr_mat[:, chunk_id*y_chunk_size:(chunk_id+1)*y_chunk_size, :] = corr_chunk

        # restore to empty
        spike_chunk = np.zeros((num_windows, y_chunk_size, num_x))

    return corr_mat


folder = "/home/shuwang/Projects/Recordings/3/"
filename_base = "lay_white_in_T_2_now_11"
# filename_base = "place_green_in_C_3_again_0"
davis_save_path = path.join(folder, filename_base+"_davis.aedat")
das_save_path = path.join(folder, filename_base+"_das.aedat")

davis_events = load_and_decode_davis_rec(davis_save_path, verbose=False)
tr_davis = find_trigger(davis_events[0])

# plt.plot([tr_davis, tr_davis], [0, max(davis_events[0])], '--y')
# plt.plot(davis_events[0])
# plt.title("Timestamps for DAVIS")
# plt.xlabel("Index")
# plt.ylabel("Timestamps")
# plt.show()

ts_davis = davis_events[0][tr_davis:]/1e6
x_addrs = davis_events[1][tr_davis:]
y_addrs = davis_events[2][tr_davis:]

# time_diff = np.diff(davis_events[0].astype(np.int64))
# print(np.where(time_diff < 0))

das_ts, das_addrs = es_utils.loadaerdat(das_save_path)
tr_das = find_trigger(das_ts)
real_ts, real_addrs = das_ts[tr_das:], das_addrs[tr_das:]
ts, ch, ear, ne, _ = es_utils.decode_ams1c(real_ts, real_addrs, return_type=False)

# plt.plot([tr_das, tr_das], [0, max(das_ts)], '--y')
# plt.plot(das_ts)
# plt.title("Timestamps for DAS")
# plt.xlabel("Index")
# plt.ylabel("Timestamps")
# plt.show()

n_windows, max_t = count_windows(ts)
das_spk_count = count_das_spikes(ts, ch, n_windows)  # window of 5ms
# print("shape of DAS spike count:{}".format(das_spk_count.shape))
# plt.imshow(das_spk_count, aspect='auto')
# plt.show()

corr_matrix = calc_events_corr(ts_davis, x_addrs, y_addrs, das_spk_count, n_windows, max_t)
if np.any(np.logical_and(corr_matrix > 1, corr_matrix < -1)):
    print("Wrong calculation on correlation.")


# ts, ch, ear, ne, _ = load_and_decode_ams1c(das_save_path, return_type=False)

# plt.clf()
# plt.plot(ts, ch, '.')
# # plt.plot(ts[ear == 0], ch[ear == 0], '.')
# # plt.plot(ts[ear == 1], ch[ear == 1], '.')
# plt.title("DAS Events")
# plt.xlabel("Timestamps (s)")
# plt.ylabel("Channels")
# plt.show()


def plot_corr_region(corr_mat, abs_value=False,
                     intensity_perc=0.5, select_perc_high=0.9, select_perc_low=0.1,
                     region_plot=True, save_plot=False):

    region_save = ""
    abs_save = ""

    if abs_value:
        corr_matrix_mean = np.mean(np.abs(corr_mat), axis=0)
        abs_save = "_abs"
    else:
        corr_matrix_mean = np.mean(corr_mat, axis=0)

    corr_mean_x = np.mean(corr_matrix_mean, axis=0)
    corr_mean_x[corr_mean_x < 0] = 0
    corr_mean_y = np.mean(corr_matrix_mean, axis=1)
    corr_mean_y[corr_mean_y < 0] = 0

    corr_order_x = np.vstack((corr_mean_x, np.arange(0, len(corr_mean_x)))).T
    corr_order_y = np.vstack((corr_mean_y, np.arange(0, len(corr_mean_y)))).T

    corr_order_x = np.flip(corr_order_x[corr_order_x[:, 0].argsort()], axis=0)
    corr_order_y = np.flip(corr_order_y[corr_order_y[:, 0].argsort()], axis=0)

    cutoff_x = np.where(np.cumsum(corr_order_x[:, 0])/np.sum(corr_order_x[:, 0]) > intensity_perc)[0][0]
    cutoff_y = np.where(np.cumsum(corr_order_y[:, 0])/np.sum(corr_order_y[:, 0]) > intensity_perc)[0][0]
    corr_select_x = corr_order_x[:cutoff_x, 1]
    corr_select_y = corr_order_x[:cutoff_y, 1]

    plot_x_max = np.percentile(corr_select_x, select_perc_high)
    plot_x_min = np.percentile(corr_select_x, select_perc_low)
    plot_y_max = np.percentile(corr_select_y, select_perc_high)
    plot_y_min = np.percentile(corr_select_y, select_perc_low)

    plt.figure()
    sns.heatmap(np.flip(corr_matrix_mean, axis=0))
    plt.title("Correlation Map")
    if region_plot:
        plt.hlines(plot_y_max, xmax=plot_x_max, xmin=plot_x_min, colors="w")
        plt.hlines(plot_y_min, xmax=plot_x_max, xmin=plot_x_min, colors="w")
        plt.vlines(plot_x_max, ymax=plot_y_max, ymin=plot_y_min, colors="w")
        plt.vlines(plot_x_min, ymax=plot_y_max, ymin=plot_y_min, colors="w")
        region_save = "_region"

    if save_plot:
        plt.savefig(filename_base+"_corr"+abs_save+region_save, dpi=500)
    else:
        plt.show()


plot_corr_region(corr_matrix[20:40],
                 abs_value=False, intensity_perc=0.99, select_perc_high=90, select_perc_low=10,
                 region_plot=False, save_plot=True)
