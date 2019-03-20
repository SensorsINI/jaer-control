"""Test DVS event correlation.

The basic idea is the on-event and off-event should be correlated
if it's not some noise.

This implementation may not be efficient.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import os

import numpy as np
import h5py

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from jaercon import procaedat


def compute_dvs_chunk_spikerate(dvs_spike_rate, timestamps,
                                x_addresses, y_addresses,
                                chunk_id, chunk_size,
                                window_size=0.005, stride_size=0.001):
    """Compute a chunk of DVS spike rate (in-place modification).

    Assume chunk on y-axis only

    # Argument
    dvs_spike_rate: numpy.ndarray
        in (num_windows, height, width), initialized elsewhere beforehand
    timestamps: numpy.ndarray
        list of time stamps  in secs
    x_addresses: numpy.ndarray
        list of x address, assume [0, 239] if DAVIS240
    y_addresses: numpy.ndarray
        list of y address, assume [0, 179] if DAVIS240
    chunk_id: int
        the index of the chunk to be computed with
    chunk_size: float
        size of the chunk in y axis to be computed with
    window_size: float
        sliding windows size, in secs
    stride_size: float
        stride size, in secs
    """
    for t, x, y in zip(timestamps, x_addresses, y_addresses):
        dvs_spike_rate[int(max(0, (t-window_size)//stride_size+1)):
                       int(t//stride_size)+1,
                       int(y-chunk_id*chunk_size), x] += 1

    return dvs_spike_rate


# load data
data_path = os.path.join(
    os.environ["HOME"], "data", "lipreading")
file_name_base = "20Hz"

davis_file_path = os.path.join(
    data_path, file_name_base+"_davis.aedat")

# extract DAVIS events
ts_dvs, x_addrs, y_addrs, pol = \
    procaedat.load_and_decode_davis_rec(
        davis_file_path, events_only=True, verbose=False)

print("[MESSAGE] Data loaded")

# remove data before trigger
#  dvs_trigger = procaedat.find_trigger((ts_dvs*1e6).astype(np.uint32))
#  end_selection = dvs_trigger+25000
#  ts_dvs = ts_dvs[dvs_trigger:]
#  x_addrs = x_addrs[dvs_trigger:]
#  y_addrs = y_addrs[dvs_trigger:]
#  pol = pol[dvs_trigger:]

data_export = h5py.File("test_recording.h5", "w")
data_export.create_dataset("ts", dtype=np.float32, data=ts_dvs)
data_export.create_dataset("x_addrs", dtype=np.float32, data=x_addrs)
data_export.create_dataset("y_addrs", dtype=np.float32, data=y_addrs)
data_export.create_dataset("pol", dtype=np.float32, data=pol)
data_export.flush()

data_export.close()

print("Duration: {}".format(ts_dvs[-1]-ts_dvs[0]))

#  fig = plt.figure()
#  ax = fig.add_subplot(111, projection='3d')
#  ax.scatter(ts_dvs, x_addrs, y_addrs, s=20, alpha=0.6, edgecolors='w')
#  plt.show()

# select events
pos_idx = (pol == 1)
neg_idx = (~pos_idx)

pos_ts, neg_ts = ts_dvs[pos_idx], ts_dvs[neg_idx]
pos_x, neg_x = x_addrs[pos_idx], x_addrs[neg_idx]
pos_y, neg_y = y_addrs[pos_idx], y_addrs[neg_idx]


#  fig = plt.figure()
#  ax = fig.add_subplot(111, projection='3d')
#  ax.scatter(pos_ts, pos_x, pos_y, s=20, alpha=0.6, edgecolors='w')
#  ax.scatter(neg_ts, neg_x, neg_y, s=20, alpha=0.6, edgecolors='r')
#  plt.show()

# construct spike rate matrix
window_size = 0.005
stride_size = 0.001
height, width = 260, 346
num_chunks = 9
chunk_size = height//num_chunks
num_windows, min_time, max_time = procaedat.count_windows(
    ts_dvs, window_size=window_size, stride_size=stride_size)

pos_spikerate = np.zeros((num_windows, height, width), dtype=np.float32)
neg_spikerate = np.zeros((num_windows, height, width), dtype=np.float32)
pos_temp_chunk = np.zeros((num_windows, chunk_size, width), dtype=np.float32)
neg_temp_chunk = np.zeros((num_windows, chunk_size, width), dtype=np.float32)

print("Prepared to compute spike rates")

# compute spike rate matrix
for chunk_id in range(num_chunks):
    # positive
    y_idx = np.logical_and((pos_y >= chunk_id*chunk_size),
                           (pos_y < (chunk_id+1)*chunk_size))
    temp_time = pos_ts[y_idx]
    temp_x_addrs = pos_x[y_idx]
    temp_y_addrs = pos_y[y_idx]

    pos_temp_chunk = compute_dvs_chunk_spikerate(
        pos_temp_chunk, temp_time, temp_x_addrs,
        temp_y_addrs, chunk_id, chunk_size,
        window_size, stride_size)
    # negative
    y_idx = np.logical_and((neg_y >= chunk_id*chunk_size),
                           (neg_y < (chunk_id+1)*chunk_size))
    temp_time = neg_ts[y_idx]
    temp_x_addrs = neg_x[y_idx]
    temp_y_addrs = neg_y[y_idx]

    neg_temp_chunk = compute_dvs_chunk_spikerate(
        neg_temp_chunk, temp_time, temp_x_addrs,
        temp_y_addrs, chunk_id, chunk_size,
        window_size, stride_size)

    # load up chunk
    pos_spikerate[:, chunk_id*chunk_size:(chunk_id+1)*chunk_size, :] = \
        pos_temp_chunk
    neg_spikerate[:, chunk_id*chunk_size:(chunk_id+1)*chunk_size, :] = \
        neg_temp_chunk

    # reset temp chunks
    pos_temp_chunk = np.zeros(
        (num_windows, chunk_size, width), dtype=np.float32)
    neg_temp_chunk = np.zeros(
        (num_windows, chunk_size, width), dtype=np.float32)

    print("Computed {} chunks".format(chunk_id+1))

# compute correlation
pos_img = np.sum(pos_spikerate, axis=0)
neg_img = np.sum(neg_spikerate, axis=0)
pos_spikerate -= np.mean(pos_spikerate, axis=0, keepdims=True)
neg_spikerate -= np.mean(neg_spikerate, axis=0, keepdims=True)

corr_matrix = np.sum(pos_spikerate*neg_spikerate, axis=0)
stds = np.sqrt(np.sum(pos_spikerate**2, axis=0)) * \
    np.sqrt(np.sum(neg_spikerate**2, axis=0))+1e-5
corr_matrix /= stds


print("Correlation matrix calculated")

plt.figure()
plt.subplot(131)
sns.heatmap(np.flip(corr_matrix, axis=(0, 1)), square=True)
plt.subplot(132)
sns.heatmap(np.flip(pos_img, axis=(0, 1)), square=True)
plt.subplot(133)
sns.heatmap(np.flip(neg_img, axis=(0, 1)), square=True)
plt.show()
