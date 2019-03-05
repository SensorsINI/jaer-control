"""Processing routines for AEDAT2.

Author: Shu Wang, Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import
import os
import numpy as np

# DAVIS related constant
EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event
sizeX = 240
sizeY = 180
x0 = 1
y0 = 1

# DAS related constants
NB_CHANNELS = 64
NB_NEURON_TYPES = 4
NB_EARS = 2
NB_FILTER_BANKS = 2
NB_ON_OFF = 2

# AEDAT2 constants
ae_len = 8  # 1 AE event takes 8 bytes
read_mode = '>u4'  # struct.unpack(), 2x ulong, 4B+4B
td = 0.000001  # timestep is 1u

x_mask = 0x003ff000
x_shift = 12
y_mask = 0x7fc00000
y_shift = 22
p_mask = 0x800
p_shift = 11
event_type_shift = 31
adc_mask = 0x000003ff
signal_mask = int('400', 16)


def skip_header(file_obj, verbose=False):
    """Skip the header of an AEDAT file.

    # Parameters
        file_obj : file
            A valid file object that handles the current file.

    # Returns
        sum_of_bytes : int
            sum of bytes for the file object
    """
    assert file_obj is not None

    sum_of_bytes = 0
    for line in file_obj:
        line = line.decode("utf-8")
        if verbose is True:
            print(line)
        if line[0] == "#":
            sum_of_bytes += len(line)
        if "#End Of ASCII Header" in line:
            break

    return sum_of_bytes


def get_file_length(file_path):
    """Return file length in bytes."""
    statinfo = os.stat(file_path)
    length = statinfo.st_size

    return length


def check_davis_rec(file_path, level=0, verbose=False):
    """Check if the DAVIS Recording is valid.

    # Parameters
    file_path : str
        the absolute path of the DAVIS
    level : int
        Checking Level.
        0: check if there is any bytes besides header
        1: check if all the DVS events are valid
        2: return the timestamps for plotting
    verbose : bool
        print debugging comments if True

    # Returns
    flag : bool
        the file is valid if True
        False otherwise
    timestamps : numpy.ndarray
        only return this object if check level is True
    """
    # check if the file exists
    assert os.path.isfile(file_path) is True

    davis_file = open(file_path, "rb")
    file_length = get_file_length(file_path)
    header_length = skip_header(davis_file)

    if verbose is True:
        print("Header length:", header_length, "bytes")
        print("File length:", file_length, "bytes")

    num_events = (file_length-header_length)//ae_len
    if verbose is True:
        print("Number of events:", num_events)

    # checking number of events
    if level == 0:
        flag = True if num_events > 0 else False
        return flag

    # decode events
    davis_events = np.fromfile(
        davis_file, dtype=read_mode, count=2*num_events).reshape(
            (num_events, 2)).astype(np.uint64)
    davis_file.close()
    if verbose is True:
        print("Number of events:", davis_events.shape)

    # decode event address
    event_address = davis_events[:, 0]
    timestamps = davis_events[:, 1]

    event_types = event_address >> event_type_shift

    # decode DVS events
    events_ts = timestamps[event_types == 0]
    event_x_addrs = (event_address[event_types == 0] & x_mask) >> x_shift
    event_y_addrs = (event_address[event_types == 0] & y_mask) >> y_shift

    event_x_valid = np.logical_and(
        (event_x_addrs > 240), (event_x_addrs < 0)).sum()
    event_y_valid = np.logical_and(
        (event_y_addrs > 180), (event_y_addrs < 0)).sum()

    if verbose is True:
        print(event_x_valid)
        print(event_y_valid)

    event_valid = not bool(event_x_valid*event_y_valid)

    if level == 1:
        return event_valid

    if level == 2:
        if event_valid is True:
            return event_valid, events_ts
        else:
            return False, None


def check_das_rec(file_path, level=0, verbose=False):
    """Check DAS recordings."""
    assert os.path.isfile(file_path)

    das_file = open(file_path, "rb")

    # get some basic statics of the file and skip header of the file
    num_bytes_per_event = 8
    file_length = get_file_length(file_path)
    header_length = skip_header(das_file)
    if verbose is True:
        print("Header length:", header_length, "bytes")
        print("File length:", file_length, "bytes")
    num_events = (file_length-header_length)//num_bytes_per_event
    if verbose is True:
        print("Number of events:", num_events)

    if level == 0:
        flag = True if num_events > 0 else False
        return flag

    # decode data
    das_events = np.fromfile(
        das_file, dtype='>u4', count=2*num_events).reshape(
            (num_events, 2)).astype(np.uint32)
    das_file.close()

    if verbose is True:
        print("Number of events:", das_events.shape)

    if level == 1:
        flag = True if num_events > 0 else False
        return flag, das_events[:, 1]


def load_and_decode_davis_rec(file_path, length=0, verbose=False):
    """Load DAVIS AER data.

    Only supports AEDAT2+DAVIS240

    # Parameters
    file_path : string
        the absolute file path for the AEDAT file.
    length : int
        Number of bytes(B) should be read.
        default: 0 to read whole file

    # Returns
    dvs_events : numpy.ndarray
        A 4D numpy array that stores the DVS events:
        [timestamps, x_pos, y_pos, polarity]
    aps_events : list
        List of frame events
    """
    # check if the file exists
    assert os.path.isfile(file_path) is True

    davis_file = open(file_path, "rb")
    file_length = get_file_length(file_path)
    header_length = skip_header(davis_file)

    if verbose is True:
        print("Header length:", header_length, "bytes")
        print("File length:", file_length, "bytes")

    num_events = (file_length-header_length)//ae_len
    if verbose is True:
        print("Number of events:", num_events)

    # decode events
    davis_events = np.fromfile(
        davis_file, dtype=read_mode, count=2*num_events).reshape(
            (num_events, 2)).astype(np.uint64)
    davis_file.close()
    if verbose is True:
        print("Number of events:", davis_events.shape)

    # decode event address
    event_address = davis_events[:, 0]
    timestamps = davis_events[:, 1]

    event_types = event_address >> event_type_shift

    # decode DVS events
    events_ts = timestamps[event_types == 0]
    event_x_addrs = (event_address[event_types == 0] & x_mask) >> x_shift
    event_y_addrs = (event_address[event_types == 0] & y_mask) >> y_shift
    event_pols = (event_address[event_types == 0] & p_mask) >> p_shift

    # decode frame events
    frame_ts = timestamps[event_types == 1]
    frame_events = event_address[event_types == 1]
    frame_x = (frame_events & x_mask) >> x_shift
    frame_y = (frame_events & y_mask) >> y_shift
    frame_adcs = np.array(np.bitwise_and(
        frame_events, adc_mask), dtype=np.uint16)
    frame_signal = (frame_events & signal_mask).astype(np.bool)

    frame_x_discont = np.abs(frame_x[1:]-frame_x[0:-1]) > 1
    frame_y_discont = np.abs(frame_y[1:]-frame_y[0:-1]) > 1

    frame_discont_index = np.where(
        np.logical_and(frame_x_discont, frame_y_discont))[0]
    frame_starts = np.concatenate(
        [[0], frame_discont_index+1, [frame_events.size]])

    num_frames = frame_starts.size-1

    outputData = {}
    outputData['reset'] = np.zeros(num_frames, 'bool')
    outputData['timeStampStart'] = np.zeros(num_frames, 'uint32')
    outputData['timeStampEnd'] = np.zeros(num_frames, 'uint32')
    outputData['samples'] = np.empty(num_frames, 'object')
    outputData['xLen'] = np.zeros(num_frames, 'uint16')
    outputData['yLen'] = np.zeros(num_frames, 'uint16')
    outputData['xPos'] = np.zeros(num_frames, 'uint16')
    outputData['yPos'] = np.zeros(num_frames, 'uint16')

    # start construct frames
    for frame_idx in range(num_frames):
        outputData["reset"][frame_idx] = \
            not frame_signal[frame_starts[frame_idx]]

        outputData["timeStampStart"][frame_idx] = \
            min(frame_ts[frame_starts[frame_idx]:frame_starts[frame_idx+1]])

        outputData["timeStampEnd"][frame_idx] = \
            max(frame_ts[frame_starts[frame_idx]:frame_starts[frame_idx+1]])

        temp_x_pos = min(
            frame_x[frame_starts[frame_idx]:frame_starts[frame_idx+1]])
        outputData["xPos"][frame_idx] = temp_x_pos

        temp_y_pos = min(
            frame_y[frame_starts[frame_idx]:frame_starts[frame_idx+1]])
        outputData["yPos"][frame_idx] = temp_y_pos

        outputData["xLen"][frame_idx] = \
            max(frame_x[
                frame_starts[frame_idx]:frame_starts[frame_idx+1]]) - \
            outputData["xPos"][frame_idx]+1
        outputData["yLen"][frame_idx] = \
            max(frame_y[
                frame_starts[frame_idx]:frame_starts[frame_idx+1]]) - \
            outputData["yPos"][frame_idx]+1

        # construct frame
        temp_samples = np.zeros(
            (outputData["yLen"][frame_idx],
             outputData["xLen"][frame_idx]),
            dtype=np.uint16)

        for sample_idx in range(
                frame_starts[frame_idx], frame_starts[frame_idx+1]):
            temp_samples[
                frame_y[sample_idx]-outputData["yPos"][frame_idx],
                frame_x[sample_idx]-outputData["xPos"][frame_idx]] = \
                frame_adcs[sample_idx]

        outputData["samples"][frame_idx] = np.flip(temp_samples, axis=0)

    frame_count = 0

    for frame_idx in range(num_frames):
        if outputData["reset"][frame_idx]:
            reset_frame = outputData["samples"][frame_idx]
            reset_x_pos = outputData["xPos"][frame_idx]
            reset_y_pos = outputData["yPos"][frame_idx]
            reset_x_len = outputData["xLen"][frame_idx]
            reset_y_len = outputData["yLen"][frame_idx]
        else:
            if "reset_frame" not in locals():
                outputData["samples"][frame_count] = \
                    outputData["samples"][frame_idx]
            else:
                if reset_x_pos != outputData["xPos"][frame_idx] \
                        or reset_y_pos != outputData["yPos"][frame_idx] \
                        or reset_x_len != outputData["xLen"][frame_idx] \
                        or reset_y_len != outputData["yLen"][frame_idx]:
                    outputData["samples"][frame_count] = \
                        outputData["samples"][frame_idx]
                else:
                    outputData["samples"][frame_count] = \
                        reset_frame-outputData["samples"][frame_idx]

                    outputData["samples"][frame_count][
                        outputData["samples"][frame_count] > 32767] = 0

                outputData["xPos"][frame_count] = \
                    outputData["xPos"][frame_idx]
                outputData["yPos"][frame_count] = \
                    outputData["yPos"][frame_idx]
                outputData["xLen"][frame_count] = \
                    outputData["xLen"][frame_idx]
                outputData["yLen"][frame_count] = \
                    outputData["yLen"][frame_idx]
                outputData["timeStampStart"][frame_count] = \
                    outputData["timeStampStart"][frame_idx]
                outputData["timeStampEnd"][frame_count] = \
                    outputData["timeStampEnd"][frame_idx]
                candidate_frame = outputData["samples"][frame_count]
                candidate_frame = (
                    (candidate_frame.astype(
                        np.float32)/candidate_frame.max())*255).astype(
                                np.uint8)
                outputData["samples"][frame_count] = candidate_frame

                frame_count += 1
    outputData["timeStampStart"] = \
        outputData["timeStampStart"][0:frame_count]
    outputData["timeStampEnd"] = \
        outputData["timeStampEnd"][0:frame_count]
    outputData["samples"] = \
        outputData["samples"][0:frame_count]

    del outputData["reset"]

    aps_ts = (outputData["timeStampStart"] + outputData["timeStampEnd"])//2

    #  dvs_tr = find_trigger(events_ts)
    #  aps_tr = find_trigger(aps_ts)
    #
    #  if aps_ts[aps_tr] < events_ts[dvs_tr]:
    #      aps_tr += 1

    return (events_ts/1e6, event_x_addrs, event_y_addrs, event_pols,
            aps_ts/1e6, outputData["samples"])


def load_das_rec(file_path, max_events=30000000, verbose=False):
    """Gets the event timestamps and addresses for a .aedat file.

    # Parameters
    file_path : string
        The absolute path of the AEDAT file for DAS.

    # Returns
    timestamps : numpy.ndarray
        A numpy array that contains timestamps
    addresses : numpy.ndarray
        A numpy array that contains addresses

    """
    assert os.path.isfile(file_path)

    das_file = open(file_path, "rb")

    # get some basic statics of the file and skip header of the file
    num_bytes_per_event = 8
    file_length = get_file_length(file_path)
    header_length = skip_header(das_file)
    if verbose is True:
        print("Header length:", header_length, "bytes")
        print("File length:", file_length, "bytes")
    num_events = (file_length-header_length)//num_bytes_per_event
    if num_events > max_events:
        num_events = max_events
    if verbose is True:
        print("Number of events:", num_events)

    # decode data
    das_events = np.fromfile(
        das_file, dtype='>u4', count=2*num_events).reshape(
            (num_events, 2)).astype(np.uint32)

    if verbose is True:
        print("Number of events:", das_events.shape)

    # close the file
    das_file.close()

    return das_events[:, 1], das_events[:, 0]


def decode_ams1c(timestamps, addresses, return_type=True):
    """Decoding ams1c events."""

    time_wrap_mask = int("80000000", 16)
    adc_event_mask = int("2000", 16)

    address_mask = int("00FC", 16)
    neuron_mask = int("0300", 16)
    ear_mask = int("0002", 16)
    filterbank_mask = int("0001", 16)

    # temporarily remove all StartOfConversion events, no clue what this means!
    soc_idx_1 = int("302C", 16)
    soc_idx_2 = int("302D", 16)
    events_without_start_of_conversion_idx = np.where(
        np.logical_and(addresses != soc_idx_1, addresses != soc_idx_2))
    timestamps = timestamps[events_without_start_of_conversion_idx]
    addresses = addresses[events_without_start_of_conversion_idx]

    # finding cochlea events
    cochlea_events_idx = np.where(
        np.logical_and(
            addresses & time_wrap_mask == 0,
            addresses & adc_event_mask == 0))
    timestamps_cochlea = timestamps[cochlea_events_idx]
    addresses_cochlea = addresses[cochlea_events_idx]

    # shouldn't subtract the first event since time is not monotonic
    #  timestamps_cochlea = timestamps_cochlea - timestamps_cochlea[0]
    timestamps_cochlea = timestamps_cochlea.astype(np.float32)/1e6

    # decoding addresses to get ear id, on off id and the channel id
    channel_id = np.array(
        (addresses_cochlea & address_mask) >> 2, dtype=np.int8)
    ear_id = np.array((addresses_cochlea & ear_mask) >> 1, dtype=np.int8)
    neuron_id = np.array(
        (addresses_cochlea & neuron_mask) >> 8, dtype=np.int8)
    filterbank_id = np.array(
        (addresses_cochlea & filterbank_mask), dtype=np.int8)

    if return_type:
        type_id = get_type_id(
            'ams1c', channel=channel_id, neuron=neuron_id,
            filterbank=filterbank_id)
        return timestamps_cochlea, ear_id, type_id

    return timestamps_cochlea, channel_id, ear_id, neuron_id, filterbank_id


def load_and_decode_ams1c(file_path,
                          max_events=30000000,
                          return_type=True,
                          verbose=False):
    """Load and decode ams1c."""
    timestamps, addresses = load_das_rec(file_path, max_events, verbose)
    #  tr = find_trigger(timestamps)
    #  real_timestamps, real_addresses = timestamps[tr:], addresses[tr:]

    return decode_ams1c(timestamps, addresses, return_type)


def get_type_id(sensor_type, channel=None,
                neuron=None, filterbank=None, on_off=None):
    if sensor_type == 'ams1b' or sensor_type == 'ams1c':
        type_id = channel+NB_CHANNELS*neuron + \
            NB_CHANNELS*NB_NEURON_TYPES*filterbank
        return type_id
    elif sensor_type == 'lp':
        type_id = channel+NB_CHANNELS*on_off
        return type_id
    else:
        raise ValueError("Sensor type is not implemented")


def find_trigger(x):
    dx = 1e-3
    dy = np.diff(x) / dx
    return np.where(dy > 1e10)[0][-1] + 1


def count_windows(timestamps, window_size=0.005, stride_size=0.001):
    """Calculate number of windows from a time stamp series.

    # Arguments
    timestamps: numpy.ndarray
        1-D array that contains timestamps
    window_size: float
        moving windows size in secs
    stride_size: float
        stride size in secs

    # Returns
    num_windows: int
        number of windows in this timestamps array
    max_time: int
        the maximum time in this timestamps array (without zeroing)
    """
    min_time = np.min(timestamps)
    max_time = np.max(timestamps)
    num_windows = int(
        ((max_time-min_time)//0.001+1-window_size*1e3)//(stride_size*1e3)+1)

    return num_windows, min_time, max_time


def compute_das_spikerate(timestamps, channels, num_windows, min_time,
                          window_size=0.005, stride_size=0.001,
                          channels_range=[1, 64]):
    """Compute DAS spikerate.

    Actually a spike count histogram

    Assume the resolution is 1ms

    # Argument
    timestamps: numpy.ndarray
        list of time stamps (stamps in secs)
    channels: numpy.ndarrary
        list of channel ids
    window_size: float
        sliding windows size, in (secs)
    stride_size: float
        stride size, in (secs)
    channels_range: list
        with two numbers that specify the range of the selected channels.
        e.g. [20, 40] selects channels 20 to 40

        The first channel with id=1

    # Returns
    das_spike_rate: numpy.ndarray
        size in (num_channels, num_windows)
    """
    num_channels = 64
    das_spike_count = np.zeros((num_windows, num_channels))
    for t, c in zip(timestamps-min_time, channels):
        das_spike_count[
            int(max(0, (t-window_size)//stride_size+1)):
            int(t//stride_size)+1, c] += 1
    return das_spike_count[:, channels_range[0]-1:channels_range[1]-1].T


def compute_dvs_chunk_spikerate(dvs_spike_rate, timestamps,
                                x_addresses, y_addresses,
                                chunk_id, chunk_size,
                                window_size=0.005, stride_size=0.001):
    """Compute a chunk of DVS spike rate (in-place modification).

    Assume chunk on y-axis only

    # Argument
    dvs_spike_rate: numpy.ndarray
        (num_windows, height, width)
    timestamps: numpy.ndarray
        list of time stamps (stamps in secs)
    x_addresses: numpy.ndarray
        list of x address, assume [0, 239] if DAVIS240
    y_addresses: numpy.ndarray
        list of y address, assume [0, 179] if DAVIS240
    window_size: float
        sliding windows size, in (secs)
    stride_size: float
        stride size, in (secs)
    channels_range: list
        with two numbers that specify the range of the selected channels.
        e.g. [20, 40] selects channels 20 to 40

        The first channel with id=1
    """
    for t, x, y in zip(timestamps, x_addresses, y_addresses):
        dvs_spike_rate[int(max(0, (t-window_size)//stride_size+1)):
                       int(t//stride_size)+1,
                       int(y-chunk_id*chunk_size), x] += 1

    return dvs_spike_rate


def compute_dvs_spikerate(dvs_spike_rate, timestamps,
                          x_addresses, y_addresses,
                          chunk_id, chunk_size,
                          window_size=0.005, stride_size=0.001):
    """Compute the entire DVS spikerate tensor.
    # Argument
    dvs_spike_rate: numpy.ndarray
        (num_windows, height, width)
    timestamps: numpy.ndarray
        list of time stamps (stamps in secs)
    x_addresses: numpy.ndarray
        list of x address, assume [0, 239] if DAVIS240
    y_addresses: numpy.ndarray
        list of y address, assume [0, 179] if DAVIS240
    window_size: float
        sliding windows size, in (secs)
    stride_size: float
        stride size, in (secs)
    channels_range: list
        with two numbers that specify the range of the selected channels.
        e.g. [20, 40] selects channels 20 to 40

        The first channel with id=1
    """
    pass


def compute_events_chunk_corr(events_corr_mat, dvs_spike_rate_chunk,
                              das_spike_rate, chunk_id):
    """Compute a chunk of event correlation (in place modification).

    # Arguments
    events_corr_mat: numpy.ndarray
        event correlation matrix
    dvs_spike_rate_chunk: numpy.ndarray
        a chunk of DVS spike rate
    das_spike_rate: numpy.ndarray
        the DAS spike rate array
    chunk_id: int
        the index of chunk
    """
    # take shape
    chunk_shape = dvs_spike_rate_chunk.shape
    # subtract mean from dvs spike rate chunk
    dvs_spike_rate_chunk -= np.mean(
        dvs_spike_rate_chunk, axis=0, keepdims=True)
    dvs_spike_rate_chunk = dvs_spike_rate_chunk.reshape(
            chunk_shape[0], -1)

    # calculate correlation
    cov_chunk = np.dot(
        das_spike_rate, dvs_spike_rate_chunk).reshape(
            (das_spike_rate.shape[0],)+chunk_shape[1:])

    std_chunk = np.dot(
        np.sqrt(np.sum(
            das_spike_rate**2, axis=1, keepdims=True)),
        np.sqrt(np.sum(
            dvs_spike_rate_chunk**2, axis=0, keepdims=True))).reshape(
                (das_spike_rate.shape[0],)+chunk_shape[1:])+1e-5

    cov_chunk /= std_chunk

    events_corr_mat[
        :, chunk_id*chunk_shape[1]:(chunk_id+1)*chunk_shape[1],
        :] = cov_chunk


def compute_events_corr(davis_timestamps, x_addresses, y_addresses,
                        das_spike_rate, num_windows, min_time, max_time,
                        window_size=0.005, stride_size=0.001, num_chunks=9,
                        height=180, width=240):
    # chunk size along y axis
    y_chunk_size = height//num_chunks

    # select signal
    time_idx = (davis_timestamps <= max_time)
    davis_timestamps = davis_timestamps[time_idx]
    x_addresses = x_addresses[time_idx]
    y_addresses = y_addresses[time_idx]

    # init spike chunk and correlation matrix
    spike_chunk = np.zeros((num_windows, y_chunk_size, width))
    corr_mat = np.zeros((das_spike_rate.shape[0], height, width))

    # subtract per-channel mean from das_spike_rate
    das_spike_rate -= np.mean(das_spike_rate, axis=1, keepdims=True)

    for chunk_id in range(num_chunks):
        # selecting right portion of data
        y_idx = np.logical_and((y_addresses >= chunk_id*y_chunk_size),
                               (y_addresses < (chunk_id+1)*y_chunk_size))
        temp_time = davis_timestamps[y_idx]
        temp_x_addrs = x_addresses[y_idx]
        temp_y_addrs = y_addresses[y_idx]

        # filling up this chunk
        compute_dvs_chunk_spikerate(
            spike_chunk, temp_time-min_time, temp_x_addrs, temp_y_addrs,
            chunk_id, y_chunk_size, window_size=window_size,
            stride_size=stride_size)

        # computing correlation
        compute_events_chunk_corr(
            corr_mat, spike_chunk, das_spike_rate, chunk_id)

        # restore to empty
        spike_chunk = np.zeros((num_windows, y_chunk_size, width))

    return corr_mat
