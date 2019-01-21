"""Examine the use of the trigger."""

from __future__ import print_function, absolute_import

import os

import numpy as np

import sounddevice as sd
import soundfile as sf

beep_data, beep_fs = sf.read(
    os.path.join("res", "beep_2s_48k.wav"))
trigger_data, trigger_fs = sf.read(
    os.path.join("res", "trigger_2s_48k.wav"))
beep_data = beep_data[:, np.newaxis]
trigger_data = trigger_data[:, np.newaxis]

total_data = np.append(trigger_data, beep_data, axis=1)

sd.play(total_data, beep_fs)
status = sd.wait()
