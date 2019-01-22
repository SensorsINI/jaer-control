"""Testing sound device."""

from __future__ import print_function, absolute_import

import sounddevice as sd
import numpy as np


def play_sound(data, fs):
    sd.play(data, fs)
    status = sd.wait()

    return status


#  fs = 48000
#  sound_1 = np.ones((fs,), dtype=np.float64)*1000
#
#  play_sound(sound_1, fs)

fs = 44100
sound_2 = np.ones((fs*2,), dtype=np.float64)*1000

play_sound(sound_2, fs)
