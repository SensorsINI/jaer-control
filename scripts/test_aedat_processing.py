"""Testing AEDAT2 Processing functions.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import os

from jaercon.procaedat import load_and_decode_ams1c, load_and_decode_davis_rec
from jaercon.procaedat import check_davis_rec
import matplotlib.pyplot as plt

# file string
davis_path = os.path.join(
    os.environ["HOME"], "place_blue_with_V_6_now_19_davis.aedat")
das_path = os.path.join(
    os.environ["HOME"], "place_blue_with_V_6_now_19_das.aedat")

#  davis_path = os.path.join(
#      os.environ["HOME"], "lay_blue_by_B_zero_again_9_davis.aedat")
#  das_path = os.path.join(
#      os.environ["HOME"], "lay_blue_by_B_zero_again_9_das.aedat")

# test das recording file
#  timestamps, channel_id, ear_id, neuron_id, filterbank_id = \
#      load_and_decode_ams1c(das_path, return_type=False)

davis_events, ts = check_davis_rec(davis_path, level=2, verbose=True)

plt.figure()
plt.plot(ts)
plt.show()
