"""Sample Audio controller for example.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import time
import subprocess as sp

process = sp.Popen(["python", "audio_logger.py", "test.wav"])

time.sleep(5)

process.terminate()
