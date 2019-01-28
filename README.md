# jaer-control
A Python module dedicated to jAER remote control

## On Loading DAVIS

1. The Loading of DAVIS roughly follows [ImportAedatDataVersion1or2.py](https://github.com/inivation/AedatTools/blob/master/PyAedatTools/ImportAedatDataVersion1or2.py).

2. We cannot decide the maximum ADC numbers after the subtraction between
the signal and the reset while decoding APS frames, therefore, we blindly scaled it according to
the maximum value of the current frame.

3. There might be failed APS frames, requires further inspection.

4. The timestamps of the APS frames are determined by the mean of the frame start time and the end time.

## On Loading DAS

1. The loading of DAS roughly follows [es_utils.py](https://github.com/SensorsAudioINI/cochlea_utils/blob/master/cochelp/es_utils.py).

## Contacts

Yuhuang Hu  
Email: yuhuang.hu@ini.uzh.ch
