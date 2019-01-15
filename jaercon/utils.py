"""Useful tools.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import os


def check_aedat(datafile):
    """Check Common AEDAT errors.

    # Paramters
    datafile: string
        path to the datafile

    # Returns
    flag : bool
        boolean flag that indicates the error of the file
    """
    aerdatafh = open(datafile, 'rb')
    fileinfo = os.stat(datafile)

    flag = True
    # check if empty events
    if (fileinfo.st_size <= 131731):
        # means 0 event wrote or unsuccessful writing
        print("FILE TOO SHORT")
        flag = False

    # check if first byte of data is #
    lt = aerdatafh.readline()
    while lt and lt[0] == "#":
        if str(lt)[:4] == "#End":
            lt = aerdatafh.readline()
            if not str(lt):
                print("FILE NO EVENT WRITTEN")
                flag = False
            elif lt[0] == "#":
                flag = False
            return fileinfo.st_size, flag

        lt = aerdatafh.readline()

    return fileinfo.st_size, flag
