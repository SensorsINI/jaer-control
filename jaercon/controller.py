"""An jAER remote controller.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import os
import time
import socket

from jaercon.utils import check_aedat


class jAERController(object):

    def __init__(self, udp_port=8997, remote_ip="localhost", bufsize=1024,
                 wait_time=0.3):
        """The central class to the controller.

        # Parameters
        udp_port: int
            The UDP port that the jAER window connected to.
            default: 8997
        remote_ip: str
            The remote IP address of the jAER running machine.
            default: "localhost"
        bufsize: int
            The buffer size.
            default: 1024
        wait_time: float
            The wait time in secs.
        """
        self.udp_port = udp_port
        self.remote_ip = remote_ip
        self.bufsize = bufsize
        self.address = (self.remote_ip, udp_port)

        self.wait_time = wait_time

        # open socket connection
        self.establish_connection()

    def establish_connection(self):
        """Open the socket connection."""
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bind(("", 0))
        print("The socket is established.")

    def close_connection(self):
        """Close socket connection."""
        try:
            self.conn.close()
            print("The socket attached to {}:{} is closed".format(
                self.remote_ip, self.udp_port))
        except TypeError:
            raise

    def reset_time(self):
        """Reset timestamps."""
        try:
            self.conn.sendto("zerotimestamps", self.address)
            time.sleep(self.wait_time)
        except TypeError:
            raise

    def start_logging(self, save_path, title, reset_time=True):
        """Start logging.

        # Parameters
        save_path: str
            the folder that saves the recordings.
        title: str
            The title of the recording
        reset_time: bool
            Reset timestamps to 0 if True
            False otherwise
            default: True

        # Returns
        rec_path: str
            The full path of the saved recording if successful.
        """
        try:
            if reset_time is True:
                self.reset_time()
            rec_path = os.path.join(save_path, title)
            line = "startlogging "+rec_path

            self.conn.sendto(line, self.address)

            data, fromaddr = self.conn.recvfrom(self.bufsize)
            print("Client received %r from %r" % (data, fromaddr))

            return rec_path
        except TypeError:
            raise

    def stop_logging(self):
        """Stop logging."""
        try:
            self.conn.sendto("stoplogging", self.address)
            data, fromaddr = self.conn.recvfrom(self.bufsize)

            print("Client received %r form %r" % (data, fromaddr))
        except TypeError:
            raise

    def logging(self, save_path, title, duration, reset_time=True):
        """A logging routine.

        # Parameters
        save_path: str
            the folder that saves the recordings.
        title: str
            The title of the recording
        reset_time: bool
            Reset timestamps to 0 if True
            False otherwise
            default: True
        """
        rec_path = self.start_logging(save_path, title, reset_time)
        time.sleep(duration)
        self.stop_logging()

        return check_aedat(rec_path+".aedat")
