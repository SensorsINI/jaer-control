"""Control loop for recording.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import argparse

import tkinter as tk
from tkinter.ttk import Progressbar

# global parameters
WIN_WIDTH, WIN_HEIGHT = 1280, 800


class LipreadingRecording(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        #  self.pack()
        self.grid()
        self.create_widgets()

    def configure_grid(self, master, num_rows, num_cols):
        """Configure a grid in a Frame."""
        for r in range(num_rows):
            master.rowconfigure(r, weight=1)
        for c in range(num_cols):
            master.columnconfigure(c, weight=1)

    def create_widgets(self):
        # get grid layout
        self.configure_grid(self.master, 11, 6)

        # text frame
        self.text_frame = tk.Frame(self.master, bg="antique white")
        self.text_frame.grid(row=0, column=0, rowspan=6, columnspan=6,
                             sticky=tk.W+tk.E+tk.N+tk.S)

        # text label
        self.text_label = tk.Label(
            self.text_frame, text="hello",
            font="Helvetica 50 bold",
            bg="antique white")
        self.text_label.place(relx=0.5, rely=0.5, anchor="center")

        # parameter frame
        self.param_frame = tk.Frame(self.master, bg="blue")
        self.param_frame.grid(
            row=6, column=0, rowspan=4, columnspan=3,
            sticky=tk.W+tk.E+tk.N+tk.S)
        self.configure_grid(self.param_frame, 4, 3)
        self.parameter_frame_widgets()

        # button frame
        self.button_frame = tk.Frame(self.master, bg="green")
        self.button_frame.grid(
            row=6, column=3, rowspan=4, columnspan=3,
            sticky=tk.W+tk.E+tk.N+tk.S)
        self.configure_grid(self.button_frame, 4, 3)
        self.button_frame_widgets()

        # progress frame
        self.progress_frame = tk.Frame(self.master, bg="yellow")
        self.progress_frame.grid(
            row=10, column=0, columnspan=6,
            sticky=tk.W+tk.E+tk.N+tk.S)

        # progress bar
        self.progress_bar = Progressbar(
            self.progress_frame, length=100)

    def button_frame_widgets(self):
        # buttons
        self.start_button = tk.Button(self.button_frame)
        self.start_button["text"] = "Start"
        self.start_button["font"] = "Helvetica 20"
        self.start_button.grid(row=0, column=0, columnspan=3)

        self.stop_button = tk.Button(self.button_frame)
        self.stop_button["text"] = "Stop"
        self.stop_button["font"] = "Helvetica 20"
        self.stop_button.grid(row=1, column=0, columnspan=3)

        self.skip_button = tk.Button(self.button_frame)
        self.skip_button["text"] = "Skip"
        self.skip_button["font"] = "Helvetica 20"
        self.skip_button.grid(row=2, column=0, columnspan=3)

        self.training_button = tk.Button(self.button_frame)
        self.training_button["text"] = "Training Session"
        self.training_button["font"] = "Helvetica 20"
        self.training_button.grid(row=3, column=0, columnspan=3)

    def parameter_frame_widgets(self):
        # text boxes
        self.subject_label = tk.Label(
            self.param_frame, text="Subject ID:",
            font="Helvetica 20")
        self.subject_label.grid(row=0, column=0)
        self.subject_text = tk.Entry(self.param_frame, font="Helvetica 20")
        self.subject_text.grid(row=0, column=1, columnspan=2)

        self.trail_label = tk.Label(
            self.param_frame, text="No. Trial(s):",
            font="Helvetica 20")
        self.trail_label.grid(row=1, column=0)
        self.trail_text = tk.Entry(self.param_frame, font="Helvetica 20")
        self.trail_text.grid(row=1, column=1, columnspan=2)

        self.num_sentence_label = tk.Label(
            self.param_frame, text="No. sentence(s):",
            font="Helvetica 20")
        self.num_sentence_label.grid(row=2, column=0)
        self.num_sentence_text = tk.Entry(
            self.param_frame, font="Helvetica 20")
        self.num_sentence_text.grid(row=2, column=1, columnspan=2)

        self.duration_label = tk.Label(
            self.param_frame, text="Duration [secs]",
            font="Helvetica 20")
        self.duration_label.grid(row=3, column=0)
        self.duration_text = tk.Entry(
            self.param_frame, font="Helvetica 20")
        self.duration_text.grid(row=3, column=1, columnspan=2)

    #  def say_hi(self):
    #      print("hi there, everyone!")


root = tk.Tk()
root.title("Lipreading Recording")
root.geometry(str(WIN_WIDTH)+"x"+str(WIN_HEIGHT))

app = LipreadingRecording(master=root)
app.mainloop()
