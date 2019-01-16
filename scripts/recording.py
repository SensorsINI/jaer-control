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
MASTER_ROWS, MASTER_COLS = 11, 6
TEXT_ROWS, TEXT_COLS, TEXT_X, TEXT_Y = 6, 6, 0, 0
PARAM_ROWS, PARAM_COLS, PARAM_X, PARAM_Y = 4, 3, 6, 0
BUTTON_ROWS, BUTTON_COLS, BUTTON_X, BUTTON_Y = 4, 3, 6, 3
PROGRESS_COLS, PROGRESS_X, PROGRESS_Y = 6, 10, 0

TEXT_BG_COLOR = "antique white"
PARAM_BG_COLOR = "#FFC107"
BUTTON_BG_COLOR = "#B2FF59"
PROGRESS_BG_COLOR = "#80D8FF"


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
        self.configure_grid(self.master, MASTER_ROWS, MASTER_COLS)

        # text frame
        self.text_frame = tk.Frame(self.master, bg=TEXT_BG_COLOR)
        self.text_frame.grid(
            row=TEXT_X, column=TEXT_Y, rowspan=TEXT_ROWS,
            columnspan=TEXT_COLS,
            sticky=tk.W+tk.E+tk.N+tk.S)

        # text label
        self.text_label = tk.Label(
            self.text_frame, text="Welcome",
            font="Helvetica 50 bold",
            bg=TEXT_BG_COLOR)
        self.text_label.place(relx=0.5, rely=0.5, anchor="center")

        # parameter frame
        self.param_frame = tk.Frame(self.master, bg=PARAM_BG_COLOR)
        self.param_frame.grid(
            row=PARAM_X, column=PARAM_Y,
            rowspan=PARAM_ROWS, columnspan=PARAM_COLS,
            sticky=tk.W+tk.E+tk.N+tk.S)
        self.configure_grid(self.param_frame, PARAM_ROWS, PARAM_COLS)
        self.parameter_frame_widgets()

        # button frame
        self.button_frame = tk.Frame(self.master, bg=BUTTON_BG_COLOR)
        self.button_frame.grid(
            row=BUTTON_X, column=BUTTON_Y,
            rowspan=BUTTON_ROWS, columnspan=BUTTON_COLS,
            sticky=tk.W+tk.E+tk.N+tk.S)
        self.configure_grid(self.button_frame, BUTTON_ROWS, BUTTON_COLS)
        self.button_frame_widgets()

        # progress frame
        self.progress_frame = tk.Frame(
            self.master, bg=PROGRESS_BG_COLOR, bd=1)
        self.progress_frame.grid(
            row=PROGRESS_X, column=PROGRESS_Y, columnspan=PROGRESS_COLS,
            sticky=tk.W+tk.E+tk.N+tk.S)

        # progress bar
        self.progress_bar = Progressbar(
            self.progress_frame, length=100)

    def button_frame_widgets(self):
        # buttons
        self.start_button = tk.Button(self.button_frame)
        self.start_button["text"] = "Start"
        self.start_button["font"] = "Helvetica 20"
        self.start_button["command"] = self.start_button_cmd
        self.start_button.grid(row=0, column=0, columnspan=3)

        self.stop_button = tk.Button(self.button_frame)
        self.stop_button["text"] = "Stop"
        self.stop_button["font"] = "Helvetica 20"
        self.stop_button["command"] = self.stop_button_cmd
        self.stop_button.grid(row=1, column=0, columnspan=3)

        self.skip_button = tk.Button(self.button_frame)
        self.skip_button["text"] = "Skip"
        self.skip_button["font"] = "Helvetica 20"
        self.skip_button["command"] = self.skip_button_cmd
        self.skip_button.grid(row=2, column=0, columnspan=3)

        self.training_button = tk.Button(self.button_frame)
        self.training_button["text"] = "Training Session"
        self.training_button["font"] = "Helvetica 20"
        self.training_button["command"] = self.training_button_cmd
        self.training_button.grid(row=3, column=0, columnspan=3)

    def parameter_frame_widgets(self):
        # text boxes
        self.subject_label = tk.Label(
            self.param_frame, text="Subject ID:",
            font="Helvetica 20",
            bg=PARAM_BG_COLOR)
        self.subject_label.grid(row=0, column=0)
        self.subject_text = tk.Entry(self.param_frame, font="Helvetica 20")
        self.subject_text.grid(row=0, column=1, columnspan=2)

        self.trail_label = tk.Label(
            self.param_frame, text="No. Trial(s):",
            font="Helvetica 20",
            bg=PARAM_BG_COLOR)
        self.trail_label.grid(row=1, column=0)
        self.trail_text = tk.Entry(self.param_frame, font="Helvetica 20")
        self.trail_text.grid(row=1, column=1, columnspan=2)

        self.num_sentence_label = tk.Label(
            self.param_frame, text="No. sentence(s):",
            font="Helvetica 20",
            bg=PARAM_BG_COLOR)
        self.num_sentence_label.grid(row=2, column=0)
        self.num_sentence_text = tk.Entry(
            self.param_frame, font="Helvetica 20")
        self.num_sentence_text.grid(row=2, column=1, columnspan=2)

        self.duration_label = tk.Label(
            self.param_frame, text="Duration [secs]:",
            font="Helvetica 20",
            bg=PARAM_BG_COLOR)
        self.duration_label.grid(row=3, column=0)
        self.duration_text = tk.Entry(
            self.param_frame, font="Helvetica 20")
        self.duration_text.grid(row=3, column=1, columnspan=2)

    def start_button_cmd(self):
        print("Start button")

    def stop_button_cmd(self):
        print("Stop button")

    def skip_button_cmd(self):
        print("Skip button")

    def training_button_cmd(self):
        print("Training button")

    #  def say_hi(self):
    #      print("hi there, everyone!")


root = tk.Tk()
root.title("Lipreading Recording")
root.geometry(str(WIN_WIDTH)+"x"+str(WIN_HEIGHT))

app = LipreadingRecording(master=root)
app.mainloop()
