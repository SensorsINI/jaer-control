"""Control loop for recording.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import os
import time

import tkinter as tk
from tkinter.ttk import Progressbar
from tkinter.filedialog import askdirectory

# global parameters
WIN_WIDTH, WIN_HEIGHT = 1280, 800
MASTER_ROWS, MASTER_COLS = 13, 6
TEXT_ROWS, TEXT_COLS, TEXT_X, TEXT_Y = 7, 6, 0, 0
PARAM_ROWS, PARAM_COLS, PARAM_X, PARAM_Y = 5, 3, 7, 0
BUTTON_ROWS, BUTTON_COLS, BUTTON_X, BUTTON_Y = 5, 3, 7, 3
PROGRESS_COLS, PROGRESS_X, PROGRESS_Y = 6, 11, 0

# color choice
TEXT_BG_COLOR = "#EEEEEE"
PARAM_BG_COLOR = "#FFC107"
BUTTON_BG_COLOR = "#B2FF59"
PROGRESS_BG_COLOR = "#80D8FF"

# font choice
TEXT_FONT = "Helvetica 50 bold"
PARAM_FONT = "Helvetica 20"
BUTTON_FONT = "Helvetica 20"

# label dimensions
PARAM_LABEL_WIDTH = 20
BUTTON_LABEL_WIDTH = 20

# GRID corpus
command = ["bin", "lay", "place", "set"]
color = ["blue", "green", "red", "white"]
preposition = ["at", "by", "in", "with"]
letter = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
digit = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "zero"]
adverb = ["again", "now", "please", "soon"]


class LipreadingRecording(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.data_root_dir = None
        #  self.pack()
        self.grid()
        self.create_widgets()

    def configure_grid(self, master, num_rows, num_cols, weight=1):
        """Configure a grid in a Frame."""
        for r in range(num_rows):
            master.rowconfigure(r, weight=weight)
        for c in range(num_cols):
            master.columnconfigure(c, weight=weight)

    def create_widgets(self):
        # get grid layout
        self.configure_grid(self.master, MASTER_ROWS, MASTER_COLS, 1)

        # text frame
        self.text_frame = tk.Frame(self.master, bg=TEXT_BG_COLOR)
        self.text_frame.grid(
            row=TEXT_X, column=TEXT_Y, rowspan=TEXT_ROWS,
            columnspan=TEXT_COLS,
            sticky=tk.W+tk.E+tk.N+tk.S)

        # text label
        self.text_label = tk.Label(
            self.text_frame, text="Welcome",
            font=TEXT_FONT,
            bg=TEXT_BG_COLOR)
        self.text_label.place(relx=0.5, rely=0.5, anchor="center")

        # parameter frame
        self.param_frame = tk.Frame(self.master, bg=PARAM_BG_COLOR)
        self.param_frame.grid(
            row=PARAM_X, column=PARAM_Y,
            rowspan=PARAM_ROWS, columnspan=PARAM_COLS,
            sticky=tk.W+tk.E+tk.N+tk.S)
        self.configure_grid(self.param_frame, PARAM_ROWS, PARAM_COLS, 0)
        self.parameter_frame_widgets()

        # button frame
        self.button_frame = tk.Frame(self.master, bg=BUTTON_BG_COLOR)
        self.button_frame.grid(
            row=BUTTON_X, column=BUTTON_Y,
            rowspan=BUTTON_ROWS, columnspan=BUTTON_COLS,
            sticky=tk.W+tk.E+tk.N+tk.S)
        self.configure_grid(self.button_frame, BUTTON_ROWS, BUTTON_COLS, 0)
        self.button_frame_widgets()

        # progress frame
        self.progress_frame = tk.Frame(
            self.master, bg=PROGRESS_BG_COLOR, bd=1)
        self.progress_frame.grid(
            row=PROGRESS_X, column=PROGRESS_Y, columnspan=PROGRESS_COLS,
            sticky=tk.W+tk.E+tk.N+tk.S)

        # progress bar
        self.progress_val = tk.IntVar()
        self.progress_val.set(40)
        self.progress_bar = Progressbar(
            self.progress_frame, length=100,
            orient="horizontal",
            maximum=100,
            variable=self.progress_val,
            mode="determinate")
        self.progress_bar.pack(fill=tk.BOTH, ipady=5)

    def button_frame_widgets(self):
        # buttons
        self.training_button = tk.Button(self.button_frame)
        self.training_button["text"] = "Training Session"
        self.training_button["font"] = BUTTON_FONT
        self.training_button["width"] = BUTTON_LABEL_WIDTH
        self.training_button["command"] = self.training_button_cmd
        self.training_button.grid(row=0, column=0, columnspan=3)

        self.start_button = tk.Button(self.button_frame)
        self.start_button["text"] = "Start"
        self.start_button["font"] = BUTTON_FONT
        self.start_button["width"] = BUTTON_LABEL_WIDTH
        self.start_button["command"] = self.start_button_cmd
        self.start_button.grid(row=1, column=0, columnspan=3)

        self.skip_button = tk.Button(self.button_frame)
        self.skip_button["text"] = "Skip"
        self.skip_button["font"] = BUTTON_FONT
        self.skip_button["width"] = BUTTON_LABEL_WIDTH
        self.skip_button["command"] = self.skip_button_cmd
        self.skip_button.grid(row=2, column=0, columnspan=3)

        self.stop_button = tk.Button(self.button_frame)
        self.stop_button["text"] = "Stop"
        self.stop_button["font"] = BUTTON_FONT
        self.stop_button["width"] = BUTTON_LABEL_WIDTH
        self.stop_button["command"] = self.stop_button_cmd
        self.stop_button.grid(row=3, column=0, columnspan=3)

        self.select_root_button = tk.Button(self.button_frame)
        self.select_root_button["text"] = "Select Data Root"
        self.select_root_button["font"] = BUTTON_FONT
        self.select_root_button["width"] = BUTTON_LABEL_WIDTH
        self.select_root_button["command"] = self.select_data_root
        self.select_root_button.grid(row=4, column=0, columnspan=3)

    def parameter_frame_widgets(self):
        # text boxes
        self.subject_label = tk.Label(
            self.param_frame, text="Subject ID:",
            font=PARAM_FONT,
            bg=PARAM_BG_COLOR,
            anchor="e",
            width=PARAM_LABEL_WIDTH)
        self.subject_label.grid(row=0, column=0)
        self.subject_text = tk.Entry(
            self.param_frame, font=PARAM_FONT)
        self.subject_text.insert(tk.END, "001")
        self.subject_text.grid(row=0, column=1, columnspan=2)

        self.trial_label = tk.Label(
            self.param_frame, text="Trial ID:",
            font=PARAM_FONT,
            bg=PARAM_BG_COLOR,
            anchor="e",
            width=PARAM_LABEL_WIDTH)
        self.trial_label.grid(row=1, column=0)
        self.trial_text = tk.Entry(self.param_frame, font=PARAM_FONT)
        self.trial_text.insert(tk.END, "1")
        self.trial_text.grid(row=1, column=1, columnspan=2)

        self.num_sentence_label = tk.Label(
            self.param_frame, text="No. sentence(s)/trial:",
            font=PARAM_FONT,
            bg=PARAM_BG_COLOR,
            anchor="e",
            width=PARAM_LABEL_WIDTH)
        self.num_sentence_label.grid(row=2, column=0)
        self.num_sentence_text = tk.Entry(
            self.param_frame, font=PARAM_FONT)
        self.num_sentence_text.insert(tk.END, "50")
        self.num_sentence_text.grid(row=2, column=1, columnspan=2)

        self.duration_label = tk.Label(
            self.param_frame, text="Duration [secs]:",
            font=PARAM_FONT,
            bg=PARAM_BG_COLOR,
            anchor="e",
            width=PARAM_LABEL_WIDTH)
        self.duration_label.grid(row=3, column=0)
        self.duration_text = tk.Entry(
            self.param_frame, font=PARAM_FONT)
        self.duration_text.insert(tk.END, "3")
        self.duration_text.grid(row=3, column=1, columnspan=2)

        self.gap_label = tk.Label(
            self.param_frame, text="Gap [secs]:",
            font=PARAM_FONT,
            bg=PARAM_BG_COLOR,
            anchor="e",
            width=PARAM_LABEL_WIDTH)
        self.gap_label.grid(row=4, column=0)
        self.gap_text = tk.Entry(
            self.param_frame, font=PARAM_FONT)
        self.gap_text.insert(tk.END, "2")
        self.gap_text.grid(row=4, column=1, columnspan=2)

    def validate_data_root(self):
        if self.data_root_dir is None or \
                os.path.isdir(self.data_root_dir) is False:
            self.data_root_dir = os.path.join(
                os.environ["HOME"], "lipreading_data")
            os.makedirs(self.data_root_dir)

    def start_button_cmd(self):
        # get all variables in the parameter boxes
        self.subject_id = self.subject_text.get()
        self.trial_id = self.trial_text.get()
        self.num_sentences = int(self.num_sentence_text.get())
        self.duration = float(self.duration_text.get())  # in secs
        self.gap = float(self.gap_text.get())  # in secs

        # freeze the change of the variables
        self.disable_param_text()

        # construct recording folder
        self.validate_data_root()
        self.current_trial_folder = os.path.join(
            self.data_root_dir, self.subject_id, self.trial_id)
        assert os.path.isdir(self.current_trial_folder) is False
        os.makedirs(self.current_trial_folder)

        # start looping through the sentences
        # TODO
        sentences = self.prepare_text(self.num_sentences)
        for sen_id in range(self.num_sentences):
            # get the text
            curr_sentence = sentences[sen_id]

            # get to control flow
            success = self.record_one_sentence(curr_sentence, sen_id)

            # pause the gap
            time.sleep(self.gap)

        # complete trial
        self.enable_param_text()

    def disable_param_text(self):
        self.subject_text["state"] = "disabled"
        self.trial_text["state"] = "disabled"
        self.num_sentence_text["state"] = "disabled"
        self.duration_text["state"] = "disabled"
        self.gap_text["state"] = "disabled"

    def enable_param_text(self):
        self.subject_text["state"] = "normal"
        self.trial_text["state"] = "normal"
        self.num_sentence_text["state"] = "normal"
        self.duration_text["state"] = "normal"
        self.gap_text["state"] = "normal"

    def record_one_sentence(self, text, text_id):
        print("I'm the control flow for one sentences.")

        # construct the save paths
        filename_base = text.replace(" ", "_")+"_"+str(text_id)
        davis_save_path = os.path.join(
            self.current_trial_folder, filename_base+"_davis.aedat")
        das_save_path = os.path.join(
            self.current_trial_folder, filename_base+"_das.aedat")
        mic_save_path = os.path.join(
            self.current_trial_folder, filename_base+"_mic.wav")

        # zero time stamps for all windows
        # start logging for all sensors

        # give beep for signal
        # display text and change the text color

        # wait for duration long
        time.sleep(self.duration)

        # change the text to None

        # save all the files
        # close logging

        # return true

    def prepare_text(self, num_sentences=50):
        print("I'm preparing {} sentencese".format(num_sentences))

    def stop_button_cmd(self):
        print("Stop button")

    def skip_button_cmd(self):
        print("Skip button")

    def training_button_cmd(self):
        print("Training button")

    def select_data_root(self):
        self.data_root_dir = askdirectory()


root = tk.Tk()
root.title("Lipreading Recording")
root.geometry(str(WIN_WIDTH)+"x"+str(WIN_HEIGHT))

app = LipreadingRecording(master=root)
app.mainloop()
