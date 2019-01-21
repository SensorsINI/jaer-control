"""Control loop for recording.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import os
import sys
import time
import subprocess as sp
from copy import deepcopy
import pickle
from random import shuffle, randint

import tkinter as tk
from tkinter.ttk import Progressbar
from tkinter.filedialog import askdirectory

import numpy as np
import sounddevice as sd
import soundfile as sf

from jaercon.controller import jAERController

# global parameters
WIN_WIDTH, WIN_HEIGHT = 1280, 800
MASTER_ROWS, MASTER_COLS = 13, 6
TEXT_ROWS, TEXT_COLS, TEXT_X, TEXT_Y = 1, 2, 0, 0
PARAM_ROWS, PARAM_COLS, PARAM_X, PARAM_Y = 1, 1, 1, 0
BUTTON_ROWS, BUTTON_COLS, BUTTON_X, BUTTON_Y = 1, 1, 1, 1
PROGRESS_COLS, PROGRESS_X, PROGRESS_Y = 2, 2, 0

# color choice
TEXT_BG_COLOR = "#EEEEEE"
PARAM_BG_COLOR = "#FFC107"
BUTTON_BG_COLOR = "#B2FF59"
PROGRESS_BG_COLOR = "#80D8FF"

# font choice
TEXT_FONT = "Helvetica 70 bold"
PARAM_FONT = "Helvetica 15"
BUTTON_FONT = "Helvetica 15"

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

# load all possible sentences and randomly choose them
GRID_CORPUS = pickle.load(
    open(os.path.join("res", "GRID_corpus.pkl"), "rb"))

# load the sound
beep_data, beep_fs = sf.read(
    os.path.join("res", "beep_2s_48k.wav"))
trigger_data, trigger_fs = sf.read(
    os.path.join("res", "trigger_2s_48k.wav"))
beep_data = beep_data[:, np.newaxis]
trigger_data = trigger_data[:, np.newaxis]

total_data = np.append(trigger_data, beep_data, axis=1)


def play_sound():
    sd.play(total_data, beep_fs)
    status = sd.wait()

    return status


def log_sound(filepath):
    print(sys.executable)
    return sp.Popen([sys.executable, "audio_logger.py", filepath])


class LipreadingRecording(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.data_root_dir = None
        self.skip_curr_item = False
        self.gap_wait_counter = 0
        # MAKE SURE IN THIS ORDER!!!
        self.davis_control = jAERController(udp_port=8997)
        self.das_control = jAERController(udp_port=8998)

        # Drawing related
        self.grid()
        self.create_widgets()

    def configure_grid(self, master, num_rows, num_cols, weight=1):
        """Configure a grid in a Frame."""
        for r in range(num_rows):
            master.rowconfigure(r, weight=weight)
        for c in range(num_cols):
            master.columnconfigure(c, weight=weight)

    def create_master_layout(self):
        # rows
        self.master.rowconfigure(0, weight=7)
        self.master.rowconfigure(1, weight=2)
        self.master.rowconfigure(2, weight=1)

        # column
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=2)

    def create_widgets(self):
        # get grid layout
        #  self.configure_grid(self.master, MASTER_ROWS, MASTER_COLS, 1)
        self.create_master_layout()

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
        self.configure_grid(self.param_frame, 5, 2, 1)
        self.parameter_frame_widgets()

        # button frame
        self.button_frame = tk.Frame(self.master, bg=BUTTON_BG_COLOR)
        self.button_frame.grid(
            row=BUTTON_X, column=BUTTON_Y,
            rowspan=BUTTON_ROWS, columnspan=BUTTON_COLS,
            sticky=tk.W+tk.E+tk.N+tk.S)
        self.configure_grid(self.button_frame, 5, 1, 1)
        self.button_frame_widgets()

        # progress frame
        self.progress_frame = tk.Frame(
            self.master, bg=PROGRESS_BG_COLOR, bd=1)
        self.progress_frame.grid(
            row=PROGRESS_X, column=PROGRESS_Y, columnspan=PROGRESS_COLS,
            sticky=tk.W+tk.E+tk.N+tk.S)

        # progress bar
        self.progress_val = tk.IntVar()
        self.progress_val.set(0)
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

        self.select_root_button = tk.Button(self.button_frame)
        self.select_root_button["text"] = "Select Data Root"
        self.select_root_button["font"] = BUTTON_FONT
        self.select_root_button["width"] = BUTTON_LABEL_WIDTH
        self.select_root_button["command"] = self.select_data_root
        self.select_root_button.grid(row=3, column=0, columnspan=3)

        self.stop_button = tk.Button(self.button_frame)
        self.stop_button["text"] = "Quit"
        self.stop_button["font"] = BUTTON_FONT
        self.stop_button["width"] = BUTTON_LABEL_WIDTH
        self.stop_button["command"] = self.stop_button_cmd
        self.stop_button.grid(row=4, column=0, columnspan=3)

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
        self.num_sentence_text.insert(tk.END, "20")
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

        # make a copy of list of sentences in the data root
        self.curr_GRID_CORPUS_path = os.path.join(
            self.data_root_dir, "temp_GRID_corpus.pkl")
        if not os.path.isfile(self.curr_GRID_CORPUS_path):
            with open(self.curr_GRID_CORPUS_path, "wb") as f:
                pickle.dump(GRID_CORPUS, f)

    def start_button_cmd(self):
        self.start_button["text"] = "Trial in place"
        self.start_button["state"] = "disabled"

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

        # setup ground truth folder
        gt_file_path = os.path.join(
            self.current_trial_folder, "gt_sentences.txt")
        gt_file = open(gt_file_path, "a+")

        # start sign
        self.display_text("We are about to start!", "green")
        self.sleep(3)

        # start looping through the sentences
        sentences = self.prepare_text(self.num_sentences)
        for sen_id in range(self.num_sentences):
            # get the text
            self.progress_bar.step(int(100/self.num_sentences))
            curr_sentence = sentences[sen_id]

            # get to control flow
            save_paths = self.record_one_sentence(curr_sentence, sen_id)

            # pause the gap
            do_skip, extra_sleep = self.check_skip_pressed()
            if do_skip is False:
                gt_file.write(curr_sentence+"\n")

            self.remove_last_recording(save_paths, do_skip)
            if extra_sleep != 0:
                self.sleep(extra_sleep)

        # End sign
        gt_file.close()
        self.display_text("This trial ends. Thank you!", "green")
        self.sleep(3)
        self.display_text("Welcome", "black")
        # complete trial
        self.enable_param_text()

        # restore button's text
        self.start_button["text"] = "Start"
        self.start_button["state"] = "normal"

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

    def record_one_sentence(self, text, text_id, training=False):
        # construct the save paths
        if training is False:
            filename_base = text.replace(" ", "_")+"_"+str(text_id)
            davis_save_path = os.path.join(
                self.current_trial_folder, filename_base+"_davis.aedat")
            das_save_path = os.path.join(
                self.current_trial_folder, filename_base+"_das.aedat")
            mic_save_path = os.path.join(
                self.current_trial_folder, filename_base+"_mic.wav")

            # zero time stamps for all windows
            # TODO: make sure zerotimestamps by sync devices
            #  self.davis_control.reset_time(no_wait=True)
            #  self.das_control.reset_time(no_wait=True)
            #  self.sleep(0.2)
            # start logging for all sensors
            play_sound()
            self.davis_control.start_logging(
                davis_save_path, title=None, reset_time=False)
            self.das_control.start_logging(
                das_save_path, title=None, reset_time=False)
            audio_contrl = log_sound(mic_save_path)
            #  self.sleep(0.2)

        # give beep for signal
        # display text and change the text color
        self.display_text(text, color="red")

        # wait for duration long
        self.sleep(self.duration)

        # change the text to None
        self.display_text("------", color="blue")

        if training is False:
            # save all the files
            # close logging
            self.davis_control.stop_logging()
            self.das_control.stop_logging()
            audio_contrl.terminate()

            # TODO quick file checker

            return (davis_save_path, das_save_path, mic_save_path)
        else:
            return ("", "", "")

    def display_text(self, text, color):
        """Change text label's content."""
        self.text_label["text"] = text
        self.text_label["fg"] = color

    def prepare_text(self, num_sentences=20):
        with open(self.curr_GRID_CORPUS_path, "rb") as f:
            curr_GRID_CORPUS = pickle.load(f)
            shuffle(curr_GRID_CORPUS)
            f.close()

        num_avail_sentences = len(curr_GRID_CORPUS)

        start_idx = randint(0, num_avail_sentences-num_sentences)
        end_idx = start_idx+num_sentences

        prepared_text = deepcopy(curr_GRID_CORPUS[start_idx:end_idx])
        del curr_GRID_CORPUS[start_idx:end_idx]

        with open(self.curr_GRID_CORPUS_path, "wb") as f:
            pickle.dump(curr_GRID_CORPUS, f)
            f.close()

        return prepared_text

    def stop_button_cmd(self):
        self.master.destroy()

    def skip_button_cmd(self):
        self.skip_curr_item = True

    def sleep(self, duration):
        """Just sleep for a second."""
        try:
            self.master.update()
            time.sleep(duration)
        except Exception:
            pass

    def check_skip_pressed(self, num_checks=10):
        """Skip if true, False otherwise."""
        sleep_time = self.gap/num_checks
        for check_idx in range(num_checks):
            self.sleep(self.gap/num_checks)

            if self.skip_curr_item is True:
                return True, sleep_time*(num_checks-check_idx+1)
        return False, 0

    def remove_last_recording(self, save_paths, do_remove=False):
        """For cleaning the last recording if skip."""
        if do_remove is True:
            try:
                print("-"*50)
                os.remove(save_paths[0])
                print("[RECORDING MSG] Recording %s removed" % (save_paths[0]))
                os.remove(save_paths[1])
                print("[RECORDING MSG] Recording %s removed" % (save_paths[1]))
                os.remove(save_paths[2])
                print("[RECORDING MSG] Recording %s removed" % (save_paths[2]))
                print("-"*50)
            except OSError:
                pass

            # restore
            self.skip_curr_item = False

    def training_button_cmd(self):
        self.training_button["text"] = "Training in place"
        self.training_button["state"] = "disabled"

        # get all variables in the parameter boxes
        self.subject_id = self.subject_text.get()
        self.trial_id = self.trial_text.get()
        self.num_sentences = int(self.num_sentence_text.get())
        self.duration = float(self.duration_text.get())  # in secs
        self.gap = float(self.gap_text.get())  # in secs

        # freeze the change of the variables
        self.disable_param_text()

        # start sign
        self.display_text("We are about to start!", "green")
        self.sleep(3)

        # start looping through the sentences
        sentences = self.prepare_text(self.num_sentences)
        for sen_id in range(self.num_sentences):
            # get the text
            self.progress_bar.step(int(100/self.num_sentences))
            curr_sentence = sentences[sen_id]

            # get to control flow
            self.record_one_sentence(curr_sentence, sen_id, training=True)

            # pause the gap
            do_skip, extra_sleep = self.check_skip_pressed()
            if extra_sleep != 0:
                self.sleep(extra_sleep)

        # End sign
        self.display_text("This trial ends. Thank you!", "green")
        self.sleep(3)
        self.display_text("Welcome", "black")
        # complete trial
        self.enable_param_text()

        self.training_button["text"] = "Start Training"
        self.training_button["state"] = "normal"

    def select_data_root(self):
        self.data_root_dir = askdirectory()
        print("[RECORDING MSG] The data directory is set to %s"
              % (self.data_root_dir))


root = tk.Tk()
root.title("Lipreading Recording")
root.geometry(str(WIN_WIDTH)+"x"+str(WIN_HEIGHT))

app = LipreadingRecording(master=root)
app.mainloop()
