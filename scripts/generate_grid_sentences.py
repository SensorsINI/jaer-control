"""Generate GRID sentences.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import pickle

command = ["bin", "lay", "place", "set"]
color = ["blue", "green", "red", "white"]
preposition = ["at", "by", "in", "with"]
letter = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
digit = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "zero"]
adverb = ["again", "now", "please", "soon"]

sentences = []

for cmd in command:
    for clr in color:
        for pre in preposition:
            for l in letter:
                for d in digit:
                    for adv in adverb:
                        temp_sen = cmd+" "+clr+" "+pre+" "+l+" "+d+" "+adv
                        sentences.append(temp_sen)

with open("GRID_corpus.pkl", "wb") as f:
    pickle.dump(sentences, f)
    f.close()
