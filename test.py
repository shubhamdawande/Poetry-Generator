import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils
from keras.models import load_model
import keras

def make_prediction(pred_text):

    # Initial Setup
    text = (open("./data/shakespear_sonnets.txt").read())
    text=text.lower()

    characters = sorted(list(set((text))))
    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}

    seq_length = 100
    keras.backend.clear_session()
    model = load_model('./model/text_generator_final.h5')

    # Text preprocessing
    pred_text = pred_text.lower()
    #pred_text = pred_text[-seq_length-1:-1]
    seq_text = [char_to_n[char] for char in pred_text]

    # prediction
    string_mapped = seq_text[-seq_length-1:-1]
    full_string = [n_to_char[value] for value in seq_text]
    
    # generating characters
    for i in range(500):
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(characters))

        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in string_mapped]
        full_string.append(n_to_char[pred_index])

        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]

    #combining text
    txt=""
    for char in full_string:
        txt = txt+char
    #txt = "<br>".join(txt.split(","))
    return txt
