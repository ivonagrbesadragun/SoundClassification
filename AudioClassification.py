from pydub import AudioSegment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from io import BytesIO
from PIL import Image
import os
import wave
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from sklearn.model_selection import ShuffleSplit
import logging
import pathlib
import librosa
from gammatone import gtgram
from gammatone import filters
import librosa.display
from scipy import signal
from gammatone import fftweight
import cv2
import requests
import tempfile
from urllib.request import urlopen


#p = "C:/Users/Meter/Desktop/tomo_diplomski/ICBHI_final_database/130_1p3_Ll_mc_AKGC417L.wav"
p = "https://github.com/ivonagrbesadragun/AudioFiles/blob/master/101_1b1_Al_sc_Meditron.wav?raw=true"
# raw URL of .wav file on GitHub




def Blackbox (wav_path):
    ispis = []
    window_len = 0.5
    overlap = 0.25
    num_channels=64
    new_model = tf.keras.models.load_model('my_model2.keras')
    klase = {'0':'both', '1':'crackles', '2':'normal', '3':'wheezes'}
    result =dict() #dict u koji ce ici ispis oblika interval - klasa

    a = AudioSegment.from_file(wav_path, format="wav") # milisekunde
    for i in np.arange(0, (len(a)/1000) - window_len, overlap): #u sekundama
        audio, sr = librosa.load(wav_path, offset=i, duration=window_len) # otvorimo prozor 
        k = "[" + str(i) + ", " + str(i+window_len) +"]"  # interval koji ce biti kljuc u dict
        f = "From " + str(i) + "s" #from
        t = " to " + str(i+window_len) + "s: " # to
        center_freqs = filters.erb_space(100, sr/2, num_channels)
        filterbank = filters.make_erb_filters(sr, center_freqs)
        filterbank_output = filters.erb_filterbank(audio, filterbank)
        D = np.abs(filterbank_output)**2
        D = 20*np.log10(D)
        plt.figure(figsize=(4,4))
        plt.imshow(D, aspect='auto', origin='lower', cmap='viridis') 
        plt.ioff()
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

        image_buffer = BytesIO()
        plt.savefig(image_buffer, format='jpg', bbox_inches='tight', pad_inches=0)
        plt.close()
        image_buffer.seek(0)
        image = Image.open(image_buffer) 
        image_array = np.array(image)


        pil_image = Image.fromarray(image_array) #convert numpy array to image
        resized_image = pil_image.resize((128, 128)) # resize
        image_array2 = np.array(resized_image)

        batched_image_tensor = tf.expand_dims(image_array2, axis=0)

        # predikcija
        predictions = new_model.predict(batched_image_tensor, verbose=0)
        predicted_label = tf.argmax(predictions[0]).numpy()
        v = klase[str(predicted_label)]
        #print("Predviđena klasa: " + str(predicted_label) + " (" + v + ")")
        el = "{" + str(f) + str(t)  + str(v) + "}"
        ispis.append(str(el))

        # dodajemo u dictionary
        result [k] = v 


    print("Results: ")
    #for k, v in result.items():
    #    print("Interval: " + str(k) + ", klasa: " + str(v)) 
    print(*ispis)



    
def Classify(p):
    if os.path.isfile(p): # lokalni file na PC
        Blackbox(p)
    else: # raw URL 
        response = requests.get(p)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(response.content)
            temp_filepath = temp_file.name
            
        Blackbox(temp_filepath)


Classify(p)


