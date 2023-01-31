# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import time
import pickle
import sys
import matplotlib.pyplot as plt


def extractStatisticalFeatures(x):
    fstd=np.std(x)
    fmax=np.max(x)
    fmin=np.min(x)
    fpp=fmax-fmin
    zero_crosses = np.nonzero(np.diff(x > 0))[0]
    fzero=zero_crosses.size/len(x)
    frms = np.sqrt(np.mean(np.square(x)))
    return fstd, fmin, fpp, fzero, frms 

path = "./EMG_data_for_gestures-master/"
folders = [file for file in os.listdir(path) if not file.startswith('.')]

all_data = pd.DataFrame()

for folder in folders:
    files = [file for file in os.listdir(path+folder) if not file.startswith('.')]
    print (folder, files)
    for file in files:
        current_data = pd.read_csv(path+folder+"/"+file,sep='\t')  
        all_data = pd.concat([all_data,current_data])

all_data=all_data.dropna()


winsize=1000
winhop=50

pickled_model = pickle.load(open('selectedModel.pkcls', 'rb'))

all_data.rename(columns={'class': 'class_value'}, inplace=True)
all_data = all_data[all_data.class_value != 7]
all_data = all_data[all_data.class_value != 0]


ch1_raw = all_data['channel1']
ch2_raw = all_data['channel2']
ch3_raw = all_data['channel3']
ch4_raw = all_data['channel4']
ch5_raw = all_data['channel5']
ch6_raw = all_data['channel6']
ch7_raw = all_data['channel7']
ch8_raw = all_data['channel8']
class_value = all_data['class_value']

i = 0
df_dict = {}
time = np.arange(ch8_raw.size) / winsize
prediction = ""

def on_press(event):
    global i
    print('press', event.key)
    sys.stdout.flush()

    lower = i
    upper = i + winsize

    ax1.cla()

    ax1.plot(time[lower:upper], ch1_raw[lower:upper])
    ax1.plot(time[lower:upper], ch2_raw[lower:upper])
    ax1.plot(time[lower:upper], ch3_raw[lower:upper])
    ax1.plot(time[lower:upper], ch4_raw[lower:upper])
    ax1.plot(time[lower:upper], ch5_raw[lower:upper])
    ax1.plot(time[lower:upper], ch6_raw[lower:upper])
    ax1.plot(time[lower:upper], ch7_raw[lower:upper])
    ax1.plot(time[lower:upper], ch8_raw[lower:upper])
    
    bincountlist=np.bincount(class_value[lower:upper])
    most_frequent_class=bincountlist.argmax()
    
    ax1.grid()
    ax1.set_title(f'Real Value: {most_frequent_class}')


    zero8 = (np.nonzero(np.diff(ch8_raw.iloc[lower:upper] > 0))[0]).size/len(ch8_raw.iloc[lower:upper])
    pp4 = np.max(ch4_raw.iloc[lower:upper]) - np.min(ch4_raw.iloc[lower:upper])
    zero7 = (np.nonzero(np.diff(ch7_raw.iloc[lower:upper] > 0))[0]).size/len(ch7_raw.iloc[lower:upper])
    min4 = np.min(ch4_raw.iloc[lower:upper])
    rms5 = np.sqrt(np.mean(np.square(ch4_raw.iloc[lower:upper])))
    

    df_dict["zero8"] = zero8
    df_dict["pp4"] = pp4
    df_dict["zero7"] = zero7
    df_dict["min4"] = min4
    df_dict["rms5"] = rms5

    df = pd.DataFrame(df_dict, index = [0])
    global prediction

    prediction=pickled_model.predict(df)
    bincountlist=np.bincount(prediction)

    most_frequent_prediction_class=bincountlist.argmax()

    ax2.cla()
    ax2.plot(['zero8'],[zero8], 'o')
    ax2.plot(['pp4'],[pp4], 'o')
    ax2.plot(['zero7'],[zero7], 'o')
    ax2.plot(['min4'],[min4], 'o')
    ax2.plot(['rms5'],[rms5], 'o')

    ax2.grid()
    ax2.set_title(f'Prediction Value: {most_frequent_prediction_class}')

    if event.key == 'right':
        i = i+winhop
        fig.canvas.draw()
    elif event.key == 'left':
        i = i-winhop
        fig.canvas.draw()
        
        
fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.grid()
ax1.set_title('Raw Signal')

ax2 = fig.add_subplot(212)
ax2.grid()

fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()
