# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import scipy
from scipy import stats

all_data = pd.DataFrame()
# Reading all foldes and files
path = "./EMG_data_for_gestures-master/"
folders = [file for file in os.listdir(path) if not file.startswith('.')]

for folder in folders:
    files = [file for file in os.listdir(
        path+folder) if not file.startswith('.')]
    print(folder, files)
    for file in files:
        current_data = pd.read_csv(path+folder+"/"+file, sep='\t')
        all_data = pd.concat([all_data, current_data])

# drop rows that contain nan values
all_data = all_data.dropna()


def extractStatisticalFeatures(x):
    fmean = np.mean(x)
    fstd = np.std(x)
    fmax = np.max(x)
    fmin = np.min(x)
    fpp = fmax-fmin
    fkurtosis = scipy.stats.kurtosis(x)
    zero_crosses = np.nonzero(np.diff(x > 0))[0]
    fzero = zero_crosses.size/len(x)
    frms = np.sqrt(np.mean(np.square(x)))
    fcrest = np.max(np.abs(x))/np.sqrt(np.mean(np.square(x)))
    fenergy=np.sum(np.square(np.abs(x)))
    return fmean, fstd, fmax, fmin, fpp, fkurtosis, fzero, frms, fcrest, fenergy


# feature extraction settings window size and windows hop/stride values
winsize = 1000
winhop = 50


# outputs
flabel = []
fpercent = []
flabel2 = []

data_dict = {
    'ch1': pd.DataFrame(),
    'ch2': pd.DataFrame(),
    'ch3': pd.DataFrame(),
    'ch4': pd.DataFrame(),
    'ch5': pd.DataFrame(),
    'ch6': pd.DataFrame(),
    'ch7': pd.DataFrame(),
    'ch8': pd.DataFrame(),
    }


# time series/ signals processing feature extraction with sliding window
for i in range(0, len(all_data), winhop):
    # load all coulmns excluding ouput label and rows (window sized)
    #selmat=all_data.iloc[i:i+winsize, 1:-1].to_numpy().flatten()
    selch1 = all_data.iloc[i:i+winsize, 1].to_numpy()
    selch2 = all_data.iloc[i:i+winsize, 2].to_numpy()
    selch3 = all_data.iloc[i:i+winsize, 3].to_numpy()
    selch4 = all_data.iloc[i:i+winsize, 4].to_numpy()
    selch5 = all_data.iloc[i:i+winsize, 5].to_numpy()
    selch6 = all_data.iloc[i:i+winsize, 6].to_numpy()
    selch7 = all_data.iloc[i:i+winsize, 7].to_numpy()
    selch8 = all_data.iloc[i:i+winsize, 8].to_numpy()

    # extraction of StatisticalFeatures
    chlist = [selch1, selch2, selch3, selch4, selch5, selch6, selch7, selch8]

    for index, ch in enumerate(chlist):
        # Input features
        fmean = []
        fstd = []
        fmax = []
        fmin = []
        fpp = []
        fkurtosis = []
        fzero = []
        frms = []
        fcrest = []
        fenergy = []

        index += 1
        m, s, ma, mi, pp, k, z, r, c, e = extractStatisticalFeatures(ch)
        # append all features to lists
        fmean.append(m)
        fstd.append(s)
        fmax.append(ma)
        fmin.append(mi),
        fpp.append(pp)
        fzero.append(z)
        fkurtosis.append(k)
        frms.append(r)
        fcrest.append(c)
        fenergy.append(e)

        columns = [f'mean{index}', f'std{index}', f'max{index}', f'min{index}', f'pp{index}',  f'kurtosis{index}',
                   f'zero{index}', f'rms{index}', f'crest{index}', f'energy{index}']
        
        
        
        temp_df = pd.DataFrame(zip(fmean, fstd, fmax, fmin, fpp, fkurtosis, fzero, frms, fcrest, fenergy), columns = columns)
        
        data_dict[f'ch{index}'] = pd.concat([data_dict[f'ch{index}'], temp_df])
        
        data_dict[f'ch{index}'] = data_dict[f'ch{index}'].reset_index(drop = True)
        
    # Label decision-1: most frequent output is the label
    bincountlist = np.bincount(
        all_data.iloc[i:i+winsize, -1].to_numpy(dtype='int64'))
    most_frequent_class = bincountlist.argmax()
    flabel.append(most_frequent_class)

    # Label decision-2: Intention read 2nd most frequent output is the label
    percentage_most_frequent = bincountlist[most_frequent_class]/len(
        all_data.iloc[i:i+winsize, -1].to_numpy(dtype='int64'))
    fpercent.append(percentage_most_frequent)
    # If the percentage of the most frequent label is equals 1.0 so the label-2 is most frequent
    if percentage_most_frequent == 1.0:
        most_frequent_class2 = most_frequent_class
    # If the percentage of the most frequent label less than 1.0 so the label-2 is 2nd most frequent
    else:
        bincountlist[most_frequent_class] = 0
        most_frequent_class2 = bincountlist.argmax()
    flabel2.append(most_frequent_class2)

    # print the current index and outputs of the sliding window loop
    print(i, i+winsize, most_frequent_class,
          percentage_most_frequent, most_frequent_class2)
    
features = list(data_dict.values())

rdf = pd.concat(features , axis=1)
rdf['label'] = flabel
rdf['2ndLabel'] = flabel2
rdf['percent'] = fpercent

rdf.replace([np.inf, -np.inf], np.nan, inplace=True)

rdf.dropna()

count = np.isinf(rdf).values.sum()

rdf = rdf[rdf.label != 0]
rdf = rdf[rdf.label != 7]

# save the features as a new file
rdf.to_csv("___emg_gesture_ws"+str(winsize)+"_hop" + str(winhop)+".csv", index=None, header=True)
