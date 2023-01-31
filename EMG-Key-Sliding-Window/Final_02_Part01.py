# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import os
import pandas as pd
import numpy as np
import scipy
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import pickle


def extractStatisticalfeatures(x):
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


fstd=[]
fmin=[]
fpp=[]
fzero=[]
frms=[]
flabel=[]

ch1mean=[]
ch2mean=[]
ch3mean=[]
ch4mean=[]
ch5mean=[]
ch6mean=[]
ch7mean=[]
ch8mean=[]

fpercent=[]
flabel2=[]
for i in range(0,len(all_data),winhop):
    selmat=all_data.iloc[i:i+winsize, 1:9].to_numpy().flatten()
    
    s,mi,pp,z,r = extractStatisticalfeatures(selmat) 
    fstd.append(s)
    fmin.append(mi),
    fpp.append(pp)
    fzero.append(z)
    frms.append(r)
    
    ch1mean.append(all_data.iloc[i:i+winsize,1].mean())
    ch2mean.append(all_data.iloc[i:i+winsize,2].mean())
    ch3mean.append(all_data.iloc[i:i+winsize,3].mean())
    ch4mean.append(all_data.iloc[i:i+winsize,4].mean())
    ch5mean.append(all_data.iloc[i:i+winsize,5].mean())
    ch6mean.append(all_data.iloc[i:i+winsize,6].mean())
    ch7mean.append(all_data.iloc[i:i+winsize,7].mean())
    ch8mean.append(all_data.iloc[i:i+winsize,8].mean())
    
    bincountlist=np.bincount(all_data.iloc[i:i+winsize,-1].to_numpy(dtype='int64'))
    most_frequent_class=bincountlist.argmax()
    flabel.append(most_frequent_class)
    
    percentage_most_frequent=bincountlist[most_frequent_class]/len(all_data.iloc[i:i+winsize,-1].to_numpy(dtype='int64'))
    fpercent.append(percentage_most_frequent)
    
    if percentage_most_frequent==1.0:
        most_frequent_class2=most_frequent_class
    else:
        bincountlist[most_frequent_class]= 0
        most_frequent_class2=bincountlist.argmax()
        
    flabel2.append(most_frequent_class2)
   
rdf = pd.DataFrame(
   {'ch1mean': ch1mean,
    'ch2mean': ch2mean,
    'ch3mean': ch3mean,
    'ch4mean': ch4mean,
    'ch5mean': ch5mean,
    'ch6mean': ch6mean,
    'ch7mean': ch7mean,
    'ch8mean': ch8mean,
    'std': fstd,
    'min': fmin,
    'peak-to-peak':fpp,
    'zerocross':fzero,
    'rms':frms,
    'label':flabel,
    'percent':fpercent,
    '2ndlabel':flabel2
    
})

rdf = rdf[rdf.label != 0]
rdf = rdf[rdf.label != 7]


rdf.to_csv("___emg_gesture_ws"+str(winsize)+"_hop"+str(winhop)+".csv", index = None, header=True)

rdf = pd.read_csv('___emg_gesture_ws1000_hop50.csv')

X=rdf.iloc[:,:-3]
y=rdf.iloc[:,-3]

#Selecting Features
X=rdf[["zero8","pp4","zero7","min4","rms5"]]
y=rdf["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the models
rf = RandomForestClassifier()
ada = AdaBoostClassifier()
gbc = GradientBoostingClassifier()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
mlp = MLPClassifier()

# Create a list of the models
models = [rf, ada, gbc, dt, knn, mlp]

results = {}

# Iterate over the models and perform 10-fold cross-validation
for model in models:
    scores = cross_val_score(model, X_train, y_train, cv=5)
    results[model] = scores

accscores=[]
for model in models:
    accscores.append(np.mean(results[model]))
    print(model, accscores[-1])

plt.figure()
plt.bar(range(len(accscores)), accscores, tick_label=models)
plt.grid()
plt.legend()
plt.show()
    
# Find the best performing model
best_model = max(results, key=lambda x: np.mean(results[x]))
clf = best_model
clf.fit(X_train, y_train)

# Save the best model to a pickle file
with open("selectedModel.pkcls", "wb") as f:
    pickle.dump(clf, f)

