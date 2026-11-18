from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D,MaxPool1D, Flatten, Input
from keras.regularizers import l2,l1, l1_l2
from keras.models import load_model
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping

import alf.io
from oneibl.one import ONE
from brainbox.processing import bincount2D
import brainbox.behavior.wheel as wh
from brainbox.io.one import load_channel_locations

from pebble import concurrent
from datetime import datetime
import numpy as np
import pandas as pd
import os 
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from random import shuffle
plt.ion()

'''
test R2=0.47
eid = '7bee9f09-a238-42cf-b499-f51f765c6ded' 
probe = 'probe00'
reg = 'CP'
T_BIN = 0.02
W1 = 0  # number of time bins that neural preceeds wheel
W2 = 10  # number of time bins of neural activity to predict wheel
'''

T_BIN = 0.02


'''
Data preprocessing
'''

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx  

def get_full_stim_intervals(eid):

    '''
    getting trial numbers, stim present interval
    '''
     
    one = ONE()
    trials = one.load_object(eid, 'trials')
    d = {} # dictionary, trial number and still interval    
    
    for tr in range(len(trials['intervals']))[1:]: #skip first trial

        b = trials['goCue_times'][tr]
        c = trials['feedback_times'][tr]
        
        if np.isnan(b):
            #print(f'goCue_times time is nan for trial {tr} and eid {eid}')
            continue          
        
        
        if np.isnan(c):
            #print(f'feedback time is nan for trial {tr} and eid {eid}')
            continue        
                    
        if np.isnan(trials['contrastLeft'][tr]):
            cont = trials['contrastRight'][tr]            
            side = 0
        else:   
            cont = trials['contrastLeft'][tr]         
            side = 1              
                
        d[tr] = [b,c-b, cont, side]
    print(f"cut {len(d)} full trials segments out of {len(trials['intervals'])}")
    return d



def get_acronyms_per_insertion(eid, probe):

    T_BIN = 0.5
    one = ONE()
    dataset_types = ['spikes.times',
                     'spikes.clusters',
                     'clusters.channels']                     
                     
    D = one.load(eid, dataset_types = dataset_types, dclass_output=True)
    local_path = one.path_from_eid(eid)  
    alf_path = local_path / 'alf'     
    probe_path = alf_path / probe                    
                     
    # bin spikes
    spikes = alf.io.load_object(probe_path, 'spikes')
    R, times, Clusters = bincount2D(
        spikes['times'], spikes['clusters'], T_BIN)                     
                     
                     
    # Get for each cluster the location acronym
    clusters = alf.io.load_object(probe_path, 'clusters')        
    cluster_chans = clusters['channels'][Clusters]
    els = load_channel_locations(eid, one=one)   
    acronyms = els[probe]['acronym'][cluster_chans]

    return acronyms


def get_binned_spikes_acronyms_wheel(eid, probe, T_BIN):

    one = ONE()
    # probe = 'probe01'
 
    dataset_types = ['spikes.times',
                     'spikes.depths',
                     'spikes.clusters',
                     'clusters.channels',
                     'wheel.position',
                     'wheel.timestamps',
                     'trials.intervals']
    
    D = one.load(eid, dataset_types = dataset_types, dclass_output=True)
    local_path = one.path_from_eid(eid)  
    alf_path = local_path / 'alf'     
    probe_path = alf_path / probe

    spikes = alf.io.load_object(probe_path, 'spikes')
  
    #T_BIN = 0.02  # time bin in sec

    # bin spikes
    R, times, Clusters = bincount2D(
        spikes['times'], spikes['clusters'], T_BIN)
   
    # Get for each cluster the location acronym
    clusters = alf.io.load_object(probe_path, 'clusters')        
    cluster_chans = clusters['channels'][Clusters]
    els = load_channel_locations(eid, one=one)   
    acronyms = els[probe]['acronym'][cluster_chans]
 


    # output R is n_samples x n_clusters, as needed for pca
    wheel = one.load_object(eid, 'wheel')
    pos, t = wh.interpolate_position(wheel.timestamps, wheel.position, freq=1/T_BIN)

    # align wheel to spikes
    
    trials = alf.io.load_object(alf_path, 'trials') # start at first trial
    t_start = trials['intervals'][0][0] 
    start_idx = find_nearest(times,t_start)
    
    D = R.T[start_idx:,:] #skip first bin
    times = times[start_idx:]

    wh_idx = find_nearest(t,times[0])
    pos = pos[wh_idx:]
    t = t[wh_idx:]
    
    cut = min(len(t),len(times))
    D = D[:cut,:]
    times = times[:cut] 
    pos = pos[:cut]  
    t = t[:cut]          
    
    assert (abs(t[0] - times[0]) < 1) and (abs(t[-1] - times[-1]) < 1), 'Temporal alignment above 1 sec!!'    

    return D, acronyms, times, pos



def pair_neural_wheel_vel(eid, probe, T_BIN, reg, W1, W2, direction):

    dat, acronyms, times, pos = get_binned_spikes_acronyms_wheel(eid, probe, T_BIN)
    v = np.append(np.diff(pos),np.diff(pos)[-1]) # wheel velocity
    d = get_full_stim_intervals(eid)

    if reg not in acronyms:
        print(f'region {reg} is not in acronyms')
        return

    m_ask = acronyms == reg
    data = dat[:,m_ask]
    obs, neu = data.shape

    print(sum(m_ask), 'clusters in reg', reg)

    neu = []
    whe = []
    times_ = []
    
    for i in d:
    
        # cut wheel velocities within trials
        start_idx = find_nearest(times,d[i][0])
        end_idx = start_idx + int(d[i][1]/T_BIN)   
        if end_idx < start_idx:
            print(i,'end_idx < start_idx') 
            return          
        whe.append(v[start_idx:end_idx])
        
        # cut neural activity
        if direction == 'preceed':  # preceeding neural info
            start_idx = start_idx - (W1 + W2)  
            end_idx = end_idx - (W1)
            
        elif direction == 'succeed':   # succeeding neural info
            start_idx = start_idx + abs(W1)
            end_idx = end_idx + (abs(W1) + W2)     
                       
        else:
            print(direction, 'must be preceed/succeed')
            return                           
        neu.append(data[start_idx:end_idx])
        times_.append(times[start_idx:end_idx])


    return neu, whe, times_


def create_XY(neu, whe, W2):
    '''
    for each wheel velocity, get
    preceding neural activity
    as a concatenated vector
    '''          
   
    x = []  # concatenated neural activity per wheel velocities 
    y = []  # wheel velocities
    for tr in range(len(neu)):
        for i in range(len(whe[tr])):
            y.append(whe[tr][i])
            xx = neu[tr][i: i + W2]                 
            x.append(np.concatenate(xx.T))
                    
    return np.array(x), np.array(y)         


def get_data(eid, probe, T_BIN, reg, W1, W2, direction):

    neu, whe, times_ = pair_neural_wheel_vel(eid, 
                                             probe, 
                                             T_BIN, 
                                             reg, 
                                             W1, W2, 
                                             direction)
    x,y = create_XY(neu, whe, W2)
    
    return x,y


'''
decoding
'''


@concurrent.process
def NN(x,y):
   
    nodes = 20  # 100
    nodes1 = 10  # 50
    EPOCHS = 25  # 15

    startTime=datetime.now()

    print('x.shape:', x.shape, 'y.shape:', y.shape, 'Epochs:', EPOCHS) 


    sc = StandardScaler() 
    x = sc.fit_transform(x) 

    sc0 = StandardScaler() 
    y = sc0.fit_transform(y.reshape(-1, 1)) 


    input_dim = len(x[0])

    k=1

    train_r2 = []
    test_r2 = []

    folds = 6
    kf = KFold(n_splits=folds, shuffle=True)

    rm = optimizers.RMSprop(lr=0.01)
    callback = EarlyStopping(monitor='loss', patience=3, min_delta=0.02)
    
    for train_index, test_index in kf.split(x):

        train_X = x[train_index]
        test_X = x[test_index]
        train_y = y[train_index]
        test_y = y[test_index]

        model = Sequential()   

        model.add(Dense(nodes,activation='relu',input_dim=input_dim))
        model.add(Dense(nodes1,activation='relu',kernel_regularizer=l2(0.001)))
        model.add(Dense(len(y[0]), input_dim=input_dim,activation='linear')) 

        model.compile(optimizer=rm,loss='mse') 

        if k==1: 
            print(model.summary())
            print('train/test samples:', len(train_y), len(test_y))
            print('')
        
        model.fit(train_X, train_y,
                  batch_size=80,
                  epochs=EPOCHS,
                  verbose=1,
                  callbacks=[callback])
        
        # test performance for this fold
        y_pred_test = model.predict_on_batch(test_X) 
        y_pred_train = model.predict_on_batch(train_X)
        
        test_y = np.reshape(test_y,(len(test_y),))
        y_pred_test = np.reshape(y_pred_test,(len(y_pred_test),))            
        r2_test = r2_score(test_y,y_pred_test) 
               
        train_y = np.reshape(train_y,(len(train_y),))
        y_pred_train = np.reshape(y_pred_train,(len(y_pred_train),))        
        r2_train = r2_score(train_y,y_pred_train)        
      
        train_r2.append(r2_train)
        test_r2.append(r2_test)    
        
        print(f'fold {k} of {folds}')
        print('Train / test r2: ',
               np.round(r2_train,3),
               np.round(r2_test,3))

        
        k+=1 
 
    tr_r2 = np.round(np.mean(train_r2),2) 
    te_r2 = np.round(np.mean(test_r2),2)      
    print('')    
    print('mean train/test r2: ', tr_r2, te_r2)  
    print(f'time to compute:', datetime.now() - startTime) 
#    print('train r2 per fold: ', [np.round(x,2) for x in train_r2])    
#    print('test r2 per fold: ', [np.round(x,2) for x in test_r2])    

    return tr_r2, te_r2


def W1_W2_grid_search():

    eid = '7bee9f09-a238-42cf-b499-f51f765c6ded' 
    probe = 'probe00'
    reg = 'CP'
    
    W1p = np.array([2**n for n in range(8)])
    W1s = np.concatenate([[0],W1p])
    
    res = []
    for W2 in [1,2,4,10]:
        for direction in ['preceed','succeed']:
            for W1 in W1s:
                x,y = get_data(eid, probe, T_BIN, reg, W1, W2, direction)
                print(direction, 'W1:',W1, 
                      'W2:',W2,'x.shape',x.shape,
                      'y.shape',y.shape)
                future = NN(x,y)
                tr_r2, te_r2 = future.result()
                print(W1,W2,tr_r2, te_r2)
                res.append([W1,W2,tr_r2, te_r2])

    print(res)
    np.save('res2.npy',res)
    return res



def plot_res():

    '''
    plot influence of W1, W2 on r2
    as heatmaps
    '''

    res = np.load('/home/mic/res2.npy')


    i = 0
    j = 0
    
    W1p = np.array([2**n for n in range(8)])
    W1s = np.concatenate([[0],W1p])
    #W1s = [0,2,4,8,16,32,64,128]
    W2s = [1,2,4,10]
    
    M_train = np.zeros((len(W1s)*2,len(W2s)))
    M_test = np.zeros((len(W1s)*2,len(W2s)))   
       
    k = 0
    i = 0  
    for W2 in range(len(W2s)):  
        j = 0         
        for direction in range(2):
            for W1 in range(len(W1s)): 
                M_train[j,i] = res[k][2]
                M_test[j,i] = res[k][3]
                j+=1
                k+=1 
        i+=1    
            
    M_train_flat = [item for sublist in M_train for item in sublist]
    M_test_flat = [item for sublist in M_test for item in sublist]    
    M_flat = np.concatenate([M_train_flat,M_test_flat])    
    Vmin, Vmax = min(M_flat), max(M_flat) 

    M_train = M_train.T
    M_test = M_test.T

    # rearrange for plotting
    Ms = [] 
    for M in [M_train, M_test]: 
        Mpre = M[:,np.array(list(reversed(range(len(W1s)))))]
        Ms.append(np.hstack([Mpre,M[:,9:]]))
    
    M_train = Ms[0]
    M_test = Ms[1]

    #return M_train, M_test
    fig = plt.figure(figsize=(8,10))
    cm = plt.get_cmap("Reds")
    ax0 = plt.subplot(2,1,1)  
      
    ax0.set_xticks(range(M_train.shape[1]))    
    ylabs = np.concatenate([list(reversed(np.array(W1s)*T_BIN)),
                            np.array(W1s)*T_BIN])     
    ax0.set_xticklabels(ylabs)    
    ax0.set_xlabel('W1: lag between neural info and wheel speed [sec] \n'
                   '<--- preceedeing; succeeding --->') 

    
    ax0.set_yticks(range(len(W2s)))
    ax0.set_yticklabels(np.array(W2s)*T_BIN)    
    ax0.set_ylabel('W2: duration of neural info for decoding [sec]')    

    ax0.set_title('train r2')
    p0 = plt.imshow(M_train, cmap= cm, vmin= Vmin, vmax=Vmax)
    fig.colorbar(p0)     
    plt.axvline(x=8.5)
    
    
    ax1 = plt.subplot(2,1,2,sharex = ax0, sharey = ax0) 
    ax1.set_xlabel('W1: lag between neural info and wheel speed [sec] \n'
                   '<--- preceedeing; succeeding --->') 
    ax1.set_ylabel('W2: duration of neural info for decoding [sec]') 
    
    ax1.set_title('test r2')    
    p1 = plt.imshow(M_test, cmap= cm, vmin= Vmin, vmax=Vmax)
    plt.axvline(x=8.5)
    fig.colorbar(p1) 
    plt.tight_layout()
    plt.suptitle('7bee9f09-a238-42cf-b499-f51f765c6ded, probe00 \n CP, T_BIN = 0.02 sec, 6-fold cross validation, goCue to feedbackTime \n') 

    




