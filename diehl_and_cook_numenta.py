
# coding: utf-8

# In[1]:


import torch
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import poisson
import time as T
plt.rcParams["figure.figsize"] = (20, 20)
lfname = 'logs/' + str(int(T.time())) + '_numenta.txt'


# In[2]:


time = 500
plot = False
network = DiehlAndCook2015(5000, dt=1.0, norm=48.95, inh=3)
exc_monitor = Monitor(network.layers['Ae'], ['v', 's'], time=time)
network.add_monitor(exc_monitor, name='exc')


# In[3]:


def log(msg):
    print(msg, end='')
    with open(lfname, 'a') as f:
        f.write(msg)


# In[4]:


def detect(spikes):
    cols = torch.sum(spikes, dim=1)
    for i, v in enumerate(cols):
        if v != 0:
            log(f'Neuron: {i} Spikes: {int(v)}\n')


# In[5]:


tracks = [pickle.load(open('encoding/track1_numenta.p', 'rb')) * 255,
          pickle.load(open('encoding/track2_numenta.p', 'rb')) * 255,
          pickle.load(open('encoding/track3_numenta.p', 'rb')) * 255]


# In[ ]:


for track_n in range(len(tracks)):
    log(f'Starting Training on Track {track_n}\n')
    track = tracks[track_n]
    for i in range(4, len(track)):
        orig = torch.cat((track[i-4], track[i-3], track[i-2], track[i-1], track[i]))
        pt = poisson(orig, time)
        
        inpts = {'X': pt}

        network.run(inpts=inpts, time=time)
        spikes = exc_monitor.get('s')
        voltage = exc_monitor.get('v')
        network.reset_()
        
        log(f'Iteration {i - 3}\n')
        detect(spikes)
        if plot:
            fig = plt.figure(figsize=(20, 20))
            plt.subplot(2, 2, 1)
            plt.imshow(spikes, cmap='binary')
            plt.subplot(2, 2, 2)
            plt.imshow(orig.view(125, 40), cmap='gist_gray')
            plt.show()
            
            clear_output(wait=True)


# In[ ]:


network.save('trained.net')

