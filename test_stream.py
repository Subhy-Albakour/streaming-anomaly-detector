from numpy import column_stack, sin, cos, arange, ones,zeros

from numpy.random import randn, rand, choice

    

###################################################

# GENERATE SOME DATA

###################################################

def get_synth(T=1000,n_anoms=250,contextual=False,win=4):

    ''' 

        make anomallies in sine waves 

        -----------------------------

        T = length of stream

        n_anoms = how many anomalies to insert.

    '''

    noise=0.05

    y = zeros(T)

    X = column_stack([sin(arange(T)/4.)+randn(T)*noise, cos(arange(T)/4.)+randn(T)*noise])



    if contextual:
        contextual_anomalies =choice(T-win,n_anoms)                       # the places of trigger for a non-anomaly
        X[contextual_anomalies+win,1] =X[contextual_anomalies,0]          # the non-anomaly



    anoms = choice(T,n_anoms)                                              #the places

    X[anoms,1] = sin(choice(n_anoms)) + randn(n_anoms) *noise + 2.               # <--- 

    y[anoms] = 1.

    return X, y

#===================================

#Usage:

#===================================
from skmultiflow.data import DataStream

    
# Create test data

T = 3000

n_anomalies = int(T / 100)

X, y = get_synth(T, n_anomalies, contextual=False)

T, D = X.shape
print(T,D)


import matplotlib.pyplot as plt

plt.plot(X[:100,0],'ro-')
plt.plot(y[:100],'go')
plt.show()

# Convert data to stream

# stream = DataStream(data=X, y=y)

# stream.prepare_for_use()