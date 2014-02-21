__author__ = 'maucher'

import numpy as np
import prettytable
np.set_printoptions(precision=3)

parameters=[' ','sdnn','sd1','sd2','rr_max','rr_mean','rr_max_min','pnn50','rr_min','rmssd']
xtab=prettytable.PrettyTable(parameters)
X=np.load("NormalizedHRVdata.npy")

CoVar=np.cov(np.transpose(X)).round(3)
#print parameters
for r in range(CoVar.shape[0]):
    row = list(CoVar[r,:])
    row.insert(0,parameters[r+1])
    xtab.add_row(row)
print xtab