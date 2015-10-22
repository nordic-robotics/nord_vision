import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

files = [ f for f in listdir(".") if "txt" in f ]

print files

fig = plt.figure()
ax = plt.subplot(111)
legends= []

colormap = {'red':'r', 'yellow':'y','green':'g','blue':'b','orange':'k','purple':'m'}


for f in files:
    data = pd.read_csv(f,delimiter=" ")
    objectName = f[:-4]
    legends.append( objectName )
    print objectName
    data = np.array(data)
    data = data[-500:,:]
#    print data.shape
    for color in colormap.keys():
        if color in objectName:
            mark = colormap[color]
#    print mark+'o'
    ax.plot(data[:,0],data[:,1],'o', alpha=0.1)
    ax.legend(legends, loc='upper center')

    hist, edge = np.histogram(data[:,0],bins=20)
#    ax.plot(edge[:-1],hist)
#    ax.legend(legends, loc='upper center')   
#    plt.ylim([0,255])
#    plt.xlim([0,180])
#    plt.show()
plt.ylim([0,255])
plt.xlim([0,180])
plt.show()

