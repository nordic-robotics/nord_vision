import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import sys
from time import time


def crossvalidation(X,y,C_range,gamma_range):
    """performs 5-fold crossvalidation for rbf svm 
    with various gamma: smoothing constant and C: regularization"""
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(y, n_iter=2, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    return grid

def readTrainAndTestData(nr_points, nr_test_points, files, classes):
    """creates a training set of nr_points from each object
    and a test set of nr_test_points from each object. Objests are contained in files"""
    total_nr_points = 14*nr_points
    total_test_points = 14*nr_test_points

    train_data = np.zeros([total_nr_points,2])
    train_targets = np.zeros(total_nr_points)
    
    test_data = np.zeros([total_test_points,2])
    test_targets = np.zeros(total_test_points)

    for i,f in enumerate(files):
        #print f
        data = pd.read_csv(f,delimiter=" ")
        objectName = f[:-4]
        data = np.array(data)
        train_data[i*nr_points:(i+1)*nr_points,:] = data[-nr_points:,:]
        train_targets[i*nr_points:(i+1)*nr_points] = classes[i]
        
        test_data[i*nr_test_points:(i+1)*nr_test_points,:] = data[-(nr_points+nr_test_points):-nr_points,:]
        test_targets[i*nr_test_points:(i+1)*nr_test_points] = classes[i]

    return train_data, train_targets, test_data, test_targets


# Files containg color data from all objects
files = ['yellowball.txt', 
         'yellowbox.txt', 
        'redbox.txt', 
         'lightredcylinder.txt', 
         'orangestar.txt', 
         'redball.txt', 
         'lightredbox.txt', 
        'purplestar.txt', 
         'purplecross.txt', 
         'lightgreencylinder.txt', 
         'lightgreenbox.txt', 
         'greenbox.txt', 
         'lightbluetriangle.txt', 
         'bluebox.txt']

# These classes are maybe incorrectly balanced since the 
# training data is sampled equally from eache object
classes = [1, 
            1, 
            2, 
            2, 
            2, 
            2, 
            2, 
            3, 
            3, 
            4, 
            4, 
            5, 
            6, 
            6 ]

colors = ['-1','y','r','m','g','g','g','b','b']

## Make the data
nr_points = 5000
nr_test_points = 5000
train_data, train_targets, test_data, test_targets  = readTrainAndTestData(nr_points,
                                                                           nr_test_points, 
                                                                           files,
                                                                           classes)
#print train_targets
train_target_colors = [colors[i] for i in map(int,train_targets)]

if False:
    """plot the training data and test data"""
    plt.subplot(121)
    plt.plot(train_data[:,0],train_data[:,1],'o',alpha=0.1)
    plt.axis([0,180,0,255])
    plt.subplot(122)
    plt.plot(test_data[:,0],test_data[:,1],'o',alpha=0.1)
    plt.axis([0,180,0,255])
    plt.show()

X = train_data
y = train_targets

## Search for parameters

C_range = np.logspace(-3, 10, 7)
gamma_range = np.logspace(-4, 2, 3)

t = time()
grid = crossvalidation(X,y,C_range,gamma_range)
print "took {}".format(time()-t)
print("The best parameters are %s with a score of %0.5f"
      % (grid.best_params_, grid.best_score_))

## display the best model:
C = grid.best_params_["C"]
gamma = grid.best_params_["gamma"]

# C = 300.
# gamma = 0.001

rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(X, y)

correct = rbf_svc.predict(test_data)
score = float(sum(correct == test_targets)) / float(len(test_targets))
print "score: " +str( score )
if True:
    h = 1
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(111)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    print "classify"
    Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
    print "done"
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    print "plot"
    #print train_target_colors
    plt.scatter(X[:, 0], X[:, 1], c=train_target_colors, cmap=plt.cm.Paired)
    print "done"
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("SVM RBF: gamma={}, C={}, N/object={}".format(gamma, C, nr_points))
    


plt.show()
