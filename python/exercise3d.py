"""
================================================
SVM: Maximum margin separating decision boundary
================================================

SVM for XOR problem
"""
print __doc__

import numpy as np
import pylab as pl
from sklearn import svm

X = np.array([ [-1,-1], [-1,1], [1,-1], [1,1] ])
Y = np.array([???,???,???,???])  # FIX!!! 


# create a mesh to plot in
h=.02 # step size in the mesh
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# fit the model

# kernel functions: see scikit documentation: 
# http://scikit-learn.sourceforge.net/stable/modules/svm.html (section 3.2.7)
# polynomial ("poly"): (gamma*<x,y>+r)**d corresponding to (gamma*<x,y>+coef0)**degree
# RBF kernel ("rbf"): exp(-gamma|x-y|**2), i.e. gamma corresponds to gamma = 1/(2*sigma*sigma)
clf = svm.SVC(kernel='???', degree=???, gamma=???, coef0=???, C=1.0)   # FIX!!!  
clf.fit(X, Y)

# plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# put the result into a color plot
Z = Z.reshape(xx.shape)
pl.set_cmap(pl.cm.Paired)     # sets colormap
pl.contourf(xx, yy, Z)
#pl.axis('off')

#plot also the training points
pl.scatter(X[:,0], X[:,1], c=Y)

pl.title('XOR')
pl.xlabel('x1')
pl.ylabel('x2')
pl.axis('tight')
pl.show()

