import numpy as np
from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from pylab import plot, title, show , legend

#load text file
#points = np.loadtxt('data1a.txt')

points_m1 = np.array([  [0,2], [2,3] ])
points1 = np.array([ [0,0],  [2,-1]])
#pointstest = np.array([ [0.5,0.5],  [1.5,1.25]])
#np.cross()

plot(points_m1[:,0],points_m1[:,1],'o')
plot(points1[:,0],points1[:,1],'*')

#plot(pointstest[:,0],pointstest[:,1],'*')

w = np.array([0,1])

xt=linspace(-2,3,100)
yt = xt * w[0]  + w[1]
plot(xt,yt)

#y = points[:,1]
#print points
#x = np.asmatrix(np.c_[np.ones(y.size),points[:,0]])
#
#b = np.dot(np.transpose(x),y)
#
#A = np.dot( np.transpose(x),x)
#
#w = np.linalg.solve(A,np.transpose(b))
#
#plot(points[:,0],y,'o')
#xt=linspace(np.min(x),np.max(x),100)
#yt = xt * w[1,0]  + w[0,0]
#plot(xt,yt)
