# quick and easy gaussian kernel implementation in python, mostly based off of
# this octave implementation: https://www.researchgate.net/publication/358654985_Slides_Regression_using_Kernel_Methods

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics import mean_squared_error

#Polynomial interpolation
import numpy as np
import matplotlib.pyplot as plt
import math

#your own data is being imported
trueData = pd.read_csv('./DS-5-1-GAP-0-1-N-0_v2.csv',header=None)
noise1 = pd.read_csv('./DS-5-1-GAP-1-1-N-1_v2.csv',header=None)
noise2 = pd.read_csv('./DS-5-1-GAP-5-1-N-3_v2.csv',header=None)

# The kernel, which describes a gaussian curve.
# Source: https://www.researchgate.net/publication/358654985_Slides_Regression_using_Kernel_Methods#pfb (10th slide)
def K1(x_data, numberPoints, center, numKernels, gaussianWidth):
  mat = [[0 for _ in range(numberPoints)] for _ in range(numKernels)] #we fill the matrix
  for j in range(0,numKernels):
    for i in range(0,numberPoints):
      mat[j][i] = math.exp(-((abs(x_data[i] - center[j])**2)/(gaussianWidth[j]**2)))
  return mat


# noise1 is a sample with noisy data. In the first column, we got the X's, and 
# in the second column, we got the first realization of y's
X = noise1[0]  #time
x = X[:, np.newaxis]
Y = noise1[1]  #first y
y = Y[:, np.newaxis]


# the actual true data, we use this to check how correct our model is
X_test = trueData[0]
Y_test = trueData[1]

x_test = X_test[:, np.newaxis]
y_test = Y_test[:, np.newaxis]


sigma = 7 #gaussian width

ones = []
for i in range(0, Y.size):
  ones.append(1)

onesNp = np.array(ones)

gramMatrix = K1(X, X.size, X, X.size, onesNp * sigma)
gramMatrixNumpy = np.matrix(gramMatrix)
pinvGram = np.linalg.pinv(gramMatrixNumpy.getH())
alpha = pinvGram.dot(Y)

alphaTransposed = alpha.getH()
alphaTransposed1Dimension = np.squeeze(alphaTransposed)

kernelModel = alphaTransposed1Dimension.dot(gramMatrixNumpy)

MSE_train = mean_squared_error(y,np.asarray(kernelModel.getH()))

kernelModelArray = np.squeeze(np.asarray(kernelModel.getH()))
y_predicted = np.interp(X_test, X, kernelModelArray) #interpolation to see missing points
MSE_test = mean_squared_error(y_test,y_predicted)


print("MSE train ",MSE_train)
print("MSE test",MSE_test)

plt.plot(x_test,y_test, color='k', label="True")
plt.scatter(X, y, edgecolor='b', s=20, label="Training samples")
plt.plot(x_test, y_predicted, color='g', label="Kernel Model")
plt.xlabel("time")
plt.ylabel("mag")
plt.legend(loc="best")
plt.title("Sigma {}\nMSE_train = {:.8}\nMSE_test = {:.8}".format(
        sigma, MSE_train, MSE_test))
plt.show()
