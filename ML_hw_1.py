import numpy
import matplotlib.pyplot
from scipy import stats
from sklearn import neighbors

from sklearn.metrics import mean_squared_error
from math import sqrt

numpy.random.seed(10)
x_i = numpy.sort(numpy.random.uniform(-1, 1, 100), axis=0)
# https://thecuriousastronomer.wordpress.com/2014/06/26/what-does-a-1-sigma-3-sigma-or-5-sigma-detection-mean/

e_i = numpy.random.normal(0, .285, 100)  # (-1,1,100) wrong. .285 *3.5= proper sigma for 10k points s.t. -1<x<1

# given f(x) = 1.8*x_i + 2

# given y_i = f(x) + e_i

# model
y_i = 1.8 * x_i + 2 + e_i

numpy.random.seed(1425)
# create test set by repeating above 10,000 times
x_i_test = numpy.sort(numpy.random.uniform(-1, 1, 10000), axis=0)

e_i_test = numpy.random.normal(0, .285, 10000)  # (-1,1,10000) wrong. .285 *3.5= proper sigma for 10k points s.t. -1<x<1

#  model
y_i_test = 1.8*x_i_test + 2 + e_i_test

#  now plot the relationship (test/train is not specified)
matplotlib.pyplot.scatter(x_i_test, y_i_test)
matplotlib.pyplot.scatter(x_i, y_i)

#  graph against plot of f(x)
true_fx_xvals = numpy.arange(-1, 1, 0.01)  # Grid of 0.01 spacing from -1 to 1
true_fx_yvals = 1.8 * true_fx_xvals + 2  # Evaluate function on xvals
matplotlib.pyplot.plot(true_fx_xvals, true_fx_yvals, 'r--', label='f')  # Create line plot with yvals against xvals
matplotlib.pyplot.show()


gradient,intercept,r_value,p_value,std_err=stats.linregress(x_i,y_i)

# https://stackoverflow.com/questions/22239691/code-for-line-of-best-fit-of-a-scatter-plot-in-python
matplotlib.pyplot.plot(numpy.unique(x_i), numpy.poly1d(numpy.polyfit(x_i, y_i, 1))(numpy.unique(x_i)),'orange')

# Plot k-NN for k=2 through k=15
# code found at https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/

# rmse_val = [] # to store rmse values for different k
# for K in range(50):
#     K = K+1
#     model = neighbors.KNeighborsRegressor(n_neighbors = K)
#
#     model.fit(x_i.reshape(-1, 1), y_i.reshape(-1, 1))  # fit the model
#     pred = model.predict(x_i_test.reshape(-1, 1))  # make prediction on test set
#     error = sqrt(mean_squared_error(y_i_test.reshape(-1, 1),pred))  # calculate rmse
#     rmse_val.append(error)  # store rmse values
#     print('RMSE value for k= ' , K , 'is:', error)  # k=30 has lowest RMSE

# to plot this data, take from https://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html
# #################################
# Fit regression model
n_neighbors = 5

X = x_i.reshape(-1, 1)
y = y_i.reshape(-1, 1)
T = x_i.reshape(-1, 1)
# for i, weights in enumerate(['uniform', 'distance']):
for i, n_neighbors in enumerate([2, 5, 10, 12]):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
    y_ = knn.fit(X, y).predict(X)  # changed from T to X
    matplotlib.pyplot.subplot(4, 1, i + 1)
    matplotlib.pyplot.scatter(x_i_test.reshape(-1, 1),y_i_test.reshape(-1,1), c='grey', label='test data')
    matplotlib.pyplot.scatter(X, y, c='k', label='training data')
    matplotlib.pyplot.plot(X,y_, c='g', label='prediction')  # changed from T to X
    matplotlib.pyplot.axis('tight')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                              'uniform'))
# plot f(x) for comparison
    matplotlib.pyplot.plot(true_fx_xvals, true_fx_yvals, 'r--', label='f')  # Create line plot with yvals against xvals

matplotlib.pyplot.tight_layout()
matplotlib.pyplot.show()

