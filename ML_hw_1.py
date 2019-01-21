import numpy
import matplotlib.pyplot
from scipy import stats
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt

numpy.random.seed(10)

# given f(x) = 1.8*x_i + 2
x_i,x_i_test = numpy.sort(numpy.random.uniform(-1, 1, 100), axis=0), \
               numpy.sort(numpy.random.uniform(-1, 1, 10000), axis=0)
# https://thecuriousastronomer.wordpress.com/2014/06/26/what-does-a-1-sigma-3-sigma-or-5-sigma-detection-mean/
e_i,e_i_test = numpy.random.normal(0, 1, 100), \
               numpy.random.normal(0, 1, 10000)
y_i,y_i_test = 1.8 * x_i + 2 + e_i, \
               1.8 * x_i_test + 2 + e_i_test  # given y_i = f(x) + e_i

#  create points for the true  f(x) function
true_fx_x_vals = numpy.linspace(-1, 1, 10000)
true_fx_y_vals = 1.8 * true_fx_x_vals + 2  # Grid of 0.01 spacing from -1 to 1


#matplotlib.pyplot.scatter(x_i_test, y_i_test)
#matplotlib.pyplot.scatter(x_i, y_i)
#matplotlib.pyplot.plot(true_fx_xvals, true_fx_yvals, 'r--', label='f')  # Create line plot with yvals against xvals
#matplotlib.pyplot.show()

true_fx_xvals_trainset = numpy.linspace(-1,1,100)

true_fx_yvals_trainset = 1.8 * true_fx_xvals_trainset + 2


gradient,intercept,r_value,p_value,std_err=stats.linregress(x_i,y_i)

# https://stackoverflow.com/questions/22239691/code-for-line-of-best-fit-of-a-scatter-plot-in-python
#matplotlib.pyplot.plot(numpy.unique(x_i), numpy.poly1d(numpy.polyfit(x_i, y_i, 1))(numpy.unique(x_i)),'orange')

# Plot k-NN for k=2 through k=15
# code found at https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/

rmse_val = [] # to store rmse values for different k
rmse_val_train = []
for K in range(1,81):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_i.reshape(-1, 1), y_i.reshape(-1, 1))  # fit the model
    pred, pred_train = model.predict(x_i_test.reshape(-1, 1)), model.predict(x_i.reshape(-1, 1))  # make prediction on test set
    error = sqrt(mean_squared_error(y_i_test, pred))  # calculate rmse against test data using y hat, not f(x)
    train_error = sqrt(mean_squared_error(y_i, pred_train)) # calculate mse against training data
    rmse_val.append(error)  # store rmse values
    rmse_val_train.append(train_error)
    print('RMSE value for k= ' , K , 'is:', error)  # k=30 has lowest RMSE
    print('In-Sample RMSE value for k= ', K, 'is:', train_error)  # k=30 has lowest RMSE
    #calculate MSE against test data
    #y_i_test

x = numpy.linspace(1,81,80)

matplotlib.pyplot.plot(numpy.log(1.0/x), numpy.array(rmse_val), c='r', label = 'RMSE')
matplotlib.pyplot.plot(numpy.log(1.0/x), numpy.array(rmse_val_train), c='black', label ='In sample RMSE')
matplotlib.pyplot.legend(loc='upper left')
matplotlib.pyplot.xlabel('Log(1/k)')
matplotlib.pyplot.ylabel('RMSE')
matplotlib.pyplot.show()

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
    matplotlib.pyplot.plot(true_fx_x_vals, true_fx_y_vals, 'r--', label='f')  # Create line plot with yvals against xvals

matplotlib.pyplot.tight_layout()
matplotlib.pyplot.show()

