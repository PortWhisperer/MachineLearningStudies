import numpy as np
import matplotlib.pyplot
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt


def eval_Knn_Performance(var_list):
    if len(var_list[0]) == 5:
        y_i, y_i_test, true_fx_yvals_train, x_i_p, x_i_p_test = var_list[0]
    else:
        y_i, y_i_test, true_fx_yvals_train = var_list[0]

    if x_i_p and x_i_p_test:
        xi_test, xi = var_list[3], var_list[4]  # pull in x values from list if avail
    else:
        xi_test, xi = x_i_test, x_i  # else pull in x values from global scope and avoid shadowing

    # create points for the true  f(x) function
    true_fx_x_vals = np.linspace(-1, 1, 10000)
    true_fx_y_vals = true_fx_yvals_train

    # matplotlib.pyplot.scatter(x_i_test, y_i_test)
    # matplotlib.pyplot.scatter(x_i, y_i)
    # matplotlib.pyplot.plot(true_fx_xvals, true_fx_yvals, 'r--', label='f')  # Plot true f(x) function
    # matplotlib.pyplot.show()

    true_fx_xvals_trainset = np.linspace(-1, 1, 100)

    # plot f(x) against the larger x,y scatterplot with 10k observations
    # may not be needed if we plot against the train set since f(x) there is plotted from (-1,1) already
    # ###################################################
    true_fx_yvals_trainset = 1.8 * true_fx_xvals_trainset + 2

    # Plot line of best fit for our fitted model
    # https://stackoverflow.com/questions/22239691/code-for-line-of-best-fit-of-a-scatter-plot-in-python
    # #########################################################################

    matplotlib.pyplot.plot(np.unique(xi), np.poly1d(np.polyfit(xi, y_i, 1))(np.unique(xi)), 'orange')

    # Plot k-NN for k=2 through k=15
    # code found at https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/

    rmse_val = []  # to store rmse values for different k
    rmse_val_train = []

    for K in range(1, 15):
        K = K + 1

        model = neighbors.KNeighborsRegressor(n_neighbors=K)  # create the model
        model.fit(xi.reshape(-1, 1), y_i)  # fit the KNN model

        pred = model.predict(xi_test.reshape(-1, 1))
        pred_train = model.predict(xi.reshape(-1, 1))  # make prediction on test set

        error = sqrt(mean_squared_error(y_i_test, pred))  # calculate rmse against test data using y hat, not f(x)
        error_train = sqrt(mean_squared_error(y_i, pred_train))  # calculate mse against training data

        rmse_val.append(error)  # store rmse values
        rmse_val_train.append(error_train)

    # Run linear regression model
    # ###########################
    lin_reg_model_test = np.polyfit(xi, y_i, deg=1)
    lin_reg_model_train = np.polyfit(xi_test, y_i_test, deg=1)

    lin_reg_model_pred_train = lin_reg_model_train[0] * np.linspace(-1, 1, 100) + lin_reg_model_train[1]
    lin_reg_model_pred_test = lin_reg_model_test[0] * np.linspace(-1, 1, 10000) + lin_reg_model_test[1]

    error_lin_reg_test = sqrt(mean_squared_error(y_i_test, lin_reg_model_pred_test))  # out of sample RMSE = 1.0233
    error_lin_reg_train = sqrt(mean_squared_error(y_i, lin_reg_model_pred_train))  # in sample RMSE = .93796

    # plot results
    # RMSE vs log(k)
    k = np.linspace(2.0, 15.0, 14)  # create for matplotlib plotting purposes
    fig, (ax1, ax2) = matplotlib.pyplot.subplots(2, 1)  # create subplots
    fig.subplots_adjust(hspace=0.5)

    ax1.plot(np.log(1.0 / k), np.array(rmse_val), c='r', label='RMSE')
    ax1.plot(np.log(1.0 / k), np.array(rmse_val_train), c='black', label='In sample RMSE')
    ax1.axhline(y=error_lin_reg_test, c='g', dashes=(5, 1), label="Lin Reg RMSE")
    ax1.set_xlabel('Log(1/k)')
    ax1.set_ylabel('RMSE')
    ax1.legend(loc='upper left')

    # RMSE vs K
    ax2.plot(k, np.array(rmse_val), c='r', label='RMSE')
    ax2.plot(k, np.array(rmse_val_train), c='black', label='In sample RMSE')
    ax2.axhline(y=error_lin_reg_test, c='g', dashes=(5, 1), label="Lin Reg RMSE")
    ax2.set_xlabel('K')
    ax2.set_ylabel('RMSE')
    ax2.legend(loc='upper left')
    matplotlib.pyplot.show()

    # to plot this data, take from https://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html
    # #################################
    # Fit regression model
    n_neighbors = 5

    X = xi.reshape(-1, 1)
    y = y_i.reshape(-1, 1)
    T = xi_test.reshape(-1, 1)
    # for i, weights in enumerate(['uniform', 'distance']):
    for i, n_neighbors in enumerate([2, 5, 10, 12]):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
        y_ = knn.fit(X, y).predict(T)  # changed from T to X
        matplotlib.pyplot.subplot(4, 1, i + 1)
        matplotlib.pyplot.scatter(xi_test.reshape(-1, 1), y_i_test.reshape(-1, 1), c='grey', label='test data')
        matplotlib.pyplot.scatter(X, y, c='k', label='training data')
        matplotlib.pyplot.plot(T, y_, c='g', label='prediction')  # changed from T to X
        matplotlib.pyplot.axis('tight')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                                  'uniform'))
        # plot f(x) for comparison
        matplotlib.pyplot.plot(true_fx_x_vals, true_fx_y_vals, 'r--',
                               label='f')  # Create line plot with yvals against xvals

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()


# given f(x) = 1.8*x_i + 2
x_i, x_i_test = np.sort(np.random.uniform(-1, 1, 100), axis=0), \
                np.sort(np.random.uniform(-1, 1, 10000), axis=0)

# https://thecuriousastronomer.wordpress.com/2014/06/26/what-does-a-1-sigma-3-sigma-or-5-sigma-detection-mean/
e_i, e_i_test = np.random.normal(0, 1, 100), \
                np.random.normal(0, 1, 10000)

true_func_x_vals = np.linspace(-1, 1, 10000)

# Plot actual function
# ################################################
true_func_y_vals_lin = 1.8 * true_func_x_vals + 2
true_func_y_vals_tan = np.tan(1.1 * true_func_x_vals) + 2
true_func_y_vals_sin = np.sin(2 * true_func_x_vals) + 2
true_func_y_vals_sin_poly = 1.8 * true_func_x_vals + 2  # placeholder for eventual polynomial func

# y variable creation
# ################################################
arr = []  # 0=y_i, 1=y_i_test, 2=true_fx_yvals_train, 3=operation 4=x_i_p, 5=x_i_p_test
arr.append((1.8 * x_i + 2 + e_i, 1.8 * x_i_test + 2 + e_i_test, true_func_y_vals_lin, 'Linear Reg'))
arr.append((np.tan(1.1 * x_i) + 2 + e_i, np.tan(1.1 * x_i_test) + 2 + e_i_test, true_func_y_vals_tan, 'Tan Func'))
arr.append((np.sin(2 * x_i) + 2 + e_i, np.sin(2 * x_i_test) + 2 + e_i_test, true_func_y_vals_sin, 'Sin Func'))
# (y_i_sin_poly, y_i_test_sin_poly) = (np.sin(2 * x_i) + 2 + e_i,np.sin(2 * x_i_test) + 2 + e_i_test)

# placeholder for loop that will generate 20 different models for 2.8. Models will then be passed into function.

[eval_Knn_Performance(data) for data in arr]
