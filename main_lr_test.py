from datetime import datetime

import numpy as np

from logistic_regression import LogisticRegression
from utilities import data_split
from sklearn.model_selection import StratifiedKFold, train_test_split
#from utilities import stratified_k_fold#, vectorised_p_strat_kfold


def validate_tolerance(X, y, tol_list, reg):
    """
    Perform 5-fold validation on dataset X, y with
    varying tolerances specified by tol_list

    Returns a 5 x len(tol_list) array
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    accuracy = np.zeros((5, 10))
    i = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for j, tol in enumerate(tol_list):
            lr = LogisticRegression(lr=1, reg=reg)
            lr.train(X_train, y_train, tol=tol)
            y_pred = lr.predict(X_test)
            acc = (y_pred == y_test).mean()
            accuracy[i,j]  = acc
            print("Tolerance: {}, Accuracy: {}".format(tol, acc))
        i += 1
    statistics = np.zeros((2, len(tol_list)))
    statistics[0, :] = np.mean(accuracy, 0)
    statistics[1, :] = np.std(accuracy, 0)
    return accuracy


def q1_regularised(X, y, reg_list, tol):
    """
    For 20 different test/train splits, calculate
    the classification accuracy on the test set using
    the values in reg_list
    """
    accuracy = np.zeros((20, len(reg_list)))
    for i in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )
        print("Run {}".format(i))
        for j, reg in enumerate(reg_list):
            lr = LogisticRegression(lr=1, reg=reg)
            lr.train(X_train, y_train, tol=tol)
            y_pred = lr.predict(X_test)
            acc = (y_pred == y_test).mean()
            accuracy[i,j]  = acc
            print("Reg Parameter: {}, Accuracy: {}".format(reg, acc))

    statistics = np.zeros((2, len(reg_list)))
    statistics[0, :] = np.mean(accuracy, 0)
    statistics[1, :] = np.std(accuracy, 0)
    return accuracy


def q2_regularised(X, y, n, reg_list, tol):
    """
    For n different 5-fold test/train splits, calculate
    the classification accuracy on the test set using
    the values in reg_list
    """
    accuracy = np.zeros((n, len(reg_list), 5))
    for i in range(n):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        k = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for j, reg in enumerate(reg_list):
                lr = LogisticRegression(lr=1, reg=reg)
                lr.train(X_train, y_train, tol=tol)
                y_pred = lr.predict(X_test)
                acc = (y_pred == y_test).mean()
                accuracy[i, j, k]  = acc
                print(
                    "Q2 -- Run {}, Fold {}, Reg: {}/{}, Accuracy: {}".format(
                        i+1, k+1, j+1, len(reg_list), acc
                    )
                )
            k += 1
    statistics = np.zeros((2, len(reg_list)))
    statistics[0, :] = np.mean(accuracy, axis=(0, 2))
    statistics[1, :] = np.std(accuracy, axis=(0, 2))
    return statistics


def main():
    # Load Data
    data = np.loadtxt("zipcombo.dat")
    X,y = data_split(data,y_col=0)

    # Investigate effect of tolerance
#    tol_list = [0.001*(tol+1) for tol in range(0, 20, 1)]
#    reg = 0.1
#    xval_tolerance_acc = validate_tolerance(X, y, tol_list, reg)
#    np.savetxt(
#        'lr_tol_variable_selection_w_reg_{}.csv'.format(reg),
#        xval_tolerance_acc,
#        delimiter=',',
#        fmt='%10.20f'
#    )

    # Optimal tolerance derived from X-val
    tol = 0.05
    reg_list = [2**(-i) for i in range(1, 15)]
    n = 3
    xval_reg_stats = q2_regularised(X, y, n, reg_list, tol)
    np.savetxt(
        'lr_reg_variable_selection.csv',
        xval_reg_stats,
        delimiter=',',
        fmt='%10.20f'
    )


if __name__ == '__main__':
    main()
