from datetime import datetime

import numpy as np

from logistic_regression import LogisticRegression
from utilities import data_split
from sklearn.model_selection import StratifiedKFold, train_test_split
#from utilities import stratified_k_fold#, vectorised_p_strat_kfold


def validate_tolerance(X, y, tol_list, reg):
    """
    Perform 5-fold validation on dataset X, y with
    varying tolerances specified by tol_list. Used to 
    investigate the effect of varying tolerances on 
    test accuracy.

    Returns
    accuracy: np.array: 5 x len(tol_list) array of 
        test accuracies for each k-fold run
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    accuracy = np.zeros((5, len(tol_list)))
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
            print(
                "Q1 -- Fold {}, Tol: {}, Accuracy: {}".format(
                    i+1, tol, acc
                )
            )
        i += 1
    statistics = np.zeros((2, len(tol_list)))
    statistics[0, :] = np.mean(accuracy, 0)
    statistics[1, :] = np.std(accuracy, 0)
    return accuracy


def q1_regularised(X, y, reg_list, tol):
    """
    For 20 different test/train splits, calculate
    the classification accuracy on the test/train set using
    the regularisation values in reg_list. The algorithm's
    tolerance is given as a single value which has already
    been predetermined.

    Returns
    test_accuracy: np.array: test accuracies for 20 runs,
        across all regularisation values
    train_accuracy: np.array: train accuracies for 20 runs,
        across all regularisation values
    """
    test_accuracy = np.zeros((20, len(reg_list)))
    train_accuracy = np.zeros((20, len(reg_list)))
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
            test_accuracy[i,j]  = acc
            print("Reg Parameter: {}, Tol: {} Accuracy: {}".format(reg, tol, acc))
            y_pred_train = lr.predict(X_train)
            train_acc = (y_pred_train == y_train).mean()
            train_accuracy[i,j]  = train_acc
    return test_accuracy, train_accuracy


def q2_regularised(X, y, n, reg_list, tol):
    """
    For n different 5-fold test/train splits, calculate
    the classification accuracy on the test set using
    the values in reg_list

    Return
    statistics: np.array: mean and std for x-validated
        test errors for taken across all 20 runs and k folds
    reg_mean: np.array: test accuracies averaged over k-folds
    reg_std: np.array: test accuracies standard deviations over k-folds
    reg_max_index: list[int]: indexes of most performant hyperparameter
        over each run
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

    reg_mean = np.mean(accuracy, axis=2)
    # To find best regularisation value index
    reg_max_index = np.argmax(reg_mean, axis=1)
    reg_std = np.std(accuracy, axis=2)
    # To find best regularisation value
    optimal_params = [reg_list[i] for i in reg_max_index]
    print(optimal_params)
    optimal_errors = []
    # Rerun test x-validation using optimised hyperparameter
    for i, reg in enumerate(optimal_params):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )
        lr = LogisticRegression(lr=1, reg=reg)
        lr.train(X_train, y_train, tol=tol)
        y_pred = lr.predict(X_test)
        error = (y_pred == y_test).mean()
        print("Reg : {}, Error : {}".format(reg, error))
        optimal_errors.append(error)

    optimal_mean = np.mean(optimal_errors)
    optimal_std = np.std(optimal_errors)
    print(optimal_errors)
    print(reg_max_index)
    print("Optimal Mean Error: {} Optimal Std Error: {}".format(optimal_mean, optimal_std))
    return statistics, reg_mean, reg_std, reg_max_index


def main():
    # Load Data
    data = np.loadtxt("zipcombo.dat")
    X,y = data_split(data,y_col=0)
    
    tol_list = [0.001*(tol+1) for tol in range(0, 10, 1)]
    tol= 0.004
    reg_list = [2**(-i) for i in range(5, 15)]

    q1_test_acc, q1_train_acc = q1_regularised(X, y, reg_list, tol)
    np.savetxt(
        'lr_q1_train.csv',
        q1_train_acc,
        delimiter=',',
        fmt='%10.20f'
    )
    np.savetxt(
        'lr_q1_test.csv',
        q1_test_acc,
        delimiter=',',
        fmt='%10.20f'
    )

    # Investigate effect of tolerance
    tol_list = [0.001*(tol+1) for tol in range(0, 20, 1)]
    reg = 0.001
    xval_tolerance_acc = validate_tolerance(X, y, tol_list, reg)
    np.savetxt(
        'lr_tol_variable_selection_w_reg_{}.csv'.format(reg),
        xval_tolerance_acc,
        delimiter=',',
        fmt='%10.20f'
    )

    # Optimal tolerance derived from X-val
    tol = 0.004
    n = 20
    xval_reg_stats, reg_mean, reg_std, reg_max_index = q2_regularised(X, y, n, reg_list, tol)
    np.savetxt(
        'lr_reg_mean.csv'.format(n),
        reg_mean,
        delimiter=',',
        fmt='%10.20f'
    )
    np.savetxt(
        'lr_reg_std.csv'.format(n),
        reg_std,
        delimiter=',',
        fmt='%10.20f'
    )
    np.savetxt(
        'lr_reg_variable_selection_n_{}.csv'.format(n),
        xval_reg_stats,
        delimiter=',',
        fmt='%10.20f'
    )
    np.savetxt(
        'lr_reg_max_index.csv',
        reg_max_index,
        delimiter=',',
        fmt='%10.20f'
    )


if __name__ == '__main__':
    main()
