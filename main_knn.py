from datetime import datetime

import numpy as np

from knn import kNN
from utilities import data_split
from sklearn.model_selection import StratifiedKFold, train_test_split
#from utilities import stratified_k_fold#, vectorised_p_strat_kfold


def validate_k(X, y, k_list):
    """
    Perform 5-fold validation on dataset X, y with
    varying K values from k_list. Used to derive some
    intuition about what range of k to validate over.

    Returns a 5 x len(k_list) array of test accuracies
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    accuracy = np.zeros((5, len(k_list)))
    i = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for j, k in enumerate(k_list):
            classifier = kNN()
            classifier.train(X_train, y_train)
            y_pred = classifier.predict(X_test, k)
            acc = (y_pred == y_test).mean()
            accuracy[i,j]  = acc
            print(
                "Q1 -- TIME: {} Fold {}, k: {}, Accuracy: {}".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i+1, k, acc
                )
            )
        i += 1
    statistics = np.zeros((2, len(k_list)))
    statistics[0, :] = np.mean(accuracy, 0)
    statistics[1, :] = np.std(accuracy, 0)
    return accuracy


def q1_regularised(X, y, k_list):
    """
    For 20 different test/train splits, calculate
    the classification accuracy on the test set using
    the values in k_list.

    Returns:
    test_acc_ar: np.array: test errors for all 20 runs,
        across all hyperparameters in k_list.
    """
    test_acc_ar = np.zeros((20, len(k_list)))
    for i in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )
        print("Run {}".format(i))
        for j, k in enumerate(k_list):
            classifier = kNN()
            classifier.train(X_train, y_train)
            y_pred = classifier.predict(X_test, k)
            test_acc = (y_pred == y_test).mean()
            test_acc_ar[i,j]  = test_acc
            print("k Parameter: {}, Accuracy: {}".format(k, test_acc))
    return test_acc_ar


def q2_regularised(X, y, n, k_list):
    """
    For n different 5-fold test/train splits, calculate
    the classification accuracy on the test set using
    the values in k_list

    Returns
    statistics: n by len(k_list) array of the mean and std
        test errors for each x-validated run. 
    k_max_index: list[int]: indexes of the most performant
        hyperparameters k for each one (use to derive optimal
        hyperparams from k_list)
    """
    accuracy = np.zeros((n, len(k_list), 5))
    for i in range(n):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        l = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for j, k in enumerate(k_list):
                classifier = kNN()
                classifier.train(X_train, y_train)
                y_pred = classifier.predict(X_test, k)
                acc = (y_pred == y_test).mean()
                accuracy[i, j, l]  = acc
                print(
                    "Q2 -- Run {}, Fold {}, k: {}/{}, Accuracy: {}".format(
                        i+1, l+1, j+1, len(k_list), acc
                    )
                )
            l += 1
    statistics = np.zeros((2, len(k_list)))
    statistics[0, :] = np.mean(accuracy, axis=(0, 2))
    statistics[1, :] = np.std(accuracy, axis=(0, 2))

    k_stats = np.mean(accuracy, axis=2)
    # Selecting most performant K parameter index
    k_max_index = np.argmax(k_stats, axis=1)
    print(k_max_index)
    k_mean = np.mean(accuracy, axis=2)
    k_max_index = np.argmax(k_mean, axis=1)
    k_std = np.std(accuracy, axis=2)
    # Selecting most performant K parameter
    optimal_params = [k_list[i] for i in k_max_index]
    print(optimal_params)
    optimal_errors = []
    # Rerun classification using most optimal parameter to find smallest error
    for i, k in enumerate(optimal_params):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )
        classifier = kNN()
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_test, k)
        error = (y_pred == y_test).mean()
        print("K : {}, Error : {}".format(k, error))
        optimal_errors.append(error)

    optimal_mean = np.mean(optimal_errors)
    optimal_std = np.std(optimal_errors)
    print(optimal_errors)
    print(k_max_index)
    print("Optimal Mean Error: {} Optimal Std Error: {}".format(optimal_mean, optimal_std))

    return statistics, k_max_index


def main():
    # Load Data
    data = np.loadtxt("zipcombo.dat")
    X,y = data_split(data,y_col=0)

    # Investigate effect of k values
    k_list = [1, 3, 7, 9, 27]
    xval_k_val = validate_k(X, y, k_list)
    np.savetxt(
        'knn_variable_selection.csv',
        xval_k_val,
        delimiter=',',
        fmt='%10.20f'
    )
    # Solve quesiton 1
    q1_errors = q1_regularised(X, y, k_list)
    np.savetxt(
        'knn_q1.csv',
        q1_errors,
        delimiter=',',
        fmt='%10.20f'
    )

    n = 20
    xval_k_stats, k_max_index = q2_regularised(X, y, n, k_list)
    np.savetxt(
        'knn_variable_selection_n_{}.csv'.format(n),
        xval_k_stats,
        delimiter=',',
        fmt='%10.20f'
    )
    np.savetxt(
        'knn_k_max_index.csv'.format(n),
        k_max_index,
        delimiter=',',
        fmt='%10.20f'
    )


if __name__ == '__main__':
    main()
