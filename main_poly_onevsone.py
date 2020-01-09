from datetime import datetime

import numpy as np

from perceptron import onevsonePerceptron
from kernels import polynomial_kernel, radial_basis_kernel
from utilities import data_split, y_encode, perceptron_learning
from utilities import stratified_k_fold, vectorised_p_strat_kfold


def q1(Perceptron, X, y, d_vals, percentage, epochs=1, seed=0, runs=1):
    """
    Calculate the test/train accuracies for polynomial perceptrons
    indexed by parameters within the list d_vals.

    Returns
    test_e_full: list[float]: test accuracies for each run
    train_e_full: list[float]: train accuracies for each run
    """
    train_e_full = [d_vals]
    test_e_full = [d_vals]
    print("Question 1")
    for r in range(runs):
        train_e_runs = []
        test_e_runs = []
        for i, d in enumerate(d_vals):
            print(
                'Time: {}, Run: {}, Polynomial: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                r+1,
                d)
            )
            P = Perceptron(polynomial_kernel, k_params=d)
            train_e, test_e = stratified_k_fold(
                P, X, y, percentage=percentage, epochs=epochs, seed=seed+r, conf=False
            )
            train_e_runs.append(train_e)
            test_e_runs.append(test_e)
        train_e_full.append(train_e_runs)
        test_e_full.append(test_e_runs)
    return train_e_full, test_e_full


def kernel_d_selection(Perceptron, X, y, d_vals, k=5, epochs=1, seed=0):
    """
    Perform stratified k-fold cross validation to determine
    the most performant hyper parameter contained within the
    list d_vals

    Returns
    np.array(d_errors): test accuracies for all x-validated
        hyperparameters contained within d_vals
    """
    d_errors = np.zeros((len(d_vals), 2))
    errors = np.zeros((2*len(d_vals), epochs+2))
    for i, d in enumerate(d_vals):
        print("Polynomial Kernel: d = {}".format(d))
        p = Perceptron(polynomial_kernel, k_params=d)
        error = vectorised_p_strat_kfold(
            p, X, y, k, epochs, seed
        )
        xval_error_mean = np.array(error).mean(axis=0)
        xval_error_std = np.array(error).std(axis=0)
        errors[(2*i):(2*i+2), :2] = [[d, 99999], [d, -99999]]
        errors[(2*i):(2*i+2), 2:] = [xval_error_mean, xval_error_std]
        d_errors[i, :] = [xval_error_mean[-1], xval_error_std[-1]]
    return np.array(d_errors), errors


def d_hyperparameter_selection(Perceptron,
                               X,
                               y,
                               d_vals,
                               k=5,
                               epochs=1,
                               seed=0,
                               runs=20):
    """
    Perform multiple runs of k-fold cross validation using
    hyperparameters within the list d_vals.

    Returns
    np.array: each row contains the most performant hyperparameter
        for a single run, along with its train and test accuracy.
    """
    d_prime_list = []
    train_errors = []
    test_errors = []
    full_confusion = np.array([]).reshape(0,2)
    for r in range(runs):
        print('Run: {}'.format(r+1))
        d_errors, _ = kernel_d_selection(
            onevsonePerceptron,
            X,
            y,
            k=k,
            d_vals=d_vals,
            epochs=epochs,
            seed=seed+r
        )
        # Select most performant hyperparameter
        d_prime = np.argmax(d_errors[:, 0]) + 1
        print("D-Prime: {}".format(d_prime))
        print(d_errors[:, 0])
        P = Perceptron(polynomial_kernel, d_prime)
        train_error, test_error = stratified_k_fold(
            P, X, y, percentage=(1/k), epochs=epochs, seed=seed+r, conf=False
        )
        d_prime_list.append(d_prime)
        train_errors.append(train_error)
        test_errors.append(test_error)
    return np.array([d_prime_list, train_errors, test_errors])


def main():
    # Load Data
    data = np.loadtxt("zipcombo.dat")
    X,y = data_split(data,y_col=0)
    poly_d = list(range(1, 8))

    d_errors, full_errors = kernel_d_selection(
        onevsonePerceptron,X,y,k=5,d_vals = list(range(1, 8)),epochs = 10
    )
    q1_train, q1_test = q1(
        onevsonePerceptron, X, y, poly_d, percentage=0.2, epochs=10, seed=0, runs=20
    )
    np.savetxt('q1_train_errors_1v1.csv', q1_train, delimiter=',', fmt='%10.20f')
    np.savetxt('q1_test_errors_1v1.csv', q1_test, delimiter=',', fmt='%10.20f')

    d_prime_errors = d_hyperparameter_selection(
        onevsonePerceptron, X, y, d_vals=poly_d, k=5, epochs=10, seed=0, runs=20
    )
    np.savetxt('d_prime_errors_1v1.csv', d_prime_errors, delimiter=',', fmt='%10.20f')

    np.savetxt('d_errors_1v1.csv', d_errors, delimiter=',', fmt='%10.20f')
    np.savetxt('full_errors_1v1.csv', full_errors, delimiter=',', fmt='%10.20f')


if __name__ == '__main__':
    main()
