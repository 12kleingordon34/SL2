import numpy as np

from perceptron import VectorisedKernelPerceptron
from kernels import polynomial_kernel, radial_basis_kernel
from utilities import data_split, y_encode, perceptron_learning
from utilities import stratified_k_fold, vectorised_p_strat_kfold


def kernel_d_selection(Perceptron, X, y, d_vals, k=5, epochs=1, seed=0):
    d_errors = np.zeros((len(d_vals), 2))
    errors = np.zeros((2*len(d_vals), epochs+2))
    for i, d in enumerate(d_vals):
        print("Polynomial Kernel: d = {}".format(d))
        p = Perceptron(polynomial_kernel, k_params=d)
        error = vectorised_p_strat_kfold(
            p, X, y, k, epochs, seed+i
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
    """
    train_errors = []
    test_errors = []
    full_confusion = np.array([]).reshape(0,2)
    for r in range(runs):
        d_errors, _ = kernel_d_selection(
            VectorisedKernelPerceptron,
            X,
            y,
            k=5,
            d_vals,
            epochs,
            seed+r
        )
        d_prime = np.argmax(d_errors[:, 0]) + 1
        P = Perceptron(polynomial_kernel, d_prime)
        train_error, test_error, y_confusion = stratified_k_fold(
            P, X, y, percentage=(1/k), epochs=epochs, seed=seed
        )
        train_errors.append(train_error)
        test_errors.append(test_error)
        confusion_vals = np.concatenate((full_confusion, y_confusion))
    return np.array([train_errors, test_errors]), confusion_vals


def main():
    # Load Data
    data = np.loadtxt("zipcombo.dat")
    X,y = data_split(data,y_col=0)

    d_errors, full_errors = kernel_d_selection(
        VectorisedKernelPerceptron,
        X,
        y,
        k=5,
        d_vals = list(range(1, 8)),
        epochs = 8
    )
    d_prime_errors, confusion = d_hyperparameter_selection(
        VectorisedKernelPerceptron, X, y, d_vals, k=5, epochs=1, seed=0, runs=20
    )
    np

    np.savetxt('confusion.csv', confusion, delimiter=',', fmt='i')
    np.savetxt('d_prime_errors.csv', d_prime_errors, delimiter=',', fmt='%10.20f')

    #np.savetxt('d_errors.csv', d_errors, delimiter=',', fmt='%10.20f')
    #np.savetxt('full_errors.csv', full_errors, delimiter=',', fmt='%10.20f')


if __name__ == '__main__':
    main()
