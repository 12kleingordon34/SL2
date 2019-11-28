import numpy as np

from perceptron import Perceptron, KernelPerceptron
from kernels import polynomial_kernel, radial_basis_kernel
from utilities import data_split, y_encode, perceptron_learning
from utilities import multip_strat_kfold


def kernel_d_selection(P, X, y, d_vals, k=5, epochs=1, seed=0):
    d_errors = np.zeros((len(d_vals), 2))
    errors = np.zeros((2*len(d_vals), epochs+2))
    for i, d in enumerate(d_vals):
        print("Polynomial Kernel: d = {}".format(d))
        P_list = [KernelPerceptron(polynomial_kernel, k_params=d)
                  for i in range(10)]
        error = multip_strat_kfold(P_list, X, y, k, epochs, seed)
        xval_error_mean = np.array(error).mean(axis=0)
        xval_error_std = np.array(error).std(axis=0)
        errors[(2*i):(2*i+2), :2] = [[d, 99999], [d, -99999]]
        errors[(2*i):(2*i+2), 2:] = [xval_error_mean, xval_error_std]
        d_errors[i, :] = [xval_error_mean[-1], xval_error_std[-1]]
    return np.array(d_errors), errors


def main():
    # Load Data
    data = np.loadtxt("zipcombo.dat")
    X,y = data_split(data,y_col=0)

    d_errors, full_errors = kernel_d_selection(
        KernelPerceptron,
        X,
        y,
        k=5,
        d_vals = list(range(1, 8)),
        epochs = 20
    )

    np.savetxt('d_errors.csv', d_errors, delimiter=',', fmt='%10.20f')
    np.savetxt('full_errors.csv', full_errors, delimiter=',', fmt='%10.20f')


if __name__ == '__main__':
    main()
