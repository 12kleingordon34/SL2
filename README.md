# Supervised Learning -- CW2
Code by Daniel Manela and Daniel Trent (T1, 2019/20)

## Code Summary
The codebase falls roughly into groups. The algorithm classes themselves, the main files used to solve the CW problems, and accompanying utility files to fascilitate our solutions. Requires scipy, sklearn and numpy to run.

### Algorithms
* `kernels.py`: Contains the kernels 
* `knn.py`: 1NN and weighted kNN classifiers
* `least_squares.py`: Least Squares Regressor algorithm
* `logistic_regression.py`: Logistic Regressor 1-vs-all classifier
* `perceptron.py`: Classes for regular, kernelised, 1-vs-all and 1-vs-1 kernelised perceptron
* `winnow.py`: Winnow Classifer algorithm
* `lrtest.py`: Obsolete logistic regressor classifiers (not used for investigation)
* `lrtest2.py`: Obsolete logistic regressor classifiers (not used for investigation)

### Main files
These main files contains the code to run the algorithms
* `main_gauss.py`: Gaussian 1-vs-all Perceptron (Part 1)
* `main_knn.py`: Weighted kNN (Part 1_
* `main_lr_test.py`: Logistic Regressor (Part 1)
* `main_poly.py`: Polynomial 1-vs-all Perceptron (Part 1)
* `main_poly_onevsone.py`: Polynomial 1-vs-1 Perceptron
* `main_pt2.py`: Solutions to Part 2 (Part 2)

### Utilities
* `pattern_generators.py`: 
* `utilities.py`: Contains functions for cross validation and data processing tasks
* `onevsonepairs.csv`: Contains mapping between the classes required for the 1-vs-1 perceptron classifier. Each classifier aims to distinguish column 1 from column 2.
