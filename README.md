# Supervised Learning -- CW2
Initialisation

## Comments on Part I Q4:

Things to consider
* Count number of times each image is predicted incorrectly
* select the incorrect images with the largest distance from the hyperplane
* Combine both methods -- select the data which has been most poorly predicted with the largest hyperplane distance
* Keep running epochs until only 5 items are incorrectly predicted (may never happen -- to be studied further)

Multiclass classification methods:
One against all: 
  -- requires multiple training sessions
  -- slower prediction
  
  
Classification methods - Not perceptron:

SVM -- hard to implement, 
Multiclass Logistic Regression
Random Forrest - (maybe not a good idea)
Neural Net
Nearest Neighbours
Naive Bayes
