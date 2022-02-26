# machine-learning-projects

Objective:

Implements the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition.

Data Used:

There are 5000 training examples in ex4data1.mat, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. 
Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is 'unrolled' 
into a 400-dimensional vector. Each of these training examples becomes a single row in a data matrix X. This gives us a 5000 by 400 matrix X where every row 
is a training example for a handwritten digit image. The second part of the training set is a 5000-dimensional vector y that contains labels for the training set.

Topics Covered:
- Model Representation
- Regularized Cost function for Neural Networks
- Backpropogation algorithm
- Gradient Checking
