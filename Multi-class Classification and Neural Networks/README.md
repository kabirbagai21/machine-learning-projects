# Multi-class Classification and Neural Networks

**Objective**

*Part 1* 
Implement one-vs-all logistic regression to recognize hand-written digits from 0 to 9.

*Part 2*
Complete the same task as above this time using the feedforward propogation algorithm for neural networks.

**Data**

 Given 5000 training examples where each training example is a 20 pixel by 20 pixel grayscale image of the digit. 
 Each pixel is represented by a floating point number indicating the grayscale intensity at that location. 
 The 20 by 20 grid of pixels is 'unrolled' into a 400-dimensional vector. Each of these training examples becomes a single row in data matrix X. 
 This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image.
 The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. 
 
**Topics Covered**

- Vectorized Logistic Regression
- One vs. all Classification
- Feed forward propogation and prediction
