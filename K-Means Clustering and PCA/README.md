# K-Means Clustering and PCA

**Objective**

*Part 1:*

Implement the K-means clustering algorithm and apply it to compress an image. 

*Part 2:*

Use principal component analysis to find a low-dimensional representation of face images.

**Data**

*Part 1:*
Given an image, use the K-means algorithm to select the 16 colors that will be used to represent the compressed image. 
Treat every pixel in the original image as a data example and use the K-means algorithm to find the 16 colors that best group (cluster) the pixels in the 3-dimensional RGB space. 
Once the centroids are computed, use the 16 colors to replace the pixels in the original image.

*Part 2:*

Run PCA on face images to see how it can be used in practice for dimension reduction. 
The dataset contains a database of face images, each 32×32 in grayscale. 
Each row of X corresponds to one face image (a row vector of length 1024). The code in this section will load and visualize the first 100 of these face images

**Topics Covered**

- K-Means Clustering/Algorithm
- Image Compression with K-Means
- Principal Component Analysis (PCA)
- Dimensionality Reduction
