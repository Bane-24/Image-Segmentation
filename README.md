# Colour Image Segmentation using EM Algorithm


---

## Project Overview

This project focuses on segmenting color images using the **Expectation Maximisation (EM) algorithm** with Gaussian Mixture Models (GMM). The segmentation was performed on three distinct images—**coins**, **a man jumping**, and **a tiger**—highlighting the strengths and challenges of the method across varying complexity levels.

---

## Problem Statement

Segmenting images based on color features poses challenges due to varying textures, lighting, and color distributions. This project aimed to demonstrate the effectiveness of the EM algorithm in separating different color segments in images by iteratively refining Gaussian components for optimal segmentation.

---

## Machine Learning Goal

To segment color images into distinct regions based on their color features using Gaussian Mixture Models and the Expectation Maximisation algorithm. The goal was to iteratively cluster pixels into meaningful segments while ensuring model convergence.

---

## Methodology

### Core Steps
1. **Initialization**:
   - Gaussian parameters were initialized for the GMM to represent different color clusters.
   - Reasonable initialization aids faster and more accurate convergence.

2. **Expectation-Maximisation (EM)**:
   - **Expectation Step (E-step)**:
     - Computed the posterior probability of each pixel belonging to a Gaussian component.
   - **Maximization Step (M-step)**:
     - Updated means and covariances of Gaussian components to better fit the pixel data.
   - Iterated until convergence or a maximum number of iterations was reached.

3. **Segmentation**:
   - Pixels were assigned to clusters based on maximum posterior probabilities.
   - Applied **K-means clustering** to adjust cluster centers.
   - Normalized pixel intensities and applied Gaussian smoothing for cleaner segmentation.

---

## Toolkits Used

- **Python Libraries**:
  - `numpy` for data manipulation.
  - `scikit-learn` for K-means clustering.
  - `matplotlib` for visualizing segmentation outputs.

---

## Results

### 1. **Coins Image**:
   - Clear segmentation due to high contrast (copper on a white background).
   - **2 Segments**: Converged quickly with distinct separation.
   - **3 Segments**: Achieved in 15 iterations; minor shadows added as a segment.

### 2. **Jump Image**:
   - Sky and snow segmented into two distinct clusters initially.
   - **3 Segments**: Jacket's dark spots were misclassified as sky.
   - **4 Segments**: Improved segmentation accuracy by accounting for shadows and finer details.

### 3. **Tiger Image**:
   - Complex segmentation due to intricate fur patterns and environmental blending.
   - Increased segments introduced haziness around borders.
   - Highlights the impact of segment count on accuracy and clarity.

---

## Key Insights

1. **Segment Count**:
   - The number of segments significantly affects clarity and convergence.
   - Optimal segment count varies based on image complexity.

2. **Challenges**:
   - Images with intricate patterns or minimal contrast (e.g., tiger fur) are harder to segment effectively.
   - Shadows and fine details require higher segment counts for accuracy.

---

## Future Work

1. **Dynamic Segment Estimation**:
   - Develop a method to dynamically determine the optimal number of segments for each image.
   
2. **Advanced Initialization**:
   - Use pre-trained models or heuristic approaches for better initialization.

3. **Integration with Deep Learning**:
   - Combine EM-based methods with convolutional neural networks (CNNs) for improved performance on complex images.

4. **Handling Intricate Patterns**:
   - Explore advanced algorithms like region-based active contours or adaptive GMMs for more detailed segmentation.

---

## Conclusion

The EM algorithm effectively segments images based on color features. While simpler images with high contrast converge quickly, complex textures like the tiger's fur highlight the limitations of basic clustering methods. By iteratively refining parameters and increasing the segment count, this approach demonstrates its capability to handle varying segmentation challenges.
