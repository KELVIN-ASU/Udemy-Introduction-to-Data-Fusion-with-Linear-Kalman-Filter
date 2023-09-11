#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

def CalculateLeastSquaresSolution(Hmatrix, Ymatrix):
    H = np.array(Hmatrix)
    Y = np.array(Ymatrix)
    H_transpose = np.transpose(H)
    H_transpose_H = np.matmul(H_transpose, H)
    H_transpose_H_inverse = np.linalg.inv(H_transpose_H)
    H_transpose_Y = np.matmul(H_transpose, Y)
    Xmatrix = np.matmul(H_transpose_H_inverse, H_transpose_Y)
    return Xmatrix.flatten()

def CalculateLineOfBestFitSolution(Dataset):
    Hmatrix = [[r, 1] for y, r in Dataset]
    Ymatrix = [[y] for y, _ in Dataset]
    LineParam = CalculateLeastSquaresSolution(Hmatrix, Ymatrix)
    return LineParam

# Generate a sample dataset of temperature measurements at different RPMs
dataset = [(12, 980), (16, 1349), (25, 1867), (31, 2587), (40, 2982), (53, 3874)]

# Calculate the line of best fit parameters using the function
line_params = CalculateLineOfBestFitSolution(dataset)

# Extract the slope and y-intercept from the line parameters
slope = line_params[0]
intercept = line_params[1]

# Create the x values for the line of best fit
x_values = np.array([r for y, r in dataset])

# Calculate the corresponding y values for the line of best fit
y_values = slope * x_values + intercept

# Plot the dataset as a scatter plot
plt.scatter([r for y, r in dataset], [y for y, r in dataset])

# Plot the line of best fit
plt.plot(x_values, y_values, 'r')

# Add labels and title to the plot
plt.xlabel('RPM')
plt.ylabel('Temperature')
plt.title('Line of Best Fit for Temperature vs RPM')

# Display the plot
plt.show()


# In[ ]:




