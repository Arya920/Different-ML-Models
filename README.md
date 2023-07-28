# Machine Learning Models Repository

Welcome to my Machine Learning Models repository! This repository contains the implementation code and documentation for various Machine Learning models and techniques that I have developed during my learning journey at VIT AP 

![ML](https://github.com/Arya920/Different-ML-Models/blob/master/12.%20Images/ML.jpg)

## Overview

In this repository, you'll find a collection of 10 different Machine Learning models. Each model is implemented in Python and comes with detailed documentation to help you understand the underlying concepts and methodologies.

## Pull Requests

[![Pull Requests](https://img.shields.io/github/issues-pr/Arya920/Different-ML-Models)](https://github.com/Arya920/Different-ML-Models/pulls)


## List of Models

Here is a list of the Machine Learning models and their respective file names available in this repository:

1. Find S Algorithm
2. Candidate Elimination Method
3. Decision Tree Classifier on Titanic DataSet
4. Simple & Multiple Linear Regression
5. Support Vector Machine
6. S.V.M Multi Class Classifier
7. Logistic Regression
8. Naive Bayes Classification
9. Forward Propagation Neural Network
10. Random Forest  Vs Decision Tree 

## Model Descriptions

### 1. Find S Algorithm
![ML](https://github.com/Arya920/Different-ML-Models/blob/master/12.%20Images/Find%20S%20algo.jpg)
The Find-S algorithm is a supervised learning algorithm used to find the most specific hypothesis that fits positive training examples. It generalizes the hypothesis by updating attribute-value pairs based on positive instances. It's suitable for simple concepts with binary-valued attributes.


### 2. Candidate Elimination Method
![ML](https://github.com/Arya920/Different-ML-Models/blob/master/12.%20Images/CandidateElimnation.JPG)
The Candidate Elimination algorithm is a supervised learning method used for concept learning. It maintains two sets of hypotheses (most general and most specific) and iteratively refines them based on observed training examples to find the concept that fits the data. It can handle both positive and negative examples and is useful for learning complex concepts and handling noisy data.



### 3. Decision Tree Classifier on Titanic DataSet
![ML](https://github.com/Arya920/Different-ML-Models/blob/master/12.%20Images/DT.png)
1. Data Preprocessing: Handle missing values, convert categorical variables to numerical form.

2. Feature Selection: Choose relevant features.

3. Split Data: Divide the dataset into training and testing sets.

4. Build Decision Tree: Create and train the Decision Tree Classifier.

5. Model Evaluation: Assess the model's performance using metrics like accuracy, precision, recall, and F1-score.

6. Visualization (Optional): Optionally, visualize the decision tree.

7. Predictions: Use the trained model to predict survival for new passengers.


### 4. Simple & Multiple Linear Regression
![ML](https://github.com/Arya920/Different-ML-Models/blob/master/12.%20Images/LR.png)
## Simple Linear Regression

### Overview

Simple Linear Regression establishes a linear relationship between a single independent variable (X) and a dependent variable (y). It assumes that the relationship between the variables can be represented by a straight line equation: y = mx + b.

### Usage

1. Data Preparation: Prepare the dataset with the independent variable (X) and the dependent variable (y).

2. Split Data: Divide the dataset into training and testing sets.

3. Model Training: Fit the linear regression model to the training data.

4. Model Evaluation: Evaluate the model's performance using metrics like Mean Squared Error (MSE) or R-squared (R²).

5. Predictions: Use the trained model to make predictions on new data.

## Multiple Linear Regression

### Overview

Multiple Linear Regression is an extension of Simple Linear Regression that deals with multiple independent variables (X₁, X₂, ..., Xₚ) to predict a dependent variable (y). The relationship between the variables is represented by a linear equation: y = b₀ + b₁X₁ + b₂X₂ + ... + bₚXₚ, where p is the number of independent variables.

### Usage

1. Data Preparation: Prepare the dataset with multiple independent variables (X) and the dependent variable (y).

2. Split Data: Divide the dataset into training and testing sets.

3. Model Training: Fit the multiple linear regression model to the training data.

4. Model Evaluation: Evaluate the model's performance using metrics like Mean Squared Error (MSE) or R-squared (R²).

5. Predictions: Use the trained model to make predictions on new data.



### 5. Support Vector Machine
![ML](https://github.com/Arya920/Different-ML-Models/blob/master/12.%20Images/SVM.png)
SVM is a binary classification algorithm that works by finding the hyperplane that maximizes the margin between two classes. The hyperplane serves as a decision boundary that separates the data points belonging to different classes.

### Usage

1. Data Preparation: Prepare the dataset with the feature matrix (X) and the target vector (y).

2. Split Data: Divide the dataset into training and testing sets.

3. Model Training: Fit the SVM model to the training data.

4. Model Evaluation: Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.

5. Predictions: Use the trained model to make predictions on new data.

## Kernel Trick

SVM can efficiently handle non-linearly separable data by applying the kernel trick. Common kernels used are the Radial Basis Function (RBF) kernel, polynomial kernel, and sigmoid kernel.


### 6. S.V.M Multi Class Classifier

## Overview

SVM as a multi-class classifier extends the binary SVM to handle multiple classes in a one-vs-rest or one-vs-one approach. It works by training multiple binary classifiers, where each classifier distinguishes one class from the rest. The final class label is determined based on the votes or decisions from these binary classifiers.

### Usage

1. Data Preparation: Prepare the dataset with the feature matrix (X) and the target vector (y) with multiple class labels.

2. Split Data: Divide the dataset into training and testing sets.

3. Model Training: Fit the SVM multi-class classifier to the training data.

4. Model Evaluation: Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.

5. Predictions: Use the trained model to make predictions on new data.

## Kernel Trick

SVM can efficiently handle non-linearly separable data by applying the kernel trick. Common kernels used are the Radial Basis Function (RBF) kernel, polynomial kernel, and sigmoid kernel.


### 7. Logistic Regression
![ML](https://github.com/Arya920/Different-ML-Models/blob/master/12.%20Images/LogR.png)
## Overview

Logistic Regression estimates the probability that an instance belongs to a particular class. It models the relationship between the input features (X) and the binary target variable (y) using the logistic function, which outputs probabilities in the range (0, 1).

### Usage

1. Data Preparation: Prepare the dataset with the feature matrix (X) and the binary target vector (y).

2. Split Data: Divide the dataset into training and testing sets.

3. Model Training: Fit the logistic regression model to the training data.

4. Model Evaluation: Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.

5. Predictions: Use the trained model to make predictions on new data.

## Regularization

Logistic Regression can be regularized to prevent overfitting. Common regularization techniques include L1 regularization (Lasso) and L2 regularization (Ridge).


### 8. 9. Naive Bayes Classification
![ML](https://github.com/Arya920/Different-ML-Models/blob/master/12.%20Images/NB.png)
## Overview

Naive Bayes Classifier is based on Bayes' theorem and assumes that the features are conditionally independent given the class label. Despite its simplicity, Naive Bayes often performs surprisingly well in various real-world scenarios.

### Usage

1. Data Preparation: Prepare the dataset with the feature matrix (X) and the target vector (y).

2. Split Data: Divide the dataset into training and testing sets.

3. Model Training: Fit the Naive Bayes model to the training data.

4. Model Evaluation: Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.

5. Predictions: Use the trained model to make predictions on new data.

## Types of Naive Bayes Classifiers

There are different types of Naive Bayes classifiers, including:
- Gaussian Naive Bayes: Used for continuous or real-valued features.
- Multinomial Naive Bayes: Used for discrete feature counts, often used in text classification.
- Bernoulli Naive Bayes: Used for binary features, often used in text classification.


### 10. Forward Propagation Neural Network
![ML](https://github.com/Arya920/Different-ML-Models/blob/master/12.%20Images/FP.png)
## Overview

A Neural Network consists of layers of interconnected neurons, each performing a weighted sum of its inputs, followed by an activation function. Forward Propagation is the process of passing input data through the network to obtain predictions.

### Usage

1. Data Preparation: Prepare the dataset with the feature matrix (X) and the target vector (y).

2. Split Data: Divide the dataset into training and testing sets.

3. Model Architecture: Define the number of layers, number of neurons in each layer, and activation functions.

4. Model Training: Implement the Forward Propagation algorithm and train the network on the training data.

5. Model Evaluation: Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.

6. Predictions: Use the trained model to make predictions on new data.

## Activation Functions

Activation functions introduce non-linearity to the network and play a crucial role in its learning. Common activation functions include Sigmoid, ReLU, Tanh, and Softmax (for multi-class classification).

### 11. 11. Random Forest Vs Decision Tree

## Decision Tree

### Overview

Decision Tree is a simple and interpretable algorithm that recursively splits the data based on the most informative feature to create a tree-like structure. Each internal node represents a decision based on a feature, and each leaf node represents a class label or a regression value.

### Advantages

- Easy to understand and interpret due to the tree-like structure.
- Handles both categorical and numerical data.
- Requires minimal data preprocessing.

### Limitations

- Prone to overfitting, especially on complex datasets.
- Sensitive to small variations in the data.

## Random Forest

### Overview

Random Forest is an ensemble learning technique that builds multiple Decision Trees and combines their predictions to make more accurate and robust predictions. It randomly selects a subset of features and data samples for each tree, ensuring diversity among the trees.

### Advantages

- Reduces overfitting by combining predictions from multiple trees.
- More accurate and stable compared to individual Decision Trees.
- Handles high-dimensional data well.

### Limitations

- Less interpretable than individual Decision Trees.
- Slightly more computationally expensive due to multiple trees.

## Usage

1. Data Preparation: Prepare the dataset with the feature matrix (X) and the target vector (y).

2. Split Data: Divide the dataset into training and testing sets.

3. Model Training: Implement the Decision Tree and Random Forest algorithms and train them on the training data.

4. Model Evaluation: Evaluate the models' performances using metrics like accuracy, precision, recall, and F1-score.

5. Predictions: Use the trained models to make predictions on new data.


## Contribution

Feel free to contribute to this repository by submitting pull requests. Your feedback, suggestions, and improvements are highly appreciated!

Thank you for visiting this repository and exploring the different Machine Learning models. Happy learning!

Author: <span style="color:blue">**Arya Chakraborty**</span>
Contact me here :<span style="color:blue">**aryachakraborty.official@gmail.com**</span>
