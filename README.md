# Introduction

This project focuses on preprocessing and predictive modeling of the Parkinson's Telemonitoring dataset, which contains biomedical voice measurements from individuals with early-stage Parkinson's disease. We chose this dataset to explore how voice data can be used to predict Unified Parkinson's Disease Rating Scale (UPDRS) scores, crucial for tracking disease progression. This approach is compelling as it leverages non-invasive data collection for remote monitoring, potentially transforming patient care. Developing an accurate predictive model can significantly impact the early detection and management of Parkinson's disease, improving patient outcomes and enabling more efficient healthcare delivery.

# CSE-151A-Group-Project: Parkinson's Telemonitoring Data Preprocessing

This README file provides an overview of the preprocessing steps applied to the Parkinson's Telemonitoring dataset, which is available [here](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring).
Here is a link to our [Jupyter notebook](https://colab.research.google.com/drive/1PaSum9sxrsMVCX2jmO0YsWdohvbplU4j?usp=sharing).

## Dataset Description

The dataset comprises biomedical voice measurements from 42 individuals with early-stage Parkinson's disease, collected over six months using a telemonitoring device. The goal is to predict the motor and total UPDRS scores from voice measurements.

## Steps to Preprocess the Data

### 1. Importing Required Libraries

Necessary libraries include numpy, matplotlib, pandas, seaborn, and ucimlrepo.

### 2. Fetching the Dataset

The dataset is fetched using the `ucimlrepo` library, providing the features and targets as pandas DataFrames.

### 3. Combining Features and Targets

Features and targets are converted into pandas DataFrames and combined into a single DataFrame.

### 4. Data Exploration

#### Display the First Few Rows

To get an initial look at the data, the first few rows of the dataset are displayed.

#### Statistical Summary

A statistical summary of the numerical features is generated.

#### Histogram of Target Variables

A histogram is created to visualize the distribution of the target variables.

#### Pairplot for Feature Relationships

Relationships between pairs of data points are examined using a pairplot.

#### Correlation Heatmap

A heatmap is used to display the correlation between columns.

### 5. Feature Selection

Based on the correlation heatmap, highly correlated features within the 'Jitter' and 'Shimmer' sets are identified and dropped to reduce redundancy and multicollinearity. Selected features remain to ensure essential information is retained.

#### Verify Remaining Features

The remaining features are verified to ensure the correct columns are kept.

### 6. Redraw Correlation Heatmap

A new correlation heatmap is visualized for the reduced dataset.

## Conclusion

The preprocessing steps included fetching the data, exploring its structure, visualizing correlations, and selecting representative features to reduce redundancy. This streamlined dataset is now ready for further analysis and model development.

For any questions or further details, please refer to the dataset creators or the associated publications listed in the dataset's metadata.

## First Model : Linear Regression with single feature

We selected the feature 'age' to predict UPDRS with linear regression since based on previous pairplot, 'age' has the highest correlation with UPDRS.

### 1. Training the Model  

The first training model uses linear regression to predict the Total UPDRS based on the age of the patients. The model's fit on the training data is represented by the red line in the graph.  

### 2. Evaluate the Model  

Mean Squared Error for training : 103.303671

Mean Squared Error for test: 104.069142

### 3. Conclusion

The MSE values for both the training and testing datasets are high. However, the MSE values of training and testing are close. This suggests that the model is somewhat underfit that it is too simplistic and does not capture the complexity of the relationship between age and Total UPDRS.  

In order to improve our model, we will incorporate more features. Adding more relevant features could help capture the complexity of the data better. Potential features could include medication information, disease duration, gender, and genetic factors.  

## Second Model : Linear Regression with multiple features

We selected the features 'age', 'HNR', 'RPDE', 'DFA', 'PPE' to predict UPDRS with linear regression since based on previous pairplot, these five features has the highest correlation with UPDRS.

### 1. Training the Model  

The second training model uses linear regression with multiple features to predict the Total UPDRS. The performance of this model is depicted in the graph where blue points represent training data, red points represent test data, and the black dashed line represents the linear fit.  

### 2. Evaluate the Model  

Training MSE: 98.72263258065661

Testing MSE: 98.48770950441289

### 3. Conclusion

Despite incorporating more features, the MSE values for both the training and testing datasets remain high, indicating that the model is still underfit. The scatter of data points around the ideal fit line suggests that even with additional features, the model is not capturing the complexity of the relationship between the predictors and the Total UPDRS.  

We will improve the model by using polynomial regression, which might better capture the relationship between the predictors and Total UPDRS.  

## Third Model : Polynomial Regression with single feature

We selected the features 'age' to predict UPDRS with polynomial regression since based on previous pairplot, this feature has the highest correlation with UPDRS.

### 1. Training the Model  

The third training model uses polynomial regression with different degrees to predict the Total UPDRS based on the age of the patients. The performance of this model for polynomial degrees 2, 3, 4, 5, and 6 is depicted in the graphs where the blue points represent the actual test data, and the red line represents the polynomial fit.  

### 2. Evaluate the Model  

Degree: 2  
Training MSE: 103.22558961899044  
Testing MSE: 104.28521627744276  

Degree: 3  
Training MSE: 102.519105410386  
Testing MSE: 104.14708158965924  

Degree: 4  
Training MSE: 91.77259183842843  
Testing MSE: 93.45164352727815  

Degree: 5  
Training MSE: 90.39710993279027  
Testing MSE: 92.25493432875028  

Degree: 6  
Training MSE: 90.39699985590552  
Testing MSE: 92.25884504149018  


### 3. Conclusion

The polynomial regression models show improved performance compared to the initial linear regression models, with lower MSE values. The MSE values of testing and training are close, indicating that the model is **not overfitting or underfiting** at degree of 5 and 6. However, the MSE stop decreasing at degree 5 and 6. The polynomial fits capture more complexity in the relationship between age and Total UPDRS, as shown by the red curves, but the MSE values are still relatively high, indicating that polynomial regression is still not able to capture the trend of UPDRS.  

## Forth Model : Polynomial Regression with multiple features

We selected the features 'age', 'HNR', 'RPDE', 'DFA', 'PPE' to predict UPDRS with polynomial regression since based on previous pairplot, these five features has the highest correlation with UPDRS.

### 1. Training the Model  

The forth training model uses polynomial regression with different degrees to predict the Total UPDRS based on the multple features of the patients. The performance of this model on degree 2, 3, 4, and 5 is depicted in the graph where blue points represent training data, red points represent test data, and the black dashed line represents the linear fit. 

### 2. Evaluate the Model  

Degree: 2  
Training MSE: 91.30149523550578  
Testing MSE: 92.59453846147005  

Degree: 3  
Training MSE: 75.5321706975659  
Testing MSE: 78.78651480142535  

Degree: 4  
Training MSE: 63.6578849887747  
Testing MSE: 85.76560525648883  

Degree: 5  
Training MSE: 58.66115873795397  
Testing MSE: 65.00506155472495  

### 3. Conclusion

The polynomial regression models show improved performance compared to the previous polynomial regression model and the linear regression models, with lower MSE values. The MSE values of testing and training are close, indicating that the model is **not overfitting or underfiting** at degree of 5. However, the MSE increased suddenly at degree 6. The overall stabilization of MSE at higher degrees indicates diminishing returns from increasing complexity, indicating that polynomial regression is still not able to capture the trend of UPDRS.

We may improve our model by utilizing **neural network**.

## Fifth Model : Neuron Network with single features

We selected the features 'age' to predict UPDRS with a neural network. The neural network is trained with the scaled versions of the feature to ensure better performance and convergence.

### 1. Training the Model  

The fifth model utilizes a neural network with multiple layers and ReLU activation functions to predict the Total UPDRS. The architecture of the neural network consists of:
Input layer with 64 neurons  
Hidden layers with 32 and 16 neurons, respectively  
Output layer with a single neuron to predict the UPDRS value  
The model is trained for 100 epochs with a batch size of 32, using the Adam optimizer and mean squared error as the loss function. The training process includes validation with a 20% validation split.

### 2. Evaluate the Model  

After training, the model is evaluated on the test data to determine its performance. The Mean Squared Error (MSE) values for the training and testing datasets are:

Training MSE: 64.45315195120001  
Testing MSE: 64.2839383791606

### 3. Conclusion

The neural network model shows an improved performance compared to the previous polynomial regression models. The MSE values for both the training and testing datasets are closer, indicating that the model is **not overfitting or underfitting.** The neural network is able to capture more complex relationships between the features and the Total UPDRS, leading to better predictions. However, the MSE values are still relatively high, suggesting that further tuning of the model or the inclusion of additional features might be necessary to capture the trend of UPDRS more accurately.

## Sixth Model : Neuron Network with multiple features

We selected the features 'age', 'HNR', 'RPDE', 'DFA', 'PPE' to predict UPDRS using a neural network. The neural network is trained with the scaled versions of these features to ensure better performance and convergence.

### 1. Training the Model  

The sixth model utilizes a neural network with multiple layers and ReLU activation functions to predict the Total UPDRS. The architecture of the neural network consists of:  
Input layer with 64 neurons  
Hidden layers with 32 and 16 neurons, respectively  
Output layer with a single neuron to predict the UPDRS value  
The model is trained for 100 epochs with a batch size of 32, using the Adam optimizer and mean squared error as the loss function. The training process includes validation with a 20% validation split.

### 2. Evaluate the Model  

After training, the model is evaluated on the test data to determine its performance. The Mean Squared Error (MSE) values for the training and testing datasets are:  

Training MSE: 42.42932937216077  
Testing MSE: 48.43486740770869

### 3. Conclusion

The neural network model shows an improved performance compared to the previous polynomial regression models. The MSE values for both the training and testing datasets are lower, indicating that the model has learned the underlying patterns in the data effectively **without overfitting or underfitting.** The neural network captures more complex relationships between the features and the Total UPDRS, resulting in better predictions.


