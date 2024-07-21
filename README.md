# CSE-151A-Group-Project: Parkinson's Telemonitoring Data Preprocessing

This README file provides an overview of the preprocessing steps applied to the Parkinson's Telemonitoring dataset, which is available [here](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring).
Here is a link to our [Jupyter notebook](https://colab.research.google.com/drive/1xcPzwsV2chVULtxJ78fbrL-X01Hw_6-6?usp=sharing).

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

Training MSE: 97.96295980665154  

Testing MSE: 98.0749777435734  

### 3. Conclusion

The MSE values for both the training and testing datasets are high. However, the MSE values of training and testing are close. This suggests that the model is somewhat underfit that it is too simplistic and does not capture the complexity of the relationship between age and Total UPDRS.  

In order to improve our model, we will incorporate more features. Adding more relevant features could help capture the complexity of the data better. Potential features could include medication information, disease duration, gender, and genetic factors.  

## Second Model : Linear Regression with multiple features

We selected the features age', 'test_time','HNR', 'RPDE', 'DFA', 'PPE' to predict UPDRS with linear regression since based on previous pairplot, these five features has the highest correlation with UPDRS.

### 1. Training the Model  

The second training model uses linear regression with multiple features to predict the Total UPDRS. The performance of this model is depicted in the graph where blue points represent training data, red points represent test data, and the black dashed line represents the linear fit.  

### 2. Evaluate the Model  

Training MSE: 97.96295980665154  

Testing MSE: 98.0749777435734  

### 3. Conclusion

Despite incorporating more features, the MSE values for both the training and testing datasets remain high, indicating that the model is still underfit. The scatter of data points around the ideal fit line suggests that even with additional features, the model is not capturing the complexity of the relationship between the predictors and the Total UPDRS.  

We will improve the model by using polynomial regression, which might better capture the relationship between the predictors and Total UPDRS.  

