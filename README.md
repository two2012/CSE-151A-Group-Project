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
