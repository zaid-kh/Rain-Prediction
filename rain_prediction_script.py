# import the libraries
import numpy as np
import pandas as pd
import seaborn as seab
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
x = "*" * 50

"""
This is a basic machine learning script that uses the sklearn library to predict the Rain for the next day.
The script relies mainly on the pandas library to read the csv file and the sklearn library to
perform the machine learning algorithms, and the matplotlib library to plot the graphs.

consisting of the following parts:

1. Exploratory Data Analysis (EDA)
2. Classification Model 1 (Logistic Regression)
3. Classification Model 2 (Decision Tree)
4. Evaluation of the models

@author: Zaid Khamis
"""

# File reading **********************************************************
# display all columns when using pandas
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# import the dataset from xls file
dataset = pd.read_excel('WeatherData.xls')

# EDA part **************************************************************
# describe() prints out a statistical summary of each column
summary = dataset.describe(include='all')
print(summary)
summary.to_csv('summary.csv')
print(x + "end of summary" + x)

# transform the date column to month
dataset['Date'] = pd.to_datetime(dataset['Date']).dt.month.apply(str)
dataset['Date'] = dataset['Date'].astype(int)
print(dataset['Date'].value_counts())
print(x + "Date transformed to month" + x)

# mapping wind and wind gust direction to range [0,1]

wind_mapping = {'N': 0, 'NNE': 0.0625, 'NE': 0.125, 'ENE': 0.1875, 'E': 0.25, 'ESE': 0.3125, 'SE': 0.375, 'SSE': 0.4375,
                'S': 0.5, 'SSW': 0.5625, 'SW': 0.625, 'WSW': 0.6875, 'W': 0.75, 'WNW': 0.8125, 'NW': 0.875,
                'NNW': 0.9375}

dataset['WindDir9am'] = dataset['WindDir9am'].map(wind_mapping)
dataset['WindDir3pm'] = dataset['WindDir3pm'].map(wind_mapping)
dataset['WindGustDir'] = dataset['WindGustDir'].map(wind_mapping)
print(dataset['WindDir9am'].value_counts())
print(dataset['WindDir3pm'].value_counts())
print(dataset['WindGustDir'].value_counts())

# check for inequality for 'RainToday' and 'RainTomorrow' in the dataset where appropriate
i = len(dataset.columns) - 1
for n in range(dataset.shape[0] - 1):
    if dataset.iloc[n, i] != dataset.iloc[n + 1, i - 1]:
        # if both are nan then skip
        if pd.isna(dataset.iloc[n, i]) and pd.isna(dataset.iloc[n + 1, i - 1]):
            continue
        else:
            print("Elements are not equal:")
            print("Element in row ", n, " and column ", i, ": ", dataset.iloc[n, i])
            print("Element in row ", n + 1, " and column ", i - 1, ": ", dataset.iloc[n + 1, i - 1])
            # replace the value in the column with the value in the previous row
            dataset.iloc[n, i] = dataset.iloc[n + 1, i - 1]
print(x + "End of checking for inequality" + x)

# check for duplicates
print("Number of duplicates in the dataset: ", dataset.duplicated().sum())

# check for nan values
print("Number of nan values in the dataset:\n", dataset.isnull().sum())

# Get the index of rows with NaN values in 'RainToday'
indexNames = dataset[dataset['RainToday'].isnull()].index

# Loop over a copy of the rows with NaN values in 'RainToday'
for i in indexNames.copy():
    # If 'RainTomorrow' in the previous row is not NaN, fill 'RainToday' with the value from 'RainTomorrow'
    if not pd.isna(dataset.iloc[i - 1, 20]):
        dataset.iloc[i, 19] = dataset.iloc[i - 1, 20]
    # If 'RainToday' in the next row is not NaN, fill 'RainTomorrow' with the value from 'RainToday' of the next row
    if not pd.isna(dataset.iloc[i + 1, 19]):
        dataset.iloc[i, 20] = dataset.iloc[i + 1, 19]
    # otherwise drop the row
    else:
        dataset.drop(i, inplace=True)
# check for nan values
print("Number of nan values in the dataset:\n", dataset.isnull().sum())

# drop the rows with nan values in 'RainTomorrow'
dataset.dropna(subset=['RainTomorrow'], inplace=True)

# drop the cloud9am and cloud3pm columns as they have a lot of nan values (about a third of the dataset)
dataset.drop(['Cloud9am', 'Cloud3pm'], axis=1, inplace=True)

# drop location column as it is not needed
dataset.drop(['Location'], axis=1, inplace=True)

# replace the nan values in the dataset with the mode of the column
for column in dataset:
    mode = dataset[column].mode()[0]
    dataset[column].fillna(value=mode, inplace=True)

# check for nan values
print("Number of nan values in the dataset:\n", dataset.isnull().sum())

# feature scaling for the numerical columns
# get the numerical columns
numerical_columns = dataset.select_dtypes(include=np.number).columns
print('numerical_columns: ', numerical_columns.size)
# get the categorical columns
categorical_columns = dataset.select_dtypes(exclude=np.number).columns

# scale the numerical columns to range [0,1] but scaling multiple numeric columns of the same category using the
# min/max of all columns in the category (temperature, pressure, wind speed, etc.)

# temperature
temp_min = dataset['MinTemp'].min()
temp_max = dataset['MaxTemp'].max()
dataset['MinTemp'] = (dataset['MinTemp'] - temp_min) / (temp_max - temp_min)
dataset['MaxTemp'] = (dataset['MaxTemp'] - temp_min) / (temp_max - temp_min)
dataset['Temp9am'] = (dataset['Temp9am'] - temp_min) / (temp_max - temp_min)
dataset['Temp3pm'] = (dataset['Temp3pm'] - temp_min) / (temp_max - temp_min)

# pressure
pressure_min = min([dataset['Pressure9am'].min(), dataset['Pressure3pm'].min()])
pressure_max = max([dataset['Pressure9am'].max(), dataset['Pressure3pm'].max()])
dataset['Pressure9am'] = (dataset['Pressure9am'] - pressure_min) / (pressure_max - pressure_min)
dataset['Pressure3pm'] = (dataset['Pressure3pm'] - pressure_min) / (pressure_max - pressure_min)

# wind and wind gust speed
wind_min = min([dataset['WindSpeed9am'].min(), dataset['WindSpeed3pm'].min(), dataset['WindGustSpeed'].min()])
wind_max = max([dataset['WindSpeed9am'].max(), dataset['WindSpeed3pm'].max(), dataset['WindGustSpeed'].max()])
dataset['WindSpeed9am'] = (dataset['WindSpeed9am'] - wind_min) / (wind_max - wind_min)
dataset['WindSpeed3pm'] = (dataset['WindSpeed3pm'] - wind_min) / (wind_max - wind_min)
dataset['WindGustSpeed'] = (dataset['WindGustSpeed'] - wind_min) / (wind_max - wind_min)

# humidity
humidity_min = min([dataset['Humidity9am'].min(), dataset['Humidity3pm'].min()])
humidity_max = max([dataset['Humidity9am'].max(), dataset['Humidity3pm'].max()])
dataset['Humidity9am'] = (dataset['Humidity9am'] - humidity_min) / (humidity_max - humidity_min)
dataset['Humidity3pm'] = (dataset['Humidity3pm'] - humidity_min) / (humidity_max - humidity_min)

# rainfall
rainfall_min = dataset['Rainfall'].min()
rainfall_max = dataset['Rainfall'].max()
dataset['Rainfall'] = (dataset['Rainfall'] - rainfall_min) / (rainfall_max - rainfall_min)

# check the dataset
print(dataset.info())

# save the dataset
dataset.to_csv('WeatherDataML_ready.csv', index=False)

# review the dataset
print(x)
print(dataset.describe())
print(dataset.columns)
print(dataset['RainTomorrow'].value_counts())
print(dataset['RainToday'].value_counts(normalize=True))
dataset.columns = dataset.columns.str.strip()
dataset['RainToday'] = dataset['RainToday'].replace({'Yes': True, 'No': False})
dataset['RainToday'] = dataset['RainToday'].astype(bool)
dataset['RainTomorrow'] = dataset['RainTomorrow'].replace({'Yes': True, 'No': False})
dataset['RainTomorrow'] = dataset['RainTomorrow'].astype(bool)

# find the correlation between the features
corr = dataset.corr()
print(corr)

# plot the correlation matrix
corr.to_csv("correlation.csv")
seab.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='Reds', annot=True)
plt.show()

# find the correlation between the features and the target
dataset = dataset.drop(['WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1)

corr_target = corr['RainTomorrow']
print(corr_target)
# splitting dataset into training set and test set
x = dataset.drop(['RainTomorrow'], axis=1)
y = dataset['RainTomorrow']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


# training the model
model = LogisticRegression()
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

# confusion matrix
print(confusion_matrix(y_test, y_pred))

# accuracy score 3 decimal places
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))

# classification report
print(classification_report(y_test, y_pred))

# ROC curve and AUC score
# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(x_test)
preds = probs[:, 1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Decision Tree Classifier
# training the model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

# confusion matrix
print(confusion_matrix(y_test, y_pred))

# accuracy score 3 decimal places
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))

# classification report
print(classification_report(y_test, y_pred))

# SVM
# training the model
model = SVC()
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

# confusion matrix
print(confusion_matrix(y_test, y_pred))

# accuracy score 3 decimal places
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))

# classification report
print(classification_report(y_test, y_pred))

