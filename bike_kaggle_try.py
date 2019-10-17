#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:56:35 2019

@author: hcamphausen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('train.csv', index_col=0, parse_dates=True)


plt.figure(figsize=(8,4))
sns.boxplot(x='season', y='count', data=df)

plt.figure(figsize=(16,8))
sns.regplot(x='windspeed', y='count', data=df)

plt.figure(figsize=(30,8))
sns.boxplot(x='humidity', y='count', data=df)

plt.figure(figsize=(16,8))
sns.regplot(x='humidity', y='count', data=df)


# FEATURE ENGINEERING

def feature_engineering(df):
    # drop columns that test data does not have
    if 'casual' and 'registered' in df.columns:
        df.drop(['casual', 'registered'], axis=1, inplace=True)
    else:
        pass
     
    # one-hot encoding season
    one_hot_encoded = pd.get_dummies(df['season']) 
    df = pd.concat((df, one_hot_encoded), axis=1)
    df.rename(columns={1: "spring", 2: "summer", 3: "fall", 4: "winter"}, inplace=True)
    df.drop(['season'], axis = 1, inplace=True)
    
    #weather - 1: Clear, 2Few clouds, 3Partly cloudy, 4Partly cloudy
    one_hot_encoded_2 = pd.get_dummies(df['weather'])
    df = pd.concat((df, one_hot_encoded_2), axis=1)
    df.rename(columns={1:"clear",2:"few_clouds",3:"partly_cloudy",4:"cloudy"}, inplace=True)
    df.drop(['cloudy'], axis=1, inplace=True)
    df.drop(['weather'], axis=1, inplace=True)
    
    # log count - remember to exponent count for test predictions
    df['count_log'] = np.log1p(df['count'])
    df.drop(['count'], axis=1, inplace=True)
    
    # add hour column
    df['hour'] = df.index.hour
    #df['year'] = df.index.year
    #df['month'] = df.index.month
    #df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    
    # one hot encoding hour and dayof week

    one_hot_encoded_day_of_week = pd.get_dummies(df['dayofweek'])
    df = pd.concat((df, one_hot_encoded_day_of_week), axis=1)
    df.rename(columns={0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}, inplace=True)
    df.drop(['dayofweek'], axis=1, inplace=True)
    
    one_hot_encoded_hour = pd.get_dummies(df['hour'])
    df = pd.concat((df, one_hot_encoded_hour), axis=1)
    df.drop(['hour'], axis=1, inplace=True)
    
    # drop temperatures
    df.drop(['temp','atemp'], axis=1, inplace=True)
    
    # drop holiday as super small dataset for 1
    df.drop(['holiday'], axis = 1, inplace=True)
    
    #scaling data
    #scaler = MinMaxScaler()
    #df[['humidity', 'windspeed']] = scaler.fit_transform(df[['humidity', 'windspeed']])
    
    #drop windspeed as weird measurings
    df.drop(['windspeed'], axis=1, inplace=True)
    
    #drop humidity as weird measurings
    df.drop(['humidity'], axis=1, inplace=True)
    
    return df


df_train = feature_engineering(df)

df_train.head()

plt.figure(figsize=(8,4))
sns.heatmap(df_train.corr(), cmap='Oranges', annot=True)

corr = df_train[df_train.columns[1:]].corr()['count_log'][:]
df_corr = pd.DataFrame(data=corr)

df_corr.plot.bar()

# SPLITTING TRAIN AND TEST DATA SET

df_train.columns

# suffle : df_train = df_train.sample(len(df_train))

X = df_train[['workingday',        'spring',        'summer',          'fall',
              'winter',         'clear',    'few_clouds', 'partly_cloudy',
              'Monday',       'Tuesday',     'Wednesday',
            'Thursday',        'Friday',      'Saturday',        'Sunday',
                     0,               1,               2,               3,
                     4,               5,               6,               7,
                     8,               9,              10,              11,
                    12,              13,              14,              15,
                    16,              17,              18,              19,
                    20,              21,              22,              23
             ]]
y = df_train['count_log']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5)


pipeline = make_pipeline(
                         MinMaxScaler(),         # transform
                         Ridge(alpha=0.0)        # predict
)

# Hyperparameter Optimization

g = GridSearchCV(pipeline, cv=5, param_grid={
    'ridge__alpha': [0.0, 0.1, 0.01, 0.001]
})

#fitting model 
    
g.fit(Xtrain,ytrain)

# train vs. test scores

train_score = g.score(Xtrain, ytrain)
print("This is my training score: " + str(train_score))

test_score = g.score(Xtest, ytest)
print("This is my testing score: " + str(test_score))

y_pred = g.predict(Xtrain)
mse_train = mean_squared_error(ytrain, y_pred)
print("This is my train MSE: " + str(mse_train))

y_predtest = g.predict(Xtest)
mse_test = mean_squared_error(ytest, y_predtest)
print("This is my test MSE: " + str(mse_test))

y2 = np.expm1(ytrain)
y2pred = np.expm1(y_pred)
mae_train = mean_absolute_error(y2, y2pred)
print("This is my train MAE: " + str(mae_train))

y3 = np.expm1(ytest)
y3pred = np.expm1(y_predtest)
mae_test = mean_absolute_error(y3, y3pred)
print("This is my test MAE: " + str(mae_test))

#######################

#CHECKING ASSUMPTIONS

sns.jointplot(ytrain,y_pred, kind="reg")


# Autocorrelation - Durbin Watson (aim at 2)

from statsmodels.stats.stattools import durbin_watson

print(durbin_watson(ytrain-y_pred, axis=0))

# Sum of residuals = 0?

residuals = ytrain - y_pred
residuals = np.array(residuals)
sum_res = residuals.mean().round(5)
print("The sum of my residuals is :" + str(sum_res))

# Normal distribution of residuals?

plt.hist(residuals, bins=20)

# Change in variance - homoscedasticity / heteroscedasticity

import statsmodels.api as sm

pl = sm.qqplot(residuals, line='r')

# Are features linearly independent?

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

vifs = [VIF(df_train.values, i) for i, colname in enumerate(df_train)]
s = pd.Series(vifs,index=df_train.columns)
s.plot.bar()



##########################

# Kaggle test set

kaggle_test = pd.read_csv('test.csv', parse_dates=True,index_col=0)

def feature_engineering_test(df):
    # drop columns that test data does not have
    if 'casual' and 'registered' in df.columns:
        df.drop(['casual', 'registered'], axis=1, inplace=True)
    else:
        pass
     
    # one-hot encoding season
    one_hot_encoded = pd.get_dummies(df['season']) 
    df = pd.concat((df, one_hot_encoded), axis=1)
    df.rename(columns={1: "spring", 2: "summer", 3: "fall", 4: "winter"}, inplace=True)
    df.drop(['season'], axis = 1, inplace=True)
    
    #weather - 1: Clear, 2Few clouds, 3Partly cloudy, 4Partly cloudy
    one_hot_encoded_2 = pd.get_dummies(df['weather'])
    df = pd.concat((df, one_hot_encoded_2), axis=1)
    df.rename(columns={1:"clear",2:"few_clouds",3:"partly_cloudy",4:"cloudy"}, inplace=True)
    df.drop(['cloudy'], axis=1, inplace=True)
    df.drop(['weather'], axis=1, inplace=True)
    
    # add hour column
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    # one hot encoding hour and dayof week

    one_hot_encoded_day_of_week = pd.get_dummies(df['dayofweek'])
    df = pd.concat((df, one_hot_encoded_day_of_week), axis=1)
    df.rename(columns={0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}, inplace=True)
    df.drop(['dayofweek'], axis=1, inplace=True)
    
    one_hot_encoded_hour = pd.get_dummies(df['hour'])
    df = pd.concat((df, one_hot_encoded_hour), axis=1)
    df.drop(['hour'], axis=1, inplace=True)
    
    # drop temperatures
    df.drop(['temp','atemp'], axis=1, inplace=True)
    
    # drop holiday as super small dataset for 1
    df.drop(['holiday'], axis = 1, inplace=True)
    
    #drop windspeed as weird measurings
    df.drop(['windspeed'], axis=1, inplace=True)
    
    #drop humidity as weird measurings
    df.drop(['humidity'], axis=1, inplace=True)
    
    return df


kaggle_test_set = feature_engineering_test(kaggle_test)

predictions = g.predict(kaggle_test_set)

output = pd.DataFrame()
output['datetime'] = kaggle_test_set.index
output['count'] = np.expm1(predictions)
output.to_csv('linear_regression_kaggle_02.csv', index=False)