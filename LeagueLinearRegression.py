# Jonathan Nister
# Advanced Topics (TICS)
# March 9 2020
# This program allows you to perform a linear regression on damage features from high elo league of legends games
# after choosing how many rows to read from the csv (where each row is one game) and whether to take the difference
# between the teams' values for each variable

# This csv includes the sums of objective, tower, and player damage per team, which team won, the lp (rating) of the 
# match, and how long the match took

# Outputs Coefficients of all the variables being looked at in the regression, the graph itself, and statistics
# about how well the regression went

# Import all our libraries
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from statistics import linear_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Use this boolean to easily set whether I want to use new columns that take the sum/difference of the variables I am 
# looking at (objective, tower, and player damage)
# Turns out the model is more accurate when it does this
SUM_COLUMNS = False

# Easily set how many rows to read from the csv
nRowsRead = 60000
# reading in csv taken from kaggle: https://www.kaggle.com/kerneler/starter-league-of-legends-high-elo-270350a6-b
df1 = pd.read_csv(r"C:\Users\jnister\OneDrive - Eastside Preparatory School\Desktop\school files\programming\Advanced Topics\Python Coding\SummedLeagueData.csv", delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'League_of_Legends_Edited_Data.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

# Get the winning team column to use as the target data and then drop it so the regression can't see it in the data
winningTeamCol = df1.iloc[:,8]
df1.drop('t1_win', axis=1, inplace=True)

# Check a quick distribution on a histogram to get an idea of what to expect
winningTeamCol.hist()

# Add new columns which take the difference of the features and drop the ones we don't want to use
# Allows us to get a better idea of how important each feature is since without doing this, we get 
# slightly different coeficients for the same variable on different teams, which doesn't make much sense
if(SUM_COLUMNS):
    df1['ObjectiveDamageDifference'] = df1['t1_TotalObjectiveDamage'] - df1['t2_TotalObjectiveDamage']
    df1['TurretDamageDifference'] = df1['t1_TotalTurretDamage'] - df1['t2_TotalTurretDamage']
    df1['ChampionDamageDifference'] = df1['t1_TotalDamageDealtToChampions'] - df1['t2_TotalDamageDealtToChampions']
    df1.drop('t1_TotalObjectiveDamage', axis=1, inplace=True), df1.drop('t2_TotalObjectiveDamage', axis=1, inplace=True)
    df1.drop('t1_TotalTurretDamage', axis=1, inplace=True), df1.drop('t2_TotalTurretDamage', axis=1, inplace=True)
    df1.drop('t1_TotalDamageDealtToChampions', axis=1, inplace=True), df1.drop('t2_TotalDamageDealtToChampions', axis=1, inplace=True)

# Normalize the data 
# Used https://www.adamsmith.haus/python/answers/how-to-normalize-the-elements-of-a-pandas-dataframe-in-python to learn how to do this
column_maxes = df1.max()
df_max = column_maxes.max()
normalized_df1 = df1 / df_max

# Set the winning team to floats of either 0.0 or 1.0 so that it is continuous rather than categorical
# This way the model can predict something in between as a percent certainty of winning
winningTeamColFloats = winningTeamCol.astype(float)

# Split the data with random state 11
X_train, X_test, y_train, y_test = train_test_split(normalized_df1,
                                                    winningTeamCol,
                                                    random_state=11)

# Create the regression object and fit to the data
linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)

# Look at all of the regression coefficients to get a better idea of what 
# it is using, and which variables are most important in the prediction
if(SUM_COLUMNS):
    print("Game duration: " + f'{linear_regression.coef_[0]}')
    print("Average lp: " + f'{linear_regression.coef_[1]}')
    print("Objective damage: " + f'{linear_regression.coef_[2]}')
    print("Turret damage: " + f'{linear_regression.coef_[3]}')
    print("Champion damage: " + f'{linear_regression.coef_[4]}')
else: 
    print("Team 1 objective damage: " + f'{linear_regression.coef_[0]}')
    print("Team 1 turret damage: " + f'{linear_regression.coef_[1]}')
    print("Team 1 champion damage: " + f'{linear_regression.coef_[2]}')
    print("Team 2 objective damage: " + f'{linear_regression.coef_[3]}')
    print("Team 2 turret damage: " + f'{linear_regression.coef_[4]}')
    print("Team 2 champion damage: " + f'{linear_regression.coef_[5]}')
    print("Game duration: " + f'{linear_regression.coef_[6]}')
    print("Average lp: " + f'{linear_regression.coef_[7]}')

# Make the prediction
predicted = linear_regression.predict(X_test)
expected = y_test

# I used this website to learn about how to use max and min functions: 
# https://stackoverflow.com/questions/19922611/maximum-and-minimum-caps-for-list-values-in-python
# We don't want predictions of over 1.0 or under 0.0 since they represent percentages of 100 and 0% 
# chance of team one winning, respectively. So, curtail predicted values to those values by selecting
# 1.0 if it is smaller than x, and then 0.0 if it is larger than x for every x in the array
predicted = [max(min(x, 1.0), 0.0) for x in predicted]

# Visualize the data to get an idea of how well the regression worked
# Set up the dataframes 
df = pd.DataFrame()
df['Expected'] = pd.Series(expected)
df['Predicted'] = pd.Series(predicted)

# Set up graph using seaborn and matplot, give x and y axes
figure = plt.figure(figsize=(9,9))
axes = sns.scatterplot(data = df, x='Expected', y='Predicted',
                        hue='Predicted', palette='cool', legend=False)
# Set up axes and bounds, display the graph
start = -0.1
end = 1.1
axes.set_xlim(start,end)
axes.set_ylim(start,end)
plt.show()

# Check r squared and mean squared error metrics
print("r squared:")
print(metrics.r2_score(expected, predicted))

print("mean squared error:")
print(metrics.mean_squared_error(expected, predicted))


