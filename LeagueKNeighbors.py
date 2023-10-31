# Jonathan Nister
# Advanced Topics (TICS)
# March 9 2020
# This program allows you to perform a K-Nearest Neighbors on damage features from high elo league of legends games
# after choosing how many rows to read from the csv (where each row is one game), whether to sum the columns to have 
# each variable appear once per team, and then whether to find the overall difference between the teams for each variable

# The original csv includes tower, objective, and player damage for each player in the game, which team won, the lp (rating)
# of the game, and the duration

# Outputs a feature reduction on the selected features, the first 100 elements of the expected and predicted values
# for each game that the modle predicted, the overall accuracy, a confusion matrix of the results, and the classification report

import matplotlib.pyplot as plt # plotting
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import seaborn as sn

# Change these to choose how much data to read in
SUM_COLUMNS = False
TAKE_DIFFERENCE = False
nRowsRead = 60000

# reading in csv taken from kaggle: https://www.kaggle.com/kerneler/starter-league-of-legends-high-elo-270350a6-b
if(SUM_COLUMNS):
    df1 = pd.read_csv(r"C:\Users\jnister\OneDrive - Eastside Preparatory School\Desktop\school files\programming\Advanced Topics\Python Coding\SummedLeagueData.csv", delimiter=',', nrows = nRowsRead)
else: 
    df1 = pd.read_csv(r"C:\Users\jnister\OneDrive - Eastside Preparatory School\Desktop\school files\programming\Advanced Topics\Python Coding\EditedLeagueData.csv", delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'League_of_Legends_Edited_Data.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

# Used this source to learn about syntax with dataframes in python: 
# https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python#question2

# Get the winning team column before deleting it from the dataframe later
winningTeamCol = df1['t1_win']

if(SUM_COLUMNS and TAKE_DIFFERENCE):
    df1['ObjectiveDamageDifference'] = df1['t1_TotalObjectiveDamage'] - df1['t2_TotalObjectiveDamage']
    df1['TurretDamageDifference'] = df1['t1_TotalTurretDamage'] - df1['t2_TotalTurretDamage']
    df1['ChampionDamageDifference'] = df1['t1_TotalDamageDealtToChampions'] - df1['t2_TotalDamageDealtToChampions']
    df1.drop('t1_TotalObjectiveDamage', axis=1, inplace=True), df1.drop('t2_TotalObjectiveDamage', axis=1, inplace=True)
    df1.drop('t1_TotalTurretDamage', axis=1, inplace=True), df1.drop('t2_TotalTurretDamage', axis=1, inplace=True)
    df1.drop('t1_TotalDamageDealtToChampions', axis=1, inplace=True), df1.drop('t2_TotalDamageDealtToChampions', axis=1, inplace=True)

# Make the feature reduction graph by taking a random 10% split from the original data
sample_df = df1.sample(frac=0.1, random_state=17)
# Reduce the features on the data so we can plot in 2d
tsne = TSNE(n_components=2, random_state=11)
winCol = sample_df['t1_win']
reduced_data = tsne.fit_transform(sample_df)
# Make the plot where the color is which team won
dots = plt.scatter(reduced_data[:,0], reduced_data[:,1], c=winCol)
colorbar = plt.colorbar(dots)
plt.show()

# Drop the columns we don't want
# Don't want to look at lp or game duration in this case because I want to mostly use them so the linear regression
# can perhaps use it to influence its certainty of a team's victory i.e. maybe it can't tell as well if the game goes
# on for a very long time
df1.drop('t1_win', axis=1, inplace=True)
df1.drop('average_lp', axis=1, inplace=True)
df1.drop('gameDuration', axis=1, inplace=True)

# Normalize the data 
# Used https://www.adamsmith.haus/python/answers/how-to-normalize-the-elements-of-a-pandas-dataframe-in-python to learn how to do this
column_maxes = df1.max()
df_max = column_maxes.max()
normalized_df1 = df1 / df_max

# Randomly split the data (75/25 percent split)
X_train, X_test, y_train, y_test = train_test_split(normalized_df1, winningTeamCol, random_state = 10)

# Create and train the K-neighbors model, should make number of dimensions in the graph equal to the number of 
# features that are in the dataframe that was passed in
# Can choose number of neighbors by doing n_neighbors = in this constructor
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)
# Now predict on the last 25% 
predicted = knn.predict(X=X_test)
expected = y_test.tolist()

# Print out the first 100 values from the predicted and expected lists to compare one by one if we like
print("predicted: ", predicted.tolist()[:100])
print("expected: ", expected[:100])

# Check how large the testing split should have been (rounding up or down?)
print("size of testing data split:", len(expected))

# Print the accuracy to the hundredth place
print("accuracy of the model: " + f'{knn.score(X_test, y_test): .2%}')

# Used this source to refresh myself on how to make confusion matrices: 
# https://datatofish.com/confusion-matrix-python/
# Make the confusion matrix to visualize how well our model did
print("our confusion matrix:")
# Basically a table where x axis is what values were expected and y axis is what values we predicted
# Label x and y axis as well
confusion_matrix = pd.crosstab(expected, predicted, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
# Add color
sn.heatmap(confusion_matrix, annot=True, cmap='nipy_spectral_r')
plt.show()

# Print the classification report
print(classification_report(expected, predicted))
