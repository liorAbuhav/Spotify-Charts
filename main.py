import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import globalFunctions
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Read charts data
charts_csv_file = 'charts.csv'
spotify_charts = pd.read_csv(charts_csv_file)

# Removed URL & Streams columns
spotify_charts.drop('URL', axis='columns', inplace=True)
spotify_charts.drop('Streams', axis='columns', inplace=True)

# Convert the date to datetime64
spotify_charts['Date'] = pd.to_datetime(spotify_charts['Date'], format='%Y-%m-%d')

# Filter charts by date (occurred in 2021)
spotify_charts = spotify_charts.loc[(spotify_charts['Date'] >= '2021-01-01')]

# Create max songs data frame
max_songs_df = pd.DataFrame([], columns=spotify_charts.columns)
print("Charts data loaded successfully :)")

# Create data frame for each song & store it in dictionary
songs_dfs_dictionary = {}
for song_index, song_row in tqdm(spotify_charts.iterrows()):
    song_key = f"{song_row['Track Name']}-{song_row['Artist']}-{song_row['code']}"
    if song_key in songs_dfs_dictionary.keys():
        songs_dfs_dictionary[song_key] = songs_dfs_dictionary[song_key].append(song_row)
    else:
        songs_dfs_dictionary[song_key] = pd.DataFrame([], columns=spotify_charts.columns)
        songs_dfs_dictionary[song_key] = songs_dfs_dictionary[song_key].append(song_row)

# For each song data frame find the max position song & push it to max songs data frame
for df_key, df in tqdm(songs_dfs_dictionary.items()):
    max_position_song = globalFunctions.get_df_max_position_song(df)
    max_songs_df = max_songs_df.append(max_position_song)

# Save the data frame into csv - for investigation purposes
max_songs_df.to_csv('max_songs_df.csv', mode='a', header=False)
print("Max ranking songs collected")

# Create rankings labels dictionary
rankings = {1: "_isTop1", 5: "_isTop5", 10: "_isTop10", 25: "_isTop25", 50: "_isTop50", 100: "_isTop100"}
# Get unique country codes
country_codes = max_songs_df.code.unique()

# Create rankings columns per country
for code in country_codes:
    for rank, rank_label in rankings.items():
        max_songs_df[code + rank_label] = 0
print("Created rankings columns")

# Fill ranking columns with ranking by the song position
for chart_index, chart_row in max_songs_df.iterrows():
    for rank, rank_label in rankings.items():
        if int(chart_row['Position']) <= rank:
            max_songs_df.at[chart_index, chart_row['code'] + rank_label] = 1
print("Filled rankings columns")

# Create rankings data frame only with the song rankings (buckets) - will contain combined rankings for each song
rankings_df = pd.DataFrame([], columns=[column for column in max_songs_df.columns if
                                        globalFunctions.is_rank_column(column, rankings)])
print("Created empty rankings data frame")

# Create dirty dict to prevent duplications
dirty_dict = {}

# Fill ranking data frame for each song
rankings_df_row = dict()
for chart_index, chart_row in tqdm(max_songs_df.iterrows()):
    if chart_index in dirty_dict.keys():
        continue

    # Get max rank for each song
    songs = globalFunctions.find_songs(max_songs_df, chart_row)
    for column in rankings_df.columns:
        max_value = globalFunctions.get_max_column_value(column, songs)
        rankings_df_row[column] = max_value
    rankings_df = rankings_df.append(rankings_df_row, ignore_index=True)
    dirty_dict = globalFunctions.set_songs_dirty(dirty_dict, songs)

# Save the data frame into csv - for investigation purposes
rankings_df.to_csv('rankings_df.csv', mode='a', header=False)
print("Filled rankings data frame with songs ratings")

# Create corr matrix
corr_matrix = rankings_df.astype('float64').corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()
print("Created corr matrix heatmap")

# init variables for learning
country_code_to_drop = 'us'
Y_data_column_name = country_code_to_drop + rankings[10]
rankings_dropped_df = rankings_df.copy()

# drop the columns of usa rankings
for rank, rank_label in rankings.items():
    column_to_drop = country_code_to_drop + rank_label
    rankings_dropped_df.drop(column_to_drop, axis='columns', inplace=True)

# split the data to train and test
X_train, X_test, Y_train, Y_test = train_test_split(rankings_dropped_df, rankings_df[Y_data_column_name],
                                                    test_size=0.30, random_state=27)
Y_test = Y_test.astype('int')
Y_train = Y_train.astype('int')

# LogisticRegression
print('start LogisticRegression')
logreg_clf = LogisticRegression()
logreg_clf.fit(X_train, Y_train)

logreg_prediction = logreg_clf.predict(X_test)

logreg_accuracy_score = accuracy_score(logreg_prediction, Y_test)
print(logreg_accuracy_score)
print(confusion_matrix(logreg_prediction, Y_test))
print(classification_report(logreg_prediction, Y_test))
print('end LogisticRegression\n')

# RandomForest
print('start RandomForest')
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, Y_train)

rf_prediction = rf_clf.predict(X_test)

rf_accuracy_score = accuracy_score(rf_prediction, Y_test)
print(rf_accuracy_score)
print(confusion_matrix(rf_prediction, Y_test))
print(classification_report(rf_prediction, Y_test))
print('end RandomForest\n')

# DecisionTree
print('start DecisionTree')
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, Y_train)

dt_prediction = dt_clf.predict(X_test)

dt_accuracy_score = accuracy_score(dt_prediction, Y_test)
print(dt_accuracy_score)
print(confusion_matrix(dt_prediction, Y_test))
print(classification_report(dt_prediction, Y_test))
print('end DecisionTree\n')

# SVC
print('start SVC')
svc_clf = SVC()
svc_clf.fit(X_train, Y_train)

svc_prediction = svc_clf.predict(X_test)

svc_accuracy_score = accuracy_score(svc_prediction, Y_test)
print(svc_accuracy_score)
print(confusion_matrix(svc_prediction, Y_test))
print(classification_report(svc_prediction, Y_test))
print('end SVC')

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(max_songs_df.head())
# max_songs_df.to_csv('heronimo.csv', mode='a', header=False)
