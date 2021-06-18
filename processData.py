import os
import numpy as np
import pandas as pd
import globalFunctions as func

# init data
spotify_charts = pd.read_csv('testChart.csv')
# drop url column
spotify_charts.drop('URL', axis='columns', inplace=True)
# get all country codes
country_codes = spotify_charts.code.unique()
# drop all non-maximum rows - only max rank of a song in a country is relevant
for row in spotify_charts.iterrows():
    # spotify_charts, row['Track Name'], row['Artist'], row['code'], row['Date']
    # song_rows = func.find_song(spotify_charts, "bla", "bla", "bla", "bla")
    song_rows = func.find_song( spotify_charts, row[1]['Track Name'], row[1]['Artist'], row[1]['code'], row[1]['Date'])

    max_rate = song_rows.iloc[0]['Position']
    max_rate_row = song_rows.iloc[0]
    for song_row in song_rows.iterrows():
        if song_row[1]['Position'] < max_rate:
            max_rate = song_row[1]['Position']
            max_rate_row = song_row
    spotify_charts.drop(song_rows)
    spotify_charts.insert(max_rate_row)

# add bucket columns
for code in country_codes:
    spotify_charts[code + '_isTop1'] = 0
    spotify_charts[code + '_isTop5'] = 0
    spotify_charts[code + '_isTop10'] = 0
    spotify_charts[code + '_isTop25'] = 0
    spotify_charts[code + '_isTop50'] = 0
    spotify_charts[code + '_isTop100'] = 0
    spotify_charts[code + '_isTop150'] = 0
# add data for bucket columns

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(spotify_charts.head())
