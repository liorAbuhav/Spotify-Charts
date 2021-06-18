import pandas as pd
import globalFunctions

if __name__ == '__main__':
    spotify_charts = pd.read_csv('smallCharts.csv')
    spotify_charts.drop('URL', axis='columns', inplace=True)
    max_songs_df = pd.read_csv('max_songs_postition.csv')

    for chart_index, chart_row in spotify_charts.iterrows():
        song_rows = globalFunctions.find_songs(spotify_charts, chart_row)
        max_position_song = chart_row

        for song_index, song_row in song_rows.iterrows():
            if song_row['Position'] < max_position_song['Position']:
                max_position_song = song_row

        if not globalFunctions.song_exists_in_df(max_songs_df, max_position_song):
            max_songs_df = max_songs_df.append(max_position_song)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(max_songs_df.head())

    # max_songs_df.to_csv('heronimo.csv', mode='a', header=False)


