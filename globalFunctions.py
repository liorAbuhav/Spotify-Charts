def find_songs(df, song):
    song_rows = df.loc[(df['Track Name'] == song['Track Name']) & (df['code'] == song['code']) & (df['Artist'] == song['Artist'])]
    return song_rows


def song_exists_in_df(df, song):
    return ((df['Track Name'] == song['Track Name']) & (df['code'] == song['code']) & (df['Artist'] == song['Artist'])).any()