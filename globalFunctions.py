def find_songs_by_country(df, song):
    return df.loc[(df['Track Name'] == song['Track Name']) & (df['code'] == song['code']) & (df['Artist'] == song['Artist'])]


def song_exists_in_df(df, song):
    return ((df['Track Name'] == song['Track Name']) & (df['code'] == song['code']) & (df['Artist'] == song['Artist'])).any()


def find_songs(df, song):
    return df.loc[(df['Track Name'] == song['Track Name']) & (df['Artist'] == song['Artist'])]


def is_rank_column(column: str, rankings: dict):
    for rank, rank_label in rankings.items():
        if rank_label in column:
            return True
    return False


def get_max_column_value(column, songs):
    max_value = 0
    for song_index, song_row in songs.iterrows():
        max_value = max(max_value, song_row[column])
        if max_value == 1:
            break
    return max_value


def set_songs_dirty(dirty_dict, songs):
    for song_index, _ in songs.iterrows():
        dirty_dict[song_index] = song_index
    return dirty_dict


def get_df_max_position_song(songs_df):
    if songs_df.empty:
        return None

    max_position_song = songs_df.iloc[0]
    for song_index, song_row in songs_df.iterrows():
        if song_row['Position'] < max_position_song['Position']:
            max_position_song = song_row
    return max_position_song
