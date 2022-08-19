def make_select_option(song_list):
    options = [f"{len(song_list)}개의 노래를 발견했어요! 🧐"]
    for i, song in enumerate(song_list):
        options.append(
            f"{i + 1}번 | {song['song_title']} | {song['artist']} | {song['public_date']} | {song['album_title']}"
        )

    return options
