import re
import requests
from bs4 import BeautifulSoup as bs

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
}


def get_song_list(query):
    url = f"https://www.melon.com/search/total/index.htm?q={query}"
    response = requests.get(url, headers=headers)
    page = bs(response.content, "lxml")
    song_list = []
    form = page.find("form", id="frm_songList")
    if not form:
        form = page.find("form", id="frm_searchSong")
    if not form:
        form = page.find("form", id="frm_searchArtist")
    if not form:
        return song_list
    datas = form.find("tbody").find_all("tr")
    for data in datas:
        song = data.find("a", class_="fc_gray")
        song_id = song["href"].split(",")[-1].rstrip(");")
        if song:
            song_id = song_id
            song_url = f"https://www.melon.com/song/detail.htm?songId={song_id}"

            response = requests.get(song_url, headers=headers)
            page = bs(response.content, "html.parser")
            artists = page.find("div", class_="artist").find_all(
                "a", class_="artist_name"
            )
            cover_link = page.find("a", class_="image_typeAll").find("img")["src"]
            cover_link = re.sub("/282/", "/512/", cover_link)
            cover_link = re.sub("/80/", "/100/", cover_link)
            song_info = {
                "song_id": song_id,
                "song_title": page.find("div", class_="song_name")
                .get_text(strip=True)
                .lstrip("곡명"),
                "artist": ", ".join([artist["title"] for artist in artists]),
                "album_title": re.sub(
                    "[\\xa0]",
                    " ",
                    page.find("div", class_="meta").find("a").get_text(strip=True),
                ),
                "genre": page.find("div", class_="meta")
                .find_all("dd")[2]
                .get_text(strip=True)
                .split(", ")[0],
                "album_cover": cover_link,
                "public_date": page.find("div", class_="meta")
                .find_all("dd")[1]
                .get_text(strip=True),
                "melon": f"https://www.melon.com/song/lyrics.htm?songId={song_id}"
            }
            song_list.append(song_info)
            song_list = [
                song
                for song in song_list
                if "MR" not in song["song_title"] and "Inst." not in song["song_title"]
            ]
    return song_list


def get_song_info(song_id):
    url = f"https://www.melon.com/song/detail.htm?songId={song_id}"
    response = requests.get(url, headers=headers)
    page = bs(response.content, "html.parser")
    artists = page.find("div", class_="artist").find_all("a", class_="artist_name")
    cover_link = page.find("a", class_="image_typeAll").find("img")["src"]
    cover_link = re.sub("/282/", "/512/", cover_link)
    cover_link = re.sub("/80/", "/100/", cover_link)
    result = {
        "song_id": song_id,
        "song_title": page.find("div", class_="song_name")
        .get_text(strip=True)
        .lstrip("곡명"),
        "artist": ", ".join([artist["title"] for artist in artists]),
        "album_title": re.sub(
            "[\\xa0]",
            " ",
            page.find("div", class_="meta").find("a").get_text(strip=True),
        ),
        "album_cover": cover_link,
        "public_date": page.find("div", class_="meta")
        .find_all("dd")[1]
        .get_text(strip=True),
        "genre": page.find("div", class_="meta")
        .find_all("dd")[2]
        .get_text(strip=True)
        .split(", ")[0],
        "lyrics": page.find(class_="wrap_lyric")
        .get_text(separator="\n", strip=True)
        .rstrip("펼치기 "),
    }

    return result
