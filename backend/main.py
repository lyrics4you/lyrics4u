from fastapi import FastAPI, HTTPException
from utils.scrapper import get_song_list
from predictions import Recommends
import logging

app = FastAPI()
recommends = Recommends()


@app.get("/")
async def root():
    return { "result": "Hello World" }


@app.get("/songs")
async def get_songs_info(query: str):
    song_info = get_song_list(query)
    if not song_info:
        raise HTTPException(status_code=404, detail="검색결과가 없습니다.")
    return { "result": song_info }


@app.get("/predictions")
async def recommends_songs(song_id: str):
    try:
        songs = recommends.recommends(song_id)
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=404, detail="검색결과가 없습니다.")
    if isinstance(songs, str):
        return { "result": songs }
    return {"result": { "recommend": songs[0], "emotions": songs[1] } }
