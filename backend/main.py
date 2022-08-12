from fastapi import FastAPI, HTTPException, status
from utils.scrapper import get_song_list

app = FastAPI()


@app.get("/songs")
async def get_songs_info(query: str):
    song_info = get_song_list(query)
    if not song_info:
        raise HTTPException(status_code=404, detail="검색결과가 없습니다.")
    return {"result": song_info}
