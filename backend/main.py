from fastapi import FastAPI, HTTPException
from utils.scrapper import get_song_list, get_song_info
from preprocessor import bert_tokenizer
from predictions import predictions

app = FastAPI()


@app.get("/songs")
async def get_songs_info(query: str):
    song_info = get_song_list(query)
    if not song_info:
        raise HTTPException(status_code=404, detail="검색결과가 없습니다.")
    return {"result": song_info}


@app.get("/preditions")
async def recommends_songs(song_id: str):
    try:
        song = get_song_info(song_id)
        lylics = song["lyrics"]
    except:
        raise HTTPException(status_code=404, detail="검색결과가 없습니다.")
    input_ids, attention_mask = bert_tokenizer(lylics)
    result = predictions(input_ids, attention_mask)[0]
    return {"result": {"predictions": result.tolist()}}
