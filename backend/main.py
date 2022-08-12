from fastapi import FastAPI, HTTPException, status

app = FastAPI()


@app.get("/")
async def get_songs_info():
    return {"result": "Hello, World!"}
