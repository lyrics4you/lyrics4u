import streamlit as st

from components.components import (
    song_info_component,
    summary_component,
    recommend_component,
    album_cover_component,
)
from time import sleep
from utils.scrapper import get_song_list, get_recommends
from utils.utils import make_select_option


def app():
    st.title("노래 가사 기반 추천")
    query = st.text_input("노래 제목과 가수를 입력해주세요. (최대 결과 10개)")
    select_area = st.empty()
    st.write("""---""")
    placeholder = st.empty()
    info_aria = st.empty()
    st.write("""---""")
    summary_area = st.empty()
    lyrics_area = st.empty()

    if not query:
        return placeholder.success("입력을 기다리고 있어요... ")

    placeholder.info(f"👀 노래를 정보를 가져오고있어요...")
    try:
        song_list = get_song_list(query)
    except:
        return placeholder.error("노래를 찾을 수 없습니다. 제목을 다시 확인해주세요.")

    if not song_list:
        return placeholder.error("노래를 찾을 수 없습니다. 제목을 다시 확인해주세요.")

    options = make_select_option(song_list)
    music = select_area.selectbox("노래를 선택해주세요. 👇", options)

    if "개의 노래를 발견했어요!" in music:
        return placeholder.success("노래를 선택하실 때까지 기다리고 있어요...")

    song_info = song_list[int(music.split("번 | ")[0]) - 1]

    placeholder.info("🧐 노래 가사를 분석하고 있어요... (⏱ 10초)")
    sleep(0.5)
    try:
        recommends = get_recommends(song_info["song_id"])
    except:
        placeholder.warning("🧚 저런, 요정이 장난을 쳤나봐요! 다시 한번 시도해주세요. 😮")
        sleep(5)
        return
    placeholder.success("완료")
    sleep(0.5)

    placeholder.empty()

    col1, col2 = info_aria.columns([0.4, 1.5])
    with col1:
        album_cover_component(song_info["album_cover"])
    with col2:
        song_info_component(song_info)

    if isinstance(recommends, str):
        summary_area.metric("첫 번째 예측", "가사없음", "가사 없음", delta_color="off")
    else:
        summary_component(summary_area, recommends["emotions"])

    tab1 = lyrics_area.tabs(["🎙 추천리스트"])[0]

    if isinstance(recommends, str):
        tab1.warning("😭 가사가 존재하지 않아요! 다른 노래를 찾아보세요!")
    else:
        recommend_component(tab1, recommends["recommend"])
