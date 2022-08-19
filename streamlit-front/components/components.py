import streamlit as st
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np


def album_cover_component(link):
    poster = Image.open(BytesIO(request.urlopen(link).read()), mode="r")
    st.image(poster)


def song_list_component(area, song_list):
    song_container = area.container()
    for i, song in enumerate(song_list):
        song_container.success(
            f"{song['song_title']} | {song['artist']} | {song['public_date']} | {song['album_title']}"
        )


def song_info_component(song_info):
    st.subheader(
        f"{song_info['song_title']} ({song_info['public_date'].split('.')[0]})"
    )
    st.caption(
        f"{song_info['album_title'] if song_info['album_title'] else '정보가 없습니다.'}, {song_info['public_date'] if song_info['public_date'] else ''}"
    )
    st.write(f"**아티스트**: {song_info['artist'] if song_info['artist'] else '정보가 없습니다.'}")
    st.write(f"**장르**: {song_info['genre'] if song_info['genre'] else '정보가 없습니다.'}")
    st.write("**멜론에서 살펴보기**:")
    st.info(f"{song_info['melon']}")


def lyrics_component(area, lyrics):
    if "가사 준비중" in lyrics:
        area.warning("등록된 가사가 없습니다.")
    else:
        area.code(lyrics)


def summary_component(area, emotion):
    cols = area.columns(3)
    for i, (text, score) in enumerate(zip(emotion[0], emotion[1])):
        cols[i].metric(f"{i + 1}위 예측", text, f"{np.round(score * 100, 1)}%", delta_color="off")


def recommend_component(area, song_list):
    for idx, song_info in enumerate(song_list):
        recommend = area.empty()
        col1, col2, col3 = recommend.columns([0.1, 0.15, 0.4])
        with col1:
            st.metric(
                "추천 지수", f"{idx + 1}번", f"{np.round(song_info['w_score'] * 100, 1)}%"
            )

        with col2:
            try:
                album_cover_component(song_info["album_cover"])
            except:
                album_cover_component(
                    "https://img1.daumcdn.net/thumb/R1280x0.fjpg/?fname=http://t1.daumcdn.net/brunch/service/user/cnoC/image/6n5cXdQLDt3_hJi8PCyvtduleAU.jpg"
                )
                with col3:
                    st.subheader(f"{song_info['song_title']} ({song_info['public_date_DV']})")
                    st.caption(
                        f"{song_info['artist']}, {song_info['album_title']}, {song_info['public_date']}, {song_info['genre']}"
                    )
                    st.write("**멜론에서 살펴보기**:")
                    st.info(f"{song_info['melon']}")
                    area.write("---")
                    st.write("**참고해주세요**:")
                    st.warning(f"해당 노래는 앨범 커버를 받아올 수 없어 보노보노로 대체합니다. 헤헤 😉")
                continue

        with col3:
            st.subheader(f"{song_info['song_title']} ({song_info['public_date_DV']})")
            st.caption(
                f"{song_info['artist']}, {song_info['album_title']}, {song_info['public_date']}, {song_info['genre']}"
            )
            st.write("**멜론에서 살펴보기**:")
            st.info(f"{song_info['melon']}")
        with area.expander("감정 확인하기"):
            emotion_text = [s.strip("'") for s in song_info["emotions"].strip('][').split(', ')]
            emotion_props = [float(f) for f in song_info["probs"].strip('][').split(', ')]
            cols = st.columns(3)
            for i, (text, score) in enumerate(zip(emotion_text, emotion_props)):
                cols[i].metric(f"{i + 1}위 예측", text, f"{np.round(score * 100, 1)}%", delta_color="off")
            # st.write("**유튜브에서 살펴보기**:")
            # st.info(f"{song_info['video_url']}")

            # expander = st.expander("가사보기")
            # expander.code(song_info["lyrics"])
        area.write("---")
