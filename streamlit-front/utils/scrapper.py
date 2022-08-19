# import re
# from bs4 import BeautifulSoup as bs

# from time import sleep

# from bs4 import BeautifulSoup as bs
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st
import requests


@st.cache(max_entries=64, show_spinner=False)
def get_song_list(query):
    url = f"https://music-recommend-359214.du.r.appspot.com/songs?query={query}"
    response = requests.get(url).json().get("result")
    return response


@st.cache(max_entries=64, show_spinner=False)
def get_recommends(song_id):
    url = (
        f"https://music-recommend-359214.du.r.appspot.com/predictions?song_id={song_id}"
    )
    response = requests.get(url).json().get("result")
    return response


# @st.cache(max_entries=128, show_spinner=False)
# def get_youtube_url(query):
#     driver = set_chrome_driver()
#     url = f"https://www.youtube.com/results?search_query={query}"
#     driver.get(url=url)
#     soup = bs(driver.page_source, "html.parser")
#     links = [link for link in soup.select("#title-wrapper > h3 a")][0]
#     return f"https://www.youtube.com{links['href']}"


# @st.experimental_singleton(show_spinner=False)
# def set_chrome_driver():
#     chrome_options = webdriver.ChromeOptions()
#     chrome_options.add_argument("--headless")

#     # headless임을 숨기기 위해서
#     # headless인 경우 Cloudflare 서비스가 동작한다.
#     chrome_options.add_argument(
#         "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
#     )
#     service = Service(ChromeDriverManager().install())
#     driver = webdriver.Chrome(service=service, options=chrome_options)
#     return driver
