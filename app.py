# import library ต่างๆ ทีเราใช้เข้ามา
from pythainlp.corpus import thai_stopwords, thai_words
from pythainlp import Tokenizer
from wordcloud import WordCloud, STOPWORDS
from flask import Flask, request, send_file
import json
import gc
import os


def word_preparing(words):
    '''ทำ list ของ words ให้ unique และทำ word เป็น lower-case'''
    words = set(words) # drop duplicate word, unique word
    return [word.lower() for word in words]  # to lower case


# get dictionary and stopword corpus
DEFAULT_DICT = list(thai_words())
DEFAULT_STOPWORLS = list(thai_stopwords()) + list(STOPWORDS)

# word preparing
DEFAULT_DICT = word_preparing(DEFAULT_DICT)
DEFAULT_STOPWORLS = word_preparing(DEFAULT_STOPWORLS)

IMAGE_FILE = "wordcloud.png"  # ชื่อไฟล์ที่จะเซฟรูป wordcloud

app = Flask(__name__)


@app.route("/wordcloud", methods=["POST"])
def gen_worldcloud():
    # get text, custom_stopwords and custom_dict
    body = json.loads(request.get_data())
    text = body['text']
    custom_stopwords = body['custom_stopwords'] if 'custom_stopwords' in body else []
    custom_dict = body['custom_dict'] if 'custom_dict' in body else []

    # เพิ่มคำที่ไม่มีใน dict ของภาษาไทยหรือภาษาอังกฤษเข้าไปให้เป็นคำเช่นถ้าเรา input "ลุงตู่" 
    # จะออกมาเป็น "ลุง","ตู่" แต่ถ้าเราเพิ่ม dict เข้าไป output จะเป็น "ลุงตู่"
    stop_words =  DEFAULT_STOPWORLS + custom_stopwords
    dictionary = DEFAULT_DICT + custom_dict

    # word preparing
    stop_words = word_preparing(stop_words)
    dictionary = word_preparing(dictionary)
    
    # ทำการแยกคำ/ตัดคำ
    tok = Tokenizer(dictionary)
    tokens = tok.word_tokenize(text)
    text = ' '.join(tokens)  # convert tokens เป็น string 
    text = text.lower()  # ทำเป็น lower-case
    
    # ทำการ generate wordcloud 
    wordcloud = WordCloud(stopwords=stop_words,
                        font_path='THSarabunNew.ttf',
                        min_word_length=2,
                        relative_scaling=1.0,
                        min_font_size=1,
                        background_color="black",
                        width=800,
                        height=600,
                        scale=10,
                        font_step=1,
                        collocations=False,
                        colormap="gist_ncar",
                        regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",
                        margin=2
                        ).generate(text)
    
    # เซฟรูปและคลีนแรม
    wordcloud.to_file(IMAGE_FILE)
    gc.collect()

    # response รูป
    return send_file(IMAGE_FILE)


@app.after_request
def cleanup(response):
    # ลบไฟล์หลังจาก response แล้ว
    os.remove(IMAGE_FILE)
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
