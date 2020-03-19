import time
from progressbar import progressbar
import gc
import os

from pythainlp import word_tokenize,Tokenizer
from pythainlp.corpus import thai_stopwords,thai_words
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import matplotlib
from flask import Flask,request,send_file,after_this_request
import json
#import library ต่างๆทีเราใช้เข้ามา

app = Flask(__name__)

@app.route("/test", methods=["POST"])
def test():
    body = json.loads(request.get_data())
    text = body['text']
    try:
        custom_stopwords = body['custom_stopwords']
    except KeyError:
        custom_stopwords = [""]
    try:
        custom_dict = body['custom_dict']
    except KeyError:
        custom_dict = [""]
    #รับ input จาก user

    stop_words = list(thai_stopwords()) + list(STOPWORDS) + custom_stopwords
    map(lambda stop_words:stop_words.lower(),stop_words)
    #ส่วนนี้คือส่วนที่เราใส่คำที่ห้ามโชว์ขึ้นไปใน wordcloud

    pythainlp_words = thai_words()
    dictionary = list(pythainlp_words) + custom_dict
    #เพิ่มคำที่ไม่มีใน dict ของภาษาไทยหรือภาษาอังกฤษเข้าไปให้เป็นคำเช่นถ้าเรา input "ลุงตู่" จะออกมาเป็น "ลุง","ตู่" แต่ถ้าเราเพิ่ม dict เข้าไป output จะเป็น "ลุงตู่"

    tok = Tokenizer(dictionary)
    #ตั้งตัวแปรเพื่อแยกคำ

    text = tok.word_tokenize(text)
    text = ' '.join(text)
    text = text.lower()
    #ทำการแยกคำ

    wordcloud = WordCloud(stopwords = stop_words,
                        font_path='THSarabunNew.ttf',
                        min_word_length = 2,
                        relative_scaling = 1.0,
                        min_font_size=1,
                        background_color="black",
                        width=800,
                        height=600,
                        scale=10,
                        font_step=1,
                        collocations=False,
                        colormap = "gist_ncar",
                        regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",
                        margin=2
                        ).generate(text)
    #ทำการ generate wordcloud 

    plt.figure(figsize=(16,9))
    plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
    plt.axis("off")
    #ทำการวาง wordcloud
    
    wordcloud.to_file('wordcloud.png')
    gc.collect()
    #เซฟรูปลง server และคลีนแรม

    return send_file('wordcloud.png')
    #return รูปให้ user
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

