import pandas
import time
from progressbar import progressbar
import gc
import codecs
import os
import json

from pythainlp import word_tokenize,Tokenizer
from pythainlp.corpus import thai_stopwords,thai_words
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from flask import Flask,request,send_file,after_this_request, render_template,redirect,send_from_directory
import numpy as np
import random
import matplotlib

stop_words = list(thai_stopwords()) + list(STOPWORDS) +\
             ["฿","ly","pic","co","th","https","com","youtu","http","www","twitter","html","bit"]
map(lambda stop_words:stop_words.lower(),stop_words)

pythainlp_words = thai_words()
custom_dict = ['โคโรนา','ลุงตู่','โควิด','โคโรน่า','เจลล้างมือ','ขบวนเสด็จ']
dictionary = list(pythainlp_words) + list(custom_dict)
    
tok = Tokenizer(dictionary)

PEOPLE_FOLDER = os.path.join('static', 'wordcloud')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/',methods=['GET'])    
def index():
    return render_template('upload.html')

@app.route('/render', methods=['GET'])
def render():
    print('===============')
    print(request.get_data())
    print('===============')
    body = json.loads(request.get_data())
    text = body['text']  # get texts input
    print('===============')
    print (text)
    print('===============')
    text = tok.word_tokenize(text)
    text2 = ' '.join(text)
    text2 = text2.lower()

    wordcloud = WordCloud(stopwords = stop_words,
                        font_path='THSarabunNew.ttf',
                        min_word_length = 2,
                        relative_scaling = 1.0,
                        min_font_size=1,
                        background_color="black",
                        width=128,
                        height=72,
                        scale=10,
                        font_step=1,
                        collocations=False,
                        colormap = "autumn",
                        regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",
                        margin=2
                        ).generate(text2)
    
    plt.figure(figsize=(1,1))
    plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
    plt.axis("off")

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'wordcloud.png')
    wordcloud.to_file(full_filename)
    
    # @after_this_request
    # def cleanup(response):
    #     os.remove('static/wordcloud/wordcloud.png')
    #     return response

    return send_file(full_filename, mimetype='image/png')

# @app.after_request
# def cleanup(response):
#     os.remove('static/wordcloud/wordcloud.png')
#     return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)