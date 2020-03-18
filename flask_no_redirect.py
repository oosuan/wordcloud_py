import pandas
import time
from progressbar import progressbar
import gc
import codecs
import os

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

@app.route('/',methods=['GET','POST'])    
def requests():
    if request.method == 'POST':
        return render(request.form.get('texts'))
    else:
        return render_template('upload.html')

def render(texts):
    print (texts)
    texts = tok.word_tokenize(texts)
    text2 = ' '.join(texts)
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

    wordcloud.to_file('static/wordcloud/wordcloud.png')
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'wordcloud.png')
    
    @after_this_request
    def cleanup(response):
        os.remove('static/wordcloud/wordcloud.png')
        return response

    return render_template("upload.html", wordcloud_image = full_filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)