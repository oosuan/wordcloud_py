import pandas
import time
from progressbar import progressbar
import gc
import codecs

from pythainlp import word_tokenize,Tokenizer
from pythainlp.corpus import thai_stopwords,thai_words
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from flask import Flask,request,send_file,after_this_request, render_template,redirect
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

class main_flask():
    app = Flask(__name__)

    @app.route('/',methods=['GET'])
    def upload_file():
        return render_template('upload_text_redirect.html')

    @app.route("/test", methods=['POST'])
    def test():
        texts = request.form['texts']
        texts = tok.word_tokenize(texts)
        text2 = ' '.join(texts)
        text2 = text2.lower()

        wordcloud = WordCloud(stopwords = stop_words,
                            font_path='THSarabunNew.ttf',
                            min_word_length = 2,
                            relative_scaling = 1.0,
                            min_font_size=1,
                            background_color="black",
                            width=1280,
                            height=720,
                            scale=10,
                            font_step=1,
                            collocations=False,
                            colormap = "autumn",
                            regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",
                            margin=2
                            ).generate(text2)
        
        plt.figure(figsize=(16,9))
        plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
        plt.axis("off")
    
        wordcloud.to_file('wordcloud.png')
        gc.collect()

        return send_file('wordcloud.png')
            
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000)

    