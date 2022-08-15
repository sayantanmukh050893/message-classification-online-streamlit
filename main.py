import streamlit as st
import pickle
from util import Util
util = Util()
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
import base64
from PIL import Image
image = Image.open('phishing-banner.png')

main_bg = "phishing-banner.png"
main_bg_ext = "png"

cwd = os.getcwd()
output_model_path = os.path.join(cwd,"model.pkl")

text_vectorizer_path = os.path.join(cwd,"text_vectorizer_path.pkl")

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('phishing-banner.png')

st.markdown("<h1 style='text-align: center; color: grey;'>SPAM Detector</h1>", unsafe_allow_html=True)

st.image(image)

st.markdown("<h3 style='text-align: center; color: grey;'>Enter Message</h3>", unsafe_allow_html=True)


input_message = st.text_area("", key="message", height = 200, max_chars=500)

if st.button('CHECK'):
    model = pickle.load(open(output_model_path,"rb"))
    X = pd.Series(input_message)
    X = X.apply(util.clean_message)
    X = X.str.lower()
    text_vectorizer = pickle.load(open(text_vectorizer_path,'rb'))
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error='replace',vocabulary=text_vectorizer)
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(X))
    y_pred = model.predict(tfidf)
    result = None
    if(y_pred[0]==0):
        result_image = Image.open('you-are-safe.jpg')
        #result = "This is a normal message"
    elif(y_pred[0]==1):
        result_image = Image.open('youve-been-hacked.png')
        #result = "This is a spam message"
    st.image(result_image)
