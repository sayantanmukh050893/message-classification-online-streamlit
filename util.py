import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from imblearn.combine import SMOTETomek

smote = SMOTETomek()

label_encoder = LabelEncoder()

lemma = WordNetLemmatizer()

class Util:

    def show_column_names(self,dataframe):
        return print("The columns of the dataframe : {}".format([(column) for column in dataframe.columns]))

    def drop_columns(self,dataframe):
        dataframe.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)
        return dataframe

    def rename_columns(self,dataframe,columnNames):
        dataframe.columns = columnNames
        return dataframe

    def clean_message(self,message):
        message = message.translate(str.maketrans('','',string.punctuation))
        words = [lemma.lemmatize(word) for word in message.split() if word.lower() not in stopwords.words("english")]
        return " ".join(words)

    def label_encode(self,dataframe,column):
        dataframe[column] = label_encoder.fit_transform(dataframe[column])
        return dataframe

    def text_features(self,dataframe,column):
        #count_vectorizer = CountVectorizer(analyzer='word',tokenizer=nltk.word_tokenize,stop_words='english',max_features=None,decode_error='replace')
        count_vectorizer = CountVectorizer(decode_error='replace')
        cv = count_vectorizer.fit_transform(dataframe[column])
        return dataframe,cv,count_vectorizer.vocabulary_

    def train_test_split(self,dataframe,tfidf):
        y = dataframe["category"]
        X = tfidf
        X_res,y_res = smote._fit_resample(X,y)
        X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.2,random_state=42)
        return X_train,X_test,y_train,y_test

    def train_model(self,X_train,y_train):
        nb_model = MultinomialNB()
        nb_model.fit(X_train,y_train)
        return nb_model

    def predict(self,X_test,nb_model):
        y_pred = nb_model.predict(X_test)
        return y_pred

    def classify_report(self,y_test,y_pred):
        #y_test = label_encoder.inverse_transform(y_test)
        #y_pred = label_encoder.inverse_transform(y_pred)
        return classification_report(y_test,y_pred)
