# Importing the required Libraries
import pandas as pd
import re
import nltk 
import spacy
import string
import pickle as pk

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Now, loading the pickle files 

# Loading the Count Vectorizer
count_vector = pk.load(open('pickle_file/count_vector.pkl','rb'))            

# Loading the TFIDF Transformer
tfidf_transformer = pk.load(open('pickle_file/tfidf_transformer.pkl','rb')) 

# Loading the Classification Model
model = pk.load(open('pickle_file/model.pkl','rb'))         

# Loading the Classification Model
product_df = pk.load(open('pickle_file/lemmatized_review_df.pkl','rb'))    

# Loading the Spacy Model
nlp = spacy.load('en_core_web_sm',disable=['ner','parser'])

# Loading the Product Data
#product_df = pd.read_csv('sample30.csv',sep=",")


# This function will remove special characters from the text given as input
def func_remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

# This function converts all characters to lowercase from list of tokenized words
def func_set_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

#Remove punctuation from list of tokenized words
def func_remove_punc_and_splchars(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = func_remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words

stopword_list= stopwords.words('english')

#This function removes stop words from list of tokenized words
def func_remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopword_list:
            new_words.append(word)
    return new_words

#This function stems words from the list of tokenized words
def stem_words(words):
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

#This function Lemmatizes verbs from the list of tokenized words
def func_lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

#This function normalizes the list of words
def func_normalize(words):
    words = func_set_lowercase(words)
    words = func_remove_punc_and_splchars(words)
    words = func_remove_stopwords(words)
    return words

#This function lemmatizes the list of words
def func_lemmatize(words):
    lemmas = func_lemmatize_verbs(words)
    return lemmas

#This function is predicting the sentiment of the product review comments
def model_predict(text):
    word_vector = count_vector.transform(text)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    output = model.predict(tfidf_vector)
    return output

#This function is normalizing and lemmatizing the input text
def normalize_and_lemmaize(input_text):
    input_text = func_remove_special_characters(input_text)
    words = nltk.word_tokenize(input_text)
    words = func_normalize(words)
    lemmas = func_lemmatize(words)
    return ' '.join(lemmas)

#This function will recommend the top 20 products based on the sentiment from model
def recommend_products(user_name):
    recommend_matrix = pk.load(open('pickle_file/user_final_rating.pkl','rb'))
    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[0:20])
    product_frame = product_df[product_df['name'].isin(product_list.index.tolist())]
    output_df = product_frame[['name','lemmatized_text']] #,'reviews_text'
    #output_df['lemmatized_text'] = output_df['reviews_text'].map(lambda text: normalize_and_lemmaize(text))
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
    #output_df.drop('reviews_text',axis=1,inplace=True)
    
    #output_df.to_csv('recommended_products.csv', index=False)
    #print("Output saved to recommended_products.csv")
    return output_df

#This function will recommend the top 5 products based on the sentiment from model
def reco_prod_5(df):
    total_product=df.groupby(['name']).agg('count')
    print("total_product : \n")
    print(total_product)
    rec_df = df.groupby(['name','predicted_sentiment']).agg('count')
    print("rec_df : \n")
    print(rec_df)
    rec_df=rec_df.reset_index()
    print("rec_df reset : \n")
    print(rec_df)
    merge_df = pd.merge(rec_df,total_product['lemmatized_text'],on='name')
    print("merge_df : \n")
    print(merge_df)
    merge_df['%percentage'] = (merge_df['lemmatized_text_x']/merge_df['lemmatized_text_y'])*100
    merge_df=merge_df.sort_values(ascending=False,by='%percentage')
    print("merge_df sorted : \n")
    print(merge_df)
    output_products = pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] ==  1][:5])
    print("output_products : \n")
    print(output_products)
    return output_products

#reco_prod_20 = recommend_products("samantha")
#print(reco_prod_20.head())
#get_top5 = reco_prod_5(reco_prod_20)