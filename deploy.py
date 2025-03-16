from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from flask_talisman import Talisman

app = Flask(__name__)


#allowed font from google
csp = {
    'default-src': "'self'",
    'style-src': ["'self'", "https://fonts.googleapis.com"],
    'font-src': ["'self'", "https://fonts.gstatic.com"]
}
Talisman(app, content_security_policy=csp)

#load the model
model = pickle.load(open('new_depresimlmodel.sav', 'rb'))
vectorizer = pickle.load(open('new_vectorizer.sav', 'rb'))


@app.route('/')
def home():
    result = ''
    return render_template('./index.html', **locals())

@app.route('/about')
def about():
    return render_template('./about.html', **locals())

@app.route('/ml', methods=['POST', 'GET'])
def ml():
    #ngambil input
    text_input = request.form.get('text_input','').strip()
    
    #lowercase
    lowercase = text_input.lower()
   
    #remove punct
    import re
    punctuation = re.sub("[^\w\s\d]","",lowercase)
   
    #convert slang
    alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
    alay_dict = alay_dict.rename(columns={0:'original', 1:'replacement'})
    alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
    def normalize_alay(text):
        return " ".join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split()])
    normalize_alay = normalize_alay(punctuation)
   
    #remove stopwords
    nltk.download('punk')
    nltk.download('stopwords')
    text_tokens = word_tokenize(normalize_alay)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    filtered_sentence = (" ").join(tokens_without_sw)
    filtered_sentence = [filtered_sentence]

    #convert input
    text_transformed = vectorizer.transform(filtered_sentence)
    # #cek dulu gaiss
    # array = text_transformed.toarray()

    #result
    result = model.predict(text_transformed)[0]
    if result==0:
        result = "Not Depression"
    else:
        result = "Depression"
    # confidence_score = model.decision_function(text_transformed)
    
    return render_template('./machinelearning.html', **locals())

@app.route('/dl', methods=['POST', 'GET'])
def dl():
    #ngambil input
    text_input = request.form.get('text_input','').strip()
    
    #lowercase
    lowercase = text_input.lower()
   
    #remove punct
    import re
    punctuation = re.sub("[^\w\s\d]","",lowercase)
   
    #convert slang
    alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
    alay_dict = alay_dict.rename(columns={0:'original', 1:'replacement'})
    alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
    def normalize_alay(text):
        return " ".join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split()])
    normalize_alay = normalize_alay(punctuation)
   
    #remove stopwords
    nltk.download('punk')
    nltk.download('stopwords')
    text_tokens = word_tokenize(normalize_alay)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    filtered_sentence = (" ").join(tokens_without_sw)
    filtered_sentence = [filtered_sentence]

    #convert input
    text_transformed = vectorizer.transform(filtered_sentence)
    # #cek dulu gaiss
    # array = text_transformed.toarray()

    #result
    result = model.predict(text_transformed)[0]
    if result==0:
        result = "Not Depression"
    else:
        result = "Depression"
    # confidence_score = model.decision_function(text_transformed)
    
    return render_template('./deeplearning.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)