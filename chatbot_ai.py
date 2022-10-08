pip install nltk

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading popular packages
nltk.download('punkt') 
nltk.download('wordnet')

import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Upload Chatbot corpus
f=open('Chatbot_corpus.txt','r',errors = 'ignore')
raw=f.read()

# Converts to lowercase
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Lemmatization
lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["Hi\n", "Hey\n", "Hi there\n", "Hello\n", "I am glad! You are talking to me\n"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

feeling_well_input=["i","am","not","feeling","well"]
feeling_well_response=["I am really sorry to hear that😞!\nKindly describe your Symptoms?\n"]
def well(sentence):
  for ord in sentence.split():
    if ord.lower() in feeling_well_input:
      return random.choice(feeling_well_response)

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you\n"  
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

flag=True
print("\033[1;36;48m\n")
print("   Welcome I AM DOCBOT 👩‍⚕️  \n ")
print("Hi there!😀 \nHow can i help You?\n")
while(flag==True):
    user_response=print("ME:")
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("DOCBOT: You are welcome..\n")
        elif(greeting(user_response)!=None):
                print("\nDOCBOT: "+greeting(user_response))
        elif(well(user_response)!=None):
                print("\nDOCBOT: "+well(user_response))
        else:
                print("\nDOCBOT: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("DOCBOT: Bye!Have a nice day!\n")
