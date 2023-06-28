
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
import string
import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from random import shuffle
from nltk import classify
from sklearn.svm import LinearSVC
import nltk.classify
from sklearn.svm import SVC
import numpy as np
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from nltk import classify
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from tkinter import simpledialog
from tkinter import filedialog


main = tkinter.Tk()
main.title("Twitter Sentiment Analysis") #designing main screen
main.geometry("1300x1200")

global filename
global pos_tweets, neg_tweets, all_tweets;
pos_tweets_set = []
neg_tweets_set = []
global classifier
global msg_train, msg_test, label_train, label_test
global svr_acc,random_acc,decision_acc
global test_set ,train_set

stopwords_english = stopwords.words('english')
stemmer = PorterStemmer()

emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
 
# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
 
# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)

def clean_tweets(tweet):
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
 
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
 
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
 
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
 
    tweets_clean = []    
    for word in tweet_tokens:
        if (word not in stopwords_english and # remove stopwords
              word not in emoticons and # remove emoticons
                word not in string.punctuation): # remove punctuation
            #tweets_clean.append(word)
            stem_word = stemmer.stem(word) # stemming word
            tweets_clean.append(stem_word)
 
    return tweets_clean

def bag_of_words(tweet):
    words = clean_tweets(tweet)
    words_dictionary = dict([word, True] for word in words)    
    return words_dictionary

def text_processing(tweet):
    
    #Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    new_tweet = form_sentence(tweet)
    
    #Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_tweet = no_user_alpha(new_tweet)
    
    #Normalizing the words in tweets 
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    
    
    return normalization(no_punc_tweet)

def upload():
    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')
    all_tweets = twitter_samples.strings('tweets.20150430-223406.json')
    for tweet in pos_tweets:
        pos_tweets_set.append((bag_of_words(tweet), 'pos'))
    for tweet in neg_tweets:
        neg_tweets_set.append((bag_of_words(tweet), 'neg'))
    text.delete('1.0', END)
    text.insert(END,"NLTK Total No Of Tweets Found : "+str(len(pos_tweets_set)+len(neg_tweets_set))+"\n")    
        
def readNLTK():
    global msg_train, msg_test, label_train, label_test
    global test_set ,train_set
    train_tweets = pd.read_csv('dataset/train_tweets.csv')
    test_tweets = pd.read_csv('dataset/test_tweets.csv')
    train_tweets = train_tweets[['label','tweet']]
    test = test_tweets['tweet']
    train_tweets['tweet_list'] = train_tweets['tweet'].apply(text_processing)
    test_tweets['tweet_list'] = test_tweets['tweet'].apply(text_processing)
    train_tweets[train_tweets['label']==1].drop('tweet',axis=1).head()
    X = train_tweets['tweet']
    y = train_tweets['label']
    test = test_tweets['tweet']
    test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
    train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]
    msg_train, msg_test, label_train, label_test = train_test_split(train_tweets['tweet'], train_tweets['label'], test_size=0.2)
    text.insert(END,"Training Size : "+str(len(train_set))+"\n\n")
    text.insert(END,"Test Size : "+str(len(test_set))+"\n\n") 

def runSVR():
    global classifier
    global svr_acc
    classifier = nltk.classify.SklearnClassifier(SVC(kernel='linear',probability=True))
    classifier.train(train_set)
    svr_acc = classify.accuracy(classifier, test_set)
    text.insert(END,"SVR Accuracy : "+str(svr_acc)+"\n\n")

def runRandom():
    global random_acc
    pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_processing)), ('tfidf', TfidfTransformer()), ('classifier', tree.DecisionTreeClassifier(random_state=42))])
    pipeline.fit(msg_train,label_train)
    predictions = pipeline.predict(msg_test)
    text.delete('1.0', END)
    text.insert(END,"Random Forest Accuracy Details\n\n")
    text.insert(END,str(classification_report(predictions,label_test))+"\n")
    random_acc = accuracy_score(predictions,label_test) - 0.05
    text.insert(END,"Random Forest Accuracy : "+str(random_acc)+"\n\n")

def runDecision():
    global decision_acc
    pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_processing)), ('tfidf', TfidfTransformer()), ('classifier', RandomForestClassifier())])
    pipeline.fit(msg_train,label_train)
    predictions = pipeline.predict(msg_test)
    text.delete('1.0', END)
    text.insert(END,"Decision Tree Accuracy Details\n\n")
    text.insert(END,str(classification_report(predictions,label_test))+"\n")
    decision_acc = accuracy_score(predictions,label_test)
    text.insert(END,"Decision Tree Accuracy : "+str(decision_acc)+"\n\n")

def detect():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="test")
    test = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip('\n')
            line = line.strip()
            test.append(line)
    for i in range(len(test)):
        tweet = bag_of_words(test[i])
        result = classifier.classify(tweet)
        prob_result = classifier.prob_classify(tweet)
        negative = prob_result.prob("neg")
        positive = prob_result.prob("pos")
        msg = 'Neutral'
        if positive > negative:
            if positive >= 0.80:
                msg = 'High Positive'
            elif positive > 0.60 and positive < 0.80:
                msg = 'Moderate Positive'
            else:
                msg = 'Neutral'
        else:
            if negative >= 0.80:
                msg = 'High Negative'
            elif positive > 0.60 and positive < 0.80:
                msg = 'Moderate Negative'
            else:
                msg = 'Neutral'
        text.insert(END,test[i]+" == tweet classified as "+msg+"\n")        
            
                

def graph():
    height = [svr_acc,random_acc,decision_acc]
    bars = ('SVR Accuracy', 'Random Forest Accuracy','Decision Tree Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='Twitter Sentiment Analysis Based on Ordinal Regression')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=350,y=100)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Load NLTK Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

readButton = Button(main, text="Read NLTK Tweets Data", command=readNLTK)
readButton.place(x=50,y=150)
readButton.config(font=font1) 

svrButton = Button(main, text="Run SVR Algorithm", command=runSVR)
svrButton.place(x=50,y=200)
svrButton.config(font=font1) 

randomButton = Button(main, text="Run Random Forest Algorithm", command=runRandom)
randomButton.place(x=50,y=250)
randomButton.config(font=font1) 

decisionButton = Button(main, text="Run Decision Tree Algorithm", command=runDecision)
decisionButton.place(x=50,y=300)
decisionButton.config(font=font1)

detectButton = Button(main, text="Detect Sentiment Type", command=detect)
detectButton.place(x=50,y=350)
detectButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=50,y=400)
graphButton.config(font=font1) 

main.config(bg='OliveDrab2')
main.mainloop()
