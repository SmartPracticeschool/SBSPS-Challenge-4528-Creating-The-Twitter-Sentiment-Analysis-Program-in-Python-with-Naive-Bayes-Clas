#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import *
import matplotlib.pyplot as plt
import seaborn as sns

import warnings


# In[2]:


dataset = pd.read_csv('tweets.csv',encoding="Latin")
dataset.head()


# In[3]:


dataset.columns = ["label","id","date","query","others","tweet"]
dataset.head()


# In[4]:


dataset = dataset.drop(["id","date","query","others"],axis = 1)
dataset.head()


# In[1]:


def cleanTxt(table):
    #put everythin in lowercase
    table=re.sub('[0-9]','',table)
    table = table.replace('RT[\s]+', '')
    table = re.sub('http?://\S+', '', table)
    table = re.sub('https?://\S+', '', table)
    table = table.lower()
    table = re.sub('@[A-Za-z0–9]+', '',table)
    #Replace rt indicating that was a retweet
    #table = table.replace('[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', '')
    #Replace occurences of mentioning @UserNames
    
    punctuations = '''!()-[]{};:"\,<>./?@'#$%^&*_~'''
  
    # traverse the given string and if any punctuation 
    # marks occur replace it with null 
    for x in table.lower(): 
        if x in punctuations: 
            table = table.replace(x, "") 
    
    #Replace links contained in the tweet
    table = table.replace(r'www.[^ ]+', '')
    #remove numbers
    
    #replace special characters and puntuation marks
    table=emoji.demojize(table)
    return table


# In[2]:


from nltk.corpus import (
    wordnet,
    stopwords
)


def in_dict(word):
    if wordnet.synsets(word):
        #if the word is in the dictionary, we'll return True
        return True

def replace_elongated_word(word):
    regex = r'(\w*)(\w+)\2(\w*)'
    repl = r'\1\2\3'    
    if in_dict(word):
        return word
    new_word = re.sub(regex, repl, word)
    if new_word != word:
        return replace_elongated_word(new_word)
    else:
        return new_word


# In[3]:


def detect_elongated_words(row):
    regexrep = r'(\w*)(\w+)(\2)(\w*)'
    words = [''.join(i) for i in re.findall(regexrep, row)]
    for word in words:
        if not in_dict(word):
            row = re.sub(word, replace_elongated_word(word), row)
    return row

def replace_antonyms(word):
    #We get all the lemma for the word
    for syn in wordnet.synsets(word): 
        for lemma in syn.lemmas(): 
            #if the lemma is an antonyms of the word
            if lemma.antonyms(): 
                #we return the antonym
                return lemma.antonyms()[0].name()
    return word


# In[4]:


def handling_negation(row):
    #Tokenize the row
    words = word_tokenize(row)
    speach_tags = ['JJ', 'JJR', 'JJS', 'NN', 'VB', 'VBD', 'VBG', 'VBN', 'VBP']
    #We obtain the type of words that we have in the text, we use the pos_tag function
    tags = nltk.pos_tag(words)
    #Now we ask if we found a negation in the words
    tags_2 = ''
    if "n't" in words and "not" in words:
        tags_2 = tags[min(words.index("n't"), words.index("not")):]
        words_2 = words[min(words.index("n't"), words.index("not")):]
        words = words[:(min(words.index("n't"), words.index("not")))+1]
    elif "n't" in words:
        tags_2 = tags[words.index("n't"):]
        words_2 = words[words.index("n't"):] 
        words = words[:words.index("n't")+1]
    elif "not" in words:
        tags_2 = tags[words.index("not"):]
        words_2 = words[words.index("not"):]
        words = words[:words.index("not")+1] 
        
    for index, word_tag in enumerate(tags_2):
        if word_tag[1] in speach_tags:
            words = words+[replace_antonyms(word_tag[0])]+words_2[index+2:]
            break
            
    return ' '.join(words)

def stop_words(table):
    #We need to remove the stop words
    stop_words_list = stopwords.words('english')
    table = table.lower()
    table = ' '.join([word for word in table.split() if word not in (stop_words_list)])
    return table


# In[5]:


df_cleaned=dataset['tweet'].apply(lambda x:cleanTxt(x))
dataset['tweet'] = df_cleaned



# In[5]:


df_cleaned=[]
def final_clean(x):
    x=cleanTxt(x)
    x=detect_elongated_words(x)
    x=handling_negation(x)
    x=stop_words(x)
    return x


# In[6]:


df_cleaned1 =dataset['tweet'].apply(lambda x:final_clean(x))
dataset['tweet'] = df_cleaned1

dataset.to_csv("cleaned_txt.csv")


# In[7]:


x = dataset.tweet
y = dataset.label

from sklearn.model_selection import train_test_split
Seed = 123
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)


# In[8]:


from textblob import TextBlob
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
tbresult = [TextBlob(i).sentiment.polarity for i in x_validation]
tbpred = [0 if n < 0 else 4 for n in tbresult]
conmat = np.array(confusion_matrix(y_validation, tbpred, labels=[4,0]))
confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
                         columns=['predicted_positive','predicted_negative'])
print ("Accuracy Score: {0:.2f}%".format(accuracy_score(y_validation, tbpred)*100))
print ("-"*80)
print ("Confusion Matrix\n")
print (confusion)
print ("-"*80)
print ("Classification Report\n")
print (classification_report(y_validation, tbpred))


# In[9]:


import pickle
def train_test(pipe, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    model = pipe.fit(x_train, y_train)
    filename = 'finalized_model123.sav'
    pickle.dump(model, open(filename, 'wb'))
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    conf1 = np.array(confusion_matrix(y_test, y_pred, labels=[0,4]))
    conf2 = pd.DataFrame(conf1, index=['Negative', 'Positive'],
                         columns=['Predicted_negative','Predicted_positive'])
    print ("Null Accuracy: {0:.2f}%".format(null_accuracy*100))
    print ("Accuracy Score: {0:.2f}%".format(acc*100))
    if acc > null_accuracy:
        print ("model is {0:.2f}% more accurate than null accuracy".format((acc-null_accuracy)*100))
    elif acc == null_accuracy:
        print ("model has the same accuracy with the null accuracy")
    else:
        print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-acc)*100))
    print ("-"*80)
    print ("Confusion Matrix\n")
    print (conf2)
    print ("-"*80)
    print ("Classification Report\n")
    print (classification_report(y_test, y_pred, target_names=['Negative','Positive']))


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
lr = LogisticRegression(max_iter = 4000)
tg_pipeline = Pipeline([
        ('vectorizer', tfidf),
        ('classifier', lr)
    ])
train_test_and_evaluate(tg_pipeline, x_train, y_train, x_validation, y_validation)


# In[ ]:




