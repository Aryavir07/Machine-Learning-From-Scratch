#!/usr/bin/env python
# coding: utf-8

# ### Create a NLP Pipeline to clean reviews data
# - Load input file and read Reviews
# - Tokenize
# - Remove Stopwords
# - Preform Stemming
# - Write cleaned data to output file  

# In[1]:


import numpy as np


# ### NLTK
# - Tokenisation -> Document -> Sentences -> Words
# - Stopward Removal -> Removing words like do, they, there etc..
# - Stemming -> to convert different form of a word into a base word like : see seen seem := see
# - Building Vocab -> Vocab is list of all words
# - Vectorization -> creating a list of size vocab to store word's frequency from Vocab
# - Classification -> classify into category

# In[2]:


from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# In[3]:


sample_text = """I loved this movie <br /><br /> since I was 7 and I saw it on the opening day. It was so touching and beautiful. I strongly recommend seeing for all. It's a movie to watch with your family by far.<br /><br />My MPAA rating: PG-13 for thematic elements, prolonged scenes of disastor, nudity/sexuality and some language."""


# In[22]:


tokenizer = RegexpTokenizer(r'\w+')


# In[23]:


en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()


# In[24]:


def getCleanReview(review):
    review = review.lower()
    review = review.replace("<br /><br />"," ")
    # Tokenize
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    cleaned_review = ' '.join(stemmed_tokens)
    return cleaned_review


# In[25]:


getStemmedReview(sample_text)


# In[26]:


# Function that accepts input file and return clean output file of movie reviews


# In[27]:


def getStemmedDoc(inputFile, outputFile):
    out = open(outputFile,'w',encoding="utf8")
    with open(inputFile, encoding ="utf8") as f:
        reviews=f.readlines()
    for review in reviews:
        cleaned_review = getStemmedReview(review)
        print((cleaned_review), file = out)
    out.close()


# In[28]:


import sys


# In[29]:


# # Read command line arguments
# inputFile  = sys.argv[1]
# outputFile = sys.argv[2]
inputFile ="./imdb_toy_x.txt"
outputFile = "./imdb_toy_clean.txt"
getStemmedDoc(inputFile,outputFile)


# 

# In[ ]:




