#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## importing 

# In[6]:


import pandas as pd


# ## Path for data can be changed depending of data set

# In[7]:


path = "/kaggle/input/million-headlines/abcnews-date-text.csv"


# ## Conversion of csv data to DataFrame
# #using : pandas 

# In[8]:


df = pd.read_csv(path)
df = pd.DataFrame(df)
df.shape


# In[9]:


df.head(100)


# In[10]:


text = df["headline_text"]

type(text)


# ## Creating a word cloud for data text

# In[11]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Convert the Series to a single string by concatenating its elements
text = ' '.join(text.astype(str))

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud using matplotlib
plt.figure(figsize=(100, 50))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Optional: Save the word cloud to an image file
wordcloud.to_file("wordcloud.png")



# ## Importing nltk library
# [Natural Language Toolkit](https://www.nltk.org/)

# In[13]:


import nltk


# ## Example to show capablity of NLTK

# In[15]:


example = df["headline_text"][123]
print("text sample 123 from head line : "  + "'" + example + "'" )
print("")
tokens = nltk.word_tokenize(example)
print("example how tokenizer converts string of text into tokens : ")
print(tokens)


# # **nltk pos_tag tags tokens Abbreviation --> Meaning**
# ## for [example](https://www.guru99.com/pos-tagging-chunking-nltk.html) nn = noun, singular (cat, tree)

# In[16]:


tagged = nltk.pos_tag(tokens)
tagged[:10]


# # [SentimentAnalyzer](https://www.nltk.org/api/nltk.sentiment.sentiment_analyzer.html)
# ### A Sentiment Analysis tool based on machine learning approaches.
# #### A SentimentAnalyzer is a tool to implement and facilitate Sentiment Analysis tasks using NLTK features and classifiers, especially for teaching and demonstrative purposes.

# In[17]:


from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


# In[18]:


example


# In[19]:


sia.polarity_scores(example)


# In[20]:


res = {}
for i in range (len(df)):
    text = df["headline_text"][i]
    res[i] = sia.polarity_scores(text)


# In[22]:


ndf = pd.DataFrame(res)


# In[23]:


ndf = ndf.T


# In[24]:


ndf.head(100)


# In[25]:


data = ndf.head(100)


# In[26]:


data


# In[27]:


data = ndf.head(1000)


# In[28]:


data


# In[29]:


import seaborn as sns


# In[30]:


odf=ndf.reset_index().rename(columns={'index': 'Id'})


# In[31]:


sns.relplot(
    data=ndf.head(100), x="neg", y="pos", hue="compound",
)


# In[32]:


odf


# In[33]:


# Create two sample DataFrames
df1 =odf
df2 = date

# Concatenate df2 below df1 (vertically)
result = pd.concat([df1, df2], axis=1)

# Reset the index of the resulting DataFrame
result = result.reset_index(drop=True)

# Display the concatenated DataFrame
print(result)


# In[ ]:


result


# In[ ]:


sns.relplot(data=result.head(10000), x="publish_date", y="neg", kind="line")


# In[ ]:


sns.displot(result.head(100000), x="compound")


# In[ ]:


sns.jointplot(
    data=result.head(1000),
    x="pos", y="neg", hue="publish_date",
    kind="kde"
)


# In[ ]:


sns.pairplot(result.head(1000))


# In[ ]:


sns.catplot(
    data=result.head(1000), y="pos", hue="publish_date", kind="count",
    palette="pastel", edgecolor=".6",
)


# In[ ]:


df_no_duplicates = year.drop_duplicates()


# ### Dnp date negative positive
# #### pose for positive
# ##### pc positive count
# ##### pda positive date
# ##### pTx positive text
# #### nege for negative
# ##### nc negative count
# ##### nda negative date
# ##### nTx negative text

# In[36]:


Dnp={
    "pose" :{
        "pc" : {
            
        },
        "pda":{
            
        },
        "pTx" : {
            
        }
    },
    "nege":{
        "nc" : {
            
        },
        "nda" : {
            
        },
        "nTx" : {
            
        }
    }
}


# In[37]:


Dnp


# In[38]:


cn=0 #counter for negative
co=0 #counter for positive
df #original data frame


# In[39]:


for i in range(len(ndf)) : 
    if ndf["pos"][i] > ndf["neg"][i]:
        Dnp["pose"]["pc"][co] = i
        Dnp["pose"]["pda"][co] = df["publish_date"][i]
        Dnp["pose"]["pTx"][co] = df["headline_text"][i]
        co+=1
    elif ndf["pos"][i] < ndf["neg"][i]:
        Dnp["nege"]["nc"][cn] = i
        Dnp["nege"]["nda"][cn] =  df["publish_date"][i]
        Dnp["nege"]["nTx"][cn] =  df["headline_text"][i]
        cn+=1


# In[40]:


dnp = Dnp


# In[41]:


pd.DataFrame(dnp)


# In[42]:


pd.DataFrame(dnp["pose"])


# In[43]:


pd.DataFrame(dnp["nege"])


# In[45]:


Ntext = pd.DataFrame(Dnp["nege"])


# In[46]:


# Convert the Series to a single string by concatenating its elements
text = ' '.join(Ntext["nTx"].astype(str))

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud using matplotlib
plt.figure(figsize=(100, 50))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Optional: Save the word cloud to an image file
wordcloud.to_file("Pwordcloud.png")


# In[47]:


Ptext = pd.DataFrame(Dnp["pose"])


# In[48]:


# Convert the Series to a single string by concatenating its elements
text = ' '.join(Ptext["pTx"].astype(str))

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud using matplotlib
plt.figure(figsize=(100, 50))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Optional: Save the word cloud to an image file
wordcloud.to_file("Nwordcloud.png")


# In[ ]:




