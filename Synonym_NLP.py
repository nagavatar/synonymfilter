#!/usr/bin/env python
# coding: utf-8

# # Synonym Expansion

# The basic idea is to generate synonyms of keywords and help them in the expansion of search query so as to give more relevant results.
# E.g. I know a song vaguely whose lyrics are **"the sky is so *up*"** . Here I don't remember the exact last word but vaguely remember that the word is similar to up. So the search result should be **the sky is so high** **the sky is so above** , **the sky is so high**

# 

# This notebook shows the synonym generation. Starting block shows the practice and handson on basic NLP tasks. Actual training of Neural Network starts from **"Billboard Lyrics Dataset"** caption which is some where down in the middle.
# 

# 

# # Generating synonyms using wordnet vocabulary

# Here, we are trying to generate synonyms using wordnet vocabulary. However, this idea has its limitation as this dictionary is manually build and cannot take into account the slangs or abbreviations like AKA, coz, bcoz etc.

# In[47]:


import nltk
import torch
from nltk.corpus import wordnet


# In[51]:


array_ant=[]
array_syn=[]


# In[37]:


wordnet.synsets("happy")


# In[38]:


wordnet.synsets("good")


# In[39]:


wordnet.synsets("good")[4].lemmas()[0].antonyms()


# In[52]:


for vayn in wordnet.synsets("high"):
    for l in vayn.lemmas():
        array_syn.append(l.name())
        if l.antonyms():
            print(l.antonyms)
            array_ant.append(l.antonyms()[0].name())


# In[41]:


print(set(array_ant))


# In[53]:


#synonyms of sky using WordNet(manual thesaurus)
print(set(array_syn))


# # PyTorch Intro - Practice

# This is a basic intro of PyTorch which I tried to learnt and tried to see how it works.

# In[10]:


from torch.nn import Embedding
n_embed, dim =10,4


# In[11]:


emb_1=Embedding(n_embed,dim)


# In[11]:


emb_1.weight


# In[23]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
# FloatTensor containing pretrained weights
weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)
# Get embeddings for index 1
input = torch.LongTensor([1])
embedding(input)


# In[22]:


import gensim
#model = gensim.models.KeyedVectors.load_word2vec_format('path/to/file')
#weights = torch.FloatTensor(model.vectors)


# In[24]:


sentence = "the quick brown fox jumped over the lazy dog"
words = sentence.split(' ')
print(words)


# In[25]:


vocab1 = list(set(words))
print(vocab1)


# In[29]:


# Number of words in our vocabulary
len(vocab1)


# # Basics of NLP - One Hot Encoding

# In[27]:


# Convert words to indexes
word_to_ix1 = {word: i for i, word in enumerate(vocab1)}
print(word_to_ix1)


# In[28]:


from torch.nn.functional import one_hot

words = torch.tensor([word_to_ix1[w] for w in vocab1], dtype=torch.long)

one_hot_encoding = one_hot(words)
print(vocab1)
print(one_hot_encoding)


# The issue with sparse one-hot encoding is that the vectors are very large and we have a very sparse representation of the vectors. As you can see there are a lot of zeros. For example, the popular data set WikiText-103 has 267,000 words in the vocabulary. This means around 267,000 zeros in each vector with one-hot encoding.
# 
# We should try to find a smaller encoding for our dataset. Let's try a denser vector using a Word Embedding.

# # Basics Practice NLP

# In[ ]:


# A Word embedding is a learned representation for text where words that have the same meaning have a similar representation.
# Since neural network can't take inputs as words but only some numerics, so we can say it's kind of numeric representation of words.
# We can visualise word embeddings using t-SNE or PCA.


# In[35]:



# Context is the number of words we are using as a context for the next word we want to predict
CONTEXT_SIZE = 2

# Embedding dimension is the size of the embedding vector
EMBEDDING_DIM = 10

# Size of the hidden layer
HIDDEN_DIM = 256


# In[151]:


#Training on small dataset
# We will use Shakespeare Sonnet 2
test_sentence = """Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time;
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player,
That struts and frets his hour upon the stage,
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.
""".lower().split()


# In[152]:


test_sentence


# In[153]:


# Build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab2 = list(set(test_sentence))
word_to_ix2 = {word: i for i, word in enumerate(vocab2)}

# Show what a trigram looks like


# ### N - Gram Language Model

# ![image.png](attachment:image.png)

# # Basics - Word Embedding using PyTorch

# In[154]:


import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, HIDDEN_DIM)
        self.linear2 = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# In[155]:


learning_rate = 0.001
losses = []
loss_function = nn.NLLLoss()  # negative log likelihood
model = NGramLanguageModeler(len(vocab2), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# In[156]:


from tqdm import tqdm

for epoch in range(25):
    total_loss = 0

    iterator = tqdm(trigrams)
    for context, target in iterator:
        # (['When', 'forty'], 'winters')
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix2[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix2[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
        iterator.set_postfix(loss=float(loss))
    losses.append(total_loss)
    # add progress bar with epochs


# In[157]:


# Check the structure of our model here
model.eval()


# In[43]:


#predicting the next word using context
with torch.no_grad():
    context = ['tomorrow,', 'and']
    context_idxs = torch.tensor([word_to_ix2[w] for w in context], dtype=torch.long)
    pred = model(context_idxs)
    print(pred)
    index_of_prediction = numpy.argmax(pred)
    print(vocab2[index_of_prediction])


# # Basics of NLP - Continuous Bag of Words

# In[44]:


CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab3 = list(set(raw_text))
vocab_size = len(vocab3)

word_to_ix3 = {word: i for i, word in enumerate(vocab3)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


# In[45]:


# create your model and train.  here are some functions to help you make
# the data ready for use by your module

def make_context_vector(context, word_to_ix3):
    idxs = [word_to_ix3[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

make_context_vector(data[0][0], word_to_ix3)  # example


# In[46]:


class CBOW(nn.Module):
    def __init__(self):
        pass

    def forward(self, inputs):
        pass


# In[54]:


#!pip install --upgrade tensorflow-hub


# In[57]:


# import necessary libraries
#trying to use ELMO
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Load pre trained ELMo model
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

# create an instance of ELMo
embeddings = elmo(
	[
		"I love to watch TV",
		"I am wearing a wrist watch"
	],
	signature="default",
	as_dict=True)["elmo"]
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Print word embeddings for word WATCH in given two sentences
print('Word embeddings for word WATCH in first sentence')
print(sess.run(embeddings[0][3]))
print('Word embeddings for word WATCH in second sentence')
print(sess.run(embeddings[1][5]))


# In[ ]:





# In[ ]:





# # Billboard lyrics Dataset
# ## Generating synonyms using Billboard lyrics dataset
# ###  A Small dataset

# In[1]:


get_ipython().system('pip install python-Levenshtein')


# In[2]:


#importing gensim library which is quite famous library for NLP
#reading the top 100 songs lyrics of 50 years dataset
import gensim.models
import pandas as pd
df = pd.read_csv("C://Users//Madhusudan//Downloads//billboard_lyrics_1964-2015.csv",encoding="latin-1")


# In[3]:


sentences=a#, "I love my cat", "I love you babe", "You love my dog!", "I love my dog too!!"]
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)


# In[ ]:


word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}

vocab_size = len(word2id) + 1 
embed_size = 100

wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in a]
print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])


# In[ ]:


from keras.preprocessing.sequence import skipgrams

# generate skip-grams
skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=10) for wid in wids]

# view sample skip-grams
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(10):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
          id2word[pairs[i][0]], pairs[i][0], 
          id2word[pairs[i][1]], pairs[i][1], 
          labels[i]))


# In[ ]:


df.head()


# In[ ]:


df1 = df[['Lyrics']]


# In[ ]:


df['class']='lyrics'


# In[ ]:


#remove null columns
df1= df1.dropna(subset=['Lyrics'])


# In[ ]:


#pre-processes i.e. remove punctuation and remove stopwords like I. It also lowercases charcters

lyrics = df1.Lyrics.apply(gensim.utils.simple_preprocess)
lyrics


# In[ ]:


#setting the model parameters for word2vec model
model=gensim.models.Word2Vec(window=10, min_count=2, workers=4)


# In[ ]:


#building voacbulary of unique words
model.build_vocab(lyrics, progress_per = 1000)


# In[ ]:


model.epochs


# In[ ]:


model.corpus_count


# In[ ]:


#training the model
model.train(lyrics, total_examples = model.corpus_count, epochs = 10)


# In[228]:


#saving the model
model.save("./word2vec-billboard.model")


# In[229]:


model.wv.most_similar("sky")


# In[ ]:





# # Review Sports

# This dataset is of amazon reviews downloaded from stanford site.
# This data is used for training of the model using word2vec algorithm. It uses **skipgram** model

# In[4]:


df_sports = pd.read_json("C://Users//Madhusudan//Downloads//reviews_Sports_and_Outdoors_5.json", lines=True)


# In[5]:


df_sports.head()


# In[6]:


df_sports.shape


# In[19]:


df_sports['class']="sports"


# In[205]:


review_text = df_sports.reviewText.apply(gensim.utils.simple_preprocess)


# In[206]:


review_text


# In[7]:


#setting the model parameters
model=gensim.models.Word2Vec(window=5,min_count=2,workers=4, sg=1)


# In[215]:


#building voacbulary of unique words
model.build_vocab(review_text, progress_per =1000)


# In[209]:


#train word2vec model by using reviews
model.train(review_text,total_examples= model.corpus_count, epochs =5)


# In[210]:


model.save("./word2vec-review-dataset.model")


# In[211]:


#TAKING COSINE SIMILARITY and finding synonyms
model.wv.most_similar("awful")


#  'awful' and find similarities between the following word tuples: ('good', 'great'), ('slow','steady')"

# In[212]:


model.wv.similarity(w1="great",w2="good")


# In[213]:


model.wv.similarity(w1="slow",w2="steady")


# In[214]:


#finding synonyms using the model generated using Neural Network
model.wv.most_similar("high")


# I got to know that there are few more techniques like GloVe and Elmo to do this. I'll be trying that as well.

# # Classification task of Airbnb and Sports Reviews

# In[13]:


airbnb_url = 'https://www.kaggle.com/tylerx/discover-sentiment-in-airbnb-reviews/data?select=reviews_dec18.csv'


# In[11]:


from urllib.request import urlretrieve


# In[14]:


urlretrieve(airbnb_url, 'airbnb.csv')


# In[1]:


import pandas as pd


# In[2]:


airbnb_df = pd.read_csv('C:\\Users\\Madhusudan\\Downloads\\reviews_dec18.csv',error_bad_lines=False, engine='python')


# In[3]:


airbnb_df.shape


# In[17]:


airbnb_df.head()


# In[18]:


airbnb_df.comments[2]


# In[ ]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
#from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


# In[ ]:


airbnb_df.isnull().values.any()


# In[ ]:


airbnb_df.shape


# In[181]:


airbnb_df['class']="airbnb"


# In[ ]:


airbnb_df.head()


# In[183]:


song_df=df[['Lyrics','class']]
song_df.rename(columns={'Lyrics':'wordings'},inplace=True)


# In[184]:


sports_df=df_sports[['reviewText','class']]
sports_df.rename(columns={'reviewText':'wordings'},inplace=True)


# In[185]:


airbnb_df = airbnb_df[['comments','class']]
airbnb_df.rename(columns={'comments':'wordings'},inplace=True)


# In[269]:


#df_sports[['reviewText','class']].append(airbnb_df[['comments','class']],ignore_index=True,
 #   verify_integrity=True)


# In[233]:


df_combined=pd.concat([sports_df,airbnb_df])


# In[188]:


df_combined[df_combined['class']=='lyrics']


# In[189]:


import seaborn as sns


# In[190]:


#looking at the count of both classes
sns.countplot(x='class', data=df_combined)


# In[191]:


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


# In[192]:


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', str(text))


# In[193]:


X = []
sentences = list(df_combined['wordings'])
for sen in sentences:
    X.append(preprocess_text(sen))


# In[194]:


X[3]


# In[195]:


y = df_combined['class']

y = np.array(list(map(lambda x: 1 if x=="airbnb" else 0, y)))


# In[196]:


y[25555]
df_combined.iloc[25555]


# In[197]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[198]:


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# In[199]:


# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100


X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# In[200]:


#using GLove embeddings
from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('C:\\Users\\Madhusudan\\Downloads\\glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()


# In[264]:


vocab_size


# In[201]:


embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# In[202]:


embedding_matrix.shape


# In[203]:


from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


# In[204]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.layers import InputLayer


# In[205]:


model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(tf.keras.layers.LSTM(128))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[206]:




print(model.summary())


# In[207]:


#1 epoch time equals 25 mins and total there were 6 epochs
history = model.fit(X_train, y_train, batch_size=128, epochs=2, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)


# In[208]:


print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[209]:


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# In[255]:


instance = X[48610]
print(instance)
type(instance)


# In[211]:


instance = tokenizer.texts_to_sequences(instance)

flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)

flat_list = [flat_list]

instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

model.predict(instance)


# In[213]:


preds = model.predict(X_test)


# In[221]:


pred=model.predict_classes(X_test)


# In[222]:


pred


# In[215]:


preds


# In[250]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm=confusion_matrix(y_test, pred)


# In[242]:


df_cm=pd.DataFrame(cm, index=['sports','airbnb'],columns=['sports','airbnb'])


# In[243]:


df_cm.head()


# In[236]:


sum(df_combined['class']=='sports')


# In[235]:


df_combined.head()


# In[241]:


sum(y_test)


# In[246]:


len(y_test)-sum(y_test)


# In[252]:


report=classification_report(y_test,pred)
print(report)


# In[253]:


#https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras


# In[262]:


instance='The roooms were neat and clean and the host was super cool'
instance = tokenizer.texts_to_sequences(instance)

flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)

flat_list = [flat_list]

instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

model.predict(instance)
#model.predict("The roooms were neat and clean and the host was super cool")


# In[257]:


model.save("./review-classification.model")


# In[267]:


model.save('review-classification.h5')


# In[260]:


df_sports.reviewText[2]


# # Contextual Embedding using BERT

# In[23]:


airbnb_comments = airbnb_df[['comments']]


# In[27]:


get_ipython().system('pip install -U sentence-transformers')


# In[30]:



from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')


# In[29]:


from nltk.corpus import stopwords


# In[33]:


import re


# In[35]:


#documents_df=pd.DataFrame(documents,columns=['documents'])

# removing special characters and stop words from the text
stop_words=stopwords.words('english')
airbnb_comments['airbnb_cleaned']=airbnb_comments.comments.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in str(x).split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )


# In[ ]:



airbnb_embeddings = sbert_model.encode(airbnb_comments['airbnb_cleaned'])

pairwise_similarities=cosine_similarity(airbnb_embeddings)
pairwise_differences=euclidean_distances(airbnb_embeddings)

most_similar(0,pairwise_similarities,'Cosine Similarity')
most_similar(0,pairwise_differences,'Euclidean Distance')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




