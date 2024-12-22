## First time building a sentiment analysis model. Done in 2 hours during a practical session in class.
## Data loading, cleaning, tokenization, training, evaluation and prediction all in one place on US Airlines sentiment data.


#Necessary imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

#nltk for language preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.utils import pad_sequences

## for building the keras model
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Embedding, SpatialDropout1D
from keras import Input
from keras.utils import plot_model

## sklearn model selection
from sklearn.model_selection import train_test_split

## Figures
import matplotlib.pyplot as plt



## Import data
dataset=pd.read_csv('Twitter US Airline Sentiment.csv', sep=',') #load the dataset
dataset=dataset[["airline_sentiment","text"]].copy() #keep only the necessary columns
dataset=dataset[dataset['airline_sentiment']!='neutral'] #on retire les messages neutres
dataset.head(10)

## Clean and tokenize data
#les sentiments neutres ont déjà été supprimés dans la cellule de code précédente
punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~' #ponctuations à retirer
max_features=1500 #valeur proposée par l'énoncé

## removal of stop words with nltk
stop_words = set(stopwords.words('english'))


# get rid of punctuation
reviews=dataset['text'].to_numpy()
all_reviews = 'separator'.join(reviews)
all_reviews = all_reviews.lower()
all_text = ''.join([c for c in all_reviews if c not in punctuation])

# split by new lines and spaces
reviews_split = all_text.split('separator')
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()

# get rid of web address, twitter id, digit, and stop_words #source: https://github.com/AmitG47/-Twitter-US-Airline-Sentiment-Analysis-with-an-RNN/blob/main/RNN_Twitter_Sentiment_Analysis.ipynb
new_reviews = []
for review in reviews_split:
    review = review.split()
    new_text = []
    for word in review:
        if (word[0] != '@') & ('http' not in word) & (~word.isdigit()) & (word not in stop_words):
            new_text.append(word)
    new_reviews.append(new_text) #re.sub(EYMERIC ME LENVOIE])

## Keras tokenizer: après avoir retiré quelques trucs, on tokenize.
t=Tokenizer(num_words=max_features, filters=punctuation, split=' ', lower= True)
t.fit_on_texts(new_reviews)

X = t.texts_to_sequences(new_reviews)
X = pad_sequences(X) #pour remplir avec des 0


## Build the model
embed_dim =128
lstm_out=196
batch_size=32
dropout_x=0.2

model=Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1])) #X.shape[1]: nombre de mots (nb de colonnes): taille d'un input - càd d'un review.
model.add(SpatialDropout1D(dropout_x)) #à voir à quoi ça sert
model.add(LSTM(lstm_out, dropout=dropout_x, recurrent_dropout=dropout_x))# activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


## Data for training
Y = pd.get_dummies(dataset, columns=['airline_sentiment'])
Y = Y[['airline_sentiment_negative']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


## Training
history = model.fit(X_train, Y_train, epochs=7, batch_size=256, verbose=2)
model.evaluate(X_train,Y_train, batch_size=128)

## Model evaluation
model.evaluate(X_test,Y_test, batch_size=128)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


## Predictions with the model 
sentence = ['While many hate baseball, I love it']
sentence = t.texts_to_sequences(sentence)
sentence = pad_sequences(sentence, maxlen=X.shape[1]) #ne pas oublier maxlen: et VOIR POURQUOI ON MET CE PARAMETRE !! 

pred = model.predict(sentence)

if pred > 0.5:
    print("Négatif")
    print(pred)
else:
    print("Positif")
    print(pred)

