from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
import nltk
from IPython.display import display


# nltk.download()
import pandas as pd
from nltk.tokenize import word_tokenize


fake = pd.read_csv("Fake.csv", low_memory=False)  # read fake and
# true csv files
genuine = pd.read_csv("True.csv", low_memory=False)

# display(fake.info())
# display(genuine.info())

# display(fake.head(10))
# display(genuine.head(10))


# display(fake.subject.value_counts())

fake['target'] = 0  # set fake ones as 0 and true ones as 1
genuine['target'] = 1

# concatinate both files row wise
data = pd.concat([fake, genuine], axis=0)
date = data.reset_index(drop=True)                           # reset index
# drop selected columns as we dont need them
data = data.drop(['subject', 'date', 'title'], axis=1)

# print(data.columns)


# make list of every sentence having words
data['text'] = data['text'].apply(word_tokenize)
# print(data.head(10))


porter = SnowballStemmer("english")                           # Stemming


def stem_it(text):
    return [porter.stem(word) for word in text]


data['text'] = data['text'].apply(stem_it)

# print(data.head(10))


# from nltk.corpus import stopwords
# nltk.download('stopwords')
# print(stopwords.words('english'))

def stop_it(t):
    dt = [word for word in t if len(word) > 2]          # StopWord Removal
    return dt


data['text'] = data['text'].apply(stop_it)
# print(data.head(10))

data['text'] = data['text'].apply(' '.join)

# SPLITTING UP OF DATA

X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['target'], test_size=0.25)
display(X_train.head())
print('\n')
display(y_train.head())


# VECTORIZATION

my_tfidf = TfidfVectorizer(max_df=0.7)

tfidf_train = my_tfidf.fit_transform(X_train)
tfidf_test = my_tfidf.transform(X_test)

print(tfidf_train)


# LOGISTIC-REGRESSION

model_1 = LogisticRegression(max_iter=900)
model_1.fit(tfidf_train, y_train)
pred_1 = model_1.predict(tfidf_test)
cr1 = accuracy_score(y_test, pred_1)

print(cr1*100)


# PASSIVE-AGGRESSIVE-CLASSIFER

from sklearn.linear_model import PassiveAggressiveClassifier

model =  PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train,y_train)

y_pred = model.predict(tfidf_test)
accscore = accuracy_score(y_test, y_pred)
print('The accuracy of prediction is ',accscore*100)
