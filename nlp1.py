from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression
from IPython.display import FileLink, display
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
import joblib
import re

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test = test.drop(columns=['keyword','location'])
train = train.drop(columns=['keyword','location'])

train.info()
print(train.head())

target_column = 'target'
plt.figure(figsize=(10, 6))
sns.countplot(data=train, x=target_column)
plt.title('Distribution of Target Column')
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.show()


nltk.download('stopwords')

def print_plot(index):
    example = train[train.index == index][['text', 'target']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Tag:', example[1])
print_plot(10)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = BeautifulSoup(text, "lxml").text 
    text = text.lower() 
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    return text

train['target'] = train['target'].apply(clean_text)
print_plot(10)

X = train.text
y = train.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42, shuffle= True)

"""**Naive** **Bayes**"""

nb_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

nb_pipeline.fit(X_train, y_train)
y_pred = nb_pipeline.predict(X_test)

print('Accuracy: %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
joblib.dump(nb_pipeline, 'nb_pipeline.pkl')
nb_pipeline_loaded = joblib.load('nb_pipeline.pkl')
display(FileLink('nb_pipeline.pkl'))

"""**Linear Support Vector Machine**"""

svm_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=1000, tol=1e-3)),
])

svm_pipeline.fit(X_train, y_train)
y_pred_svm = svm_pipeline.predict(X_test)

print('Linear SVM Model:')
print('Accuracy: %s' % accuracy_score(y_pred_svm, y_test))
print(classification_report(y_test, y_pred_svm, target_names=['negative', 'positive']))
joblib.dump(svm_pipeline, 'LSVM.pkl')
LSVM_loaded = joblib.load('LSVM.pkl')
display(FileLink('LSVM.pkl'))

"""**Logistic Regression**"""

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=['negative', 'positive']))


joblib.dump(logreg, 'logreg.pkl')
logreg_loaded = joblib.load('logreg.pkl')
display(FileLink('logreg.pkl'))



