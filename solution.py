import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import spacy
from spacy.lang.it.stop_words import STOP_WORDS
from spacy.lang.it import Italian
import string
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression ,SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#READ FILE

data_training=pd.read_csv('development.csv')
data_test=pd.read_csv('evaluation.csv')
data_training.shape , data_test.shape

data_training.head()

#Exploratory Data analysis
data_test.info(), data_training.info()
data_training['length'] = data_training['text'].apply(len)
fig1 = sns.barplot('class','length',data = data_training,palette='PRGn')
plt.title('Average Word Length vs Class')
plot = fig1.get_figure()

fig2 = sns.countplot(x= 'class',data =data_training)
plt.title('Class Counts')
plot = fig2.get_figure()
plot.savefig('Count Plot.png')

c=Counter(data_training['class'])
for i in ['pos','neg']:
    print(i, c[i]/len(data_training['class']))
    
#Preprocessing
#Create list of punctuation , list of stopwords and create tokenizer function. Create token object, which is used to create documents with linguistic annotations. and return mytoken
punctuations = string.punctuation
nlp = spacy.load("it_core_news_sm")
stop_words = spacy.lang.it.stop_words.STOP_WORDS
stop_words=[]
parser = Italian()


def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    filtered_tokens = []
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in punctuations ]
   
    return mytokens
   
def clean_text(text):
    return text.strip().lower()

#Feature extraction
X = data_training['text']
y = data_training['class']
test = data_test['text']

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer,stop_words=stop_words,preprocessor=clean_text,ngram_range=(1, 2),max_df=0.8,min_df=5)
vecto_without_stopword = TfidfVectorizer()

#SPLIT
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#Algorithm Choice

models = []
names=['LogisticRegression','MLPClassifier','SGDClassifier','BernoulliNB','RandomForestClassifier','SVC()', 'LinearSVC']
models.append(( LogisticRegression()))
models.append(( MLPClassifier()))
models.append(( SGDClassifier()))
models.append(( BernoulliNB()))
models.append(( RandomForestClassifier()))
models.append(( SVC(gamma=0.1)))
models.append(( LinearSVC()))
accuracy=[]
f1=[]
df=pd.DataFrame

X_train_vect=vectorizer.fit_transform(X_train)
X_test_vect=vectorizer.transform(X_test)
for clf in models:
    clf.fit(X_train_vect, y_train)
    y_pred= clf.predict(X_test_vect)
    accuracy.append(accuracy_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred,average='weighted'))

df = pd.DataFrame(dict(name=names,accu=accuracy,score=f1)) 
df

ind = np.arange(len(df))
width = 0.4
fig3, ax = plt.subplots(figsize=(15,4))
ax.barh(ind, df.score, width, label='f1_score' )
ax.barh(ind + width, df.accu, width, label='Accuracy')
ax.set(yticks=ind + width, yticklabels=df.name, ylim=[2*width-1 , len(df)])
plt.xticks(np.arange(0, max(df.score), 0.05))
plt.axvline(x=0.95,linestyle='--',color='black')
ax.legend(loc=(0.8,1))
plt.title('Classification Performance')
plt.show()
fig3.savefig('ClassificationPerformancesol2.png')

#Creation Pipeline
ipeline = Pipeline ([('vect',vectorizer),('clf',LinearSVC())])
pipeline_without_stopword=Pipeline ([('vect',vecto_without_stopword),('clf',LinearSVC())])
pipeline.fit(X_train,y_train)
y_predict=pipeline.predict(X_test)

print(classification_report(y_test,y_predict, digits=4))
print("The accuracy score is: ",accuracy_score(y_test,y_predict))
pd.DataFrame(
    confusion_matrix(y_test,y_predict), 
    columns=["Negatives", "Positives"],
    index=["Negatives", "Positives"])
param_grid={'C':[0.1,0.5,1,1.5,2,2.5,3,5,10],
           'tol':[ 1,1e-01,1e-02,1e-03],
           'random_state':[42],
           'class_weight':['balanced',None],
            'dual':[False,True],
            'fit_intercept':[True,False],
            'multi_class':['ovr'],
          }
X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)
f1_scorer = make_scorer(f1_score, pos_label="pos")
gridsearch= GridSearchCV(LinearSVC(), param_grid, cv=5, n_jobs=-1,scoring=f1_scorer)
gridsearch.fit(X_train, y_train)
y_predict_grid=gridsearch.predict(X_test)
print("Best cross-validation score: {:.2f}".format(gridsearch.best_score_))
print("Best parameters: ", gridsearch.best_params_)
print("Best estimator: ", gridsearch.best_estimator_)
print(classification_report(y_test, y_predict_grid, digits=4))
pd.DataFrame(
    confusion_matrix(y_test,y_predict_grid), 
    columns=["Negatives", "Positives"],
    index=["Negatives", "Positives"])
    
X_vect=vectorizer.fit_transform(X)
f1_scorer = make_scorer(f1_score, pos_label="pos",)
gridsearch_all= GridSearchCV(LinearSVC(), param_grid, cv=5, n_jobs=-1,scoring=f1_scorer)
gridsearch_all.fit(X_vect, y)
print("Best cross-validation score: {:.2f}".format(gridsearch_all.best_score_))
print("Best parameters: ",gridsearch_all.best_params_)
print("Best estimator: ", gridsearch_all.best_estimator_)


pipeline_after_gridsearch= Pipeline ([('vect',vectorizer),('clf',LinearSVC(C=10,fit_intercept=False,class_weight= None,multi_class= 'ovr', penalty= 'l2', tol= 0.1,random_state=42,dual=False))])
pipeline_after_gridsearch.fit(X,y)

y_eval=pipeline_after_gridsearch.predict(test)
y_eval[12322][:]

import csv
field=['Id','Predicted']
with open('submissionssol1.csv', 'w') as csvfile: 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(field)
    for i,y in enumerate(y_eval):
        row=[i,y_eval[i][:]]
        csvwriter.writerow(row)





