import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    zero = PorterStemmer()
    review = [zero.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

'''from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = None)
x_lda = lda.fit(x, y)

lda_ratio = lda.explained_variance_ratio_
def select_n_componets(var_ratio, goal_var: float):
    total_variance = 0.0
    n_components = 0
    for explained_ratio in var_ratio:
        total_variance += explained_ratio
        n_components += 1
        if total_variance >= goal_var:
            break
    return n_components
select_n_componets(lda_ratio, 2)

lda = LDA(n_components = 1)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)'''

from sklearn.tree import DecisionTreeClassifier
gnb = DecisionTreeClassifier(criterion = 'entropy')
gnb = gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)

from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator = gnb,
                      X = x_train,
                      y = y_train,
                      cv = 10)
cvs.mean()