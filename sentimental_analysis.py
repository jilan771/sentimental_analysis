import pandas as pd
import numpy as np
data = pd.read_csv("Restaurant_Reviews.tsv",sep="\t")
X = data.loc[:,'Review'].values
y = data.loc[:,'Liked'].values
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def cleaned(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

X_cleaned = data.loc[:,'Review'].apply(cleaned)
print(X_cleaned.head())
from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer(max_features=3000)
X = vector.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(C=1, solver='liblinear')
classifier.fit(X_train, y_train)
from sklearn.metrics import accuracy_score,classification_report

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

print(classification_report(y_test,y_pred))
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
new_d = ["It is not so good"]

new_d_transformed = vector.transform(new_d)

prediction = classifier.predict(new_d_transformed)

print("👍 This is a positive review!" if prediction[0] == 1 else "👎 This is a negative review.")