# Exp 7: Multinomial vs Bernoulli Naive Bayes on 20 Newsgroups (simple version)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# 1. Load data
df = pd.read_csv("newsgroups.csv").dropna()
X = df['text']
y = df['label']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Both models (only difference: binary=False / True)
for name, model, binary in [("Multinomial", MultinomialNB(), False),
                            ("Bernoulli", BernoulliNB(), True)]:
    vec = CountVectorizer(stop_words='english', max_features=5000, binary=binary)
    Xtr = vec.fit_transform(X_train)      # learn words + convert
    Xte = vec.transform(X_test)           # convert with same words
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    print(name, " Accuracy:", round(accuracy_score(y_test, pred), 3),
          " F1:", round(f1_score(y_test, pred, average='macro'), 3))
