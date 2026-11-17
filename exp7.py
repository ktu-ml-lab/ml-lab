# Exp 7: Text classification on 20 Newsgroups
# Multinomial Naive Bayes (word COUNTS) vs Bernoulli Naive Bayes (word PRESENT 0/1)
# Metrics: Accuracy, F1-score    Plot: bar chart comparison

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------- STEP 1: LOAD DATA (keep ONE option, comment the others) ----------------
# OPTION A: from scikit-learn (downloads once, needs internet the first time)
# from sklearn.datasets import fetch_20newsgroups
# data = fetch_20newsgroups(subset='all', categories=['sci.space', 'rec.sport.baseball', 'comp.graphics'], remove=('headers', 'footers', 'quotes'))
# X = data.data                  # list of text documents
# y = data.target                # 0, 1, 2

# OPTION B: from CSV file (columns: text, label)
df = pd.read_csv("newsgroups.csv")
df = df.dropna()                 # some posts are empty after cleaning
X = df['text']
y = df['label']

# OPTION C: from .data file (tab separated, no header, label is TEXT topic name)
# df = pd.read_csv("newsgroups.data", sep='\t', header=None, names=['topic', 'text'], quoting=3)
# df = df.dropna()
# X = df['text']
# y = LabelEncoder().fit_transform(df['topic'])    # 'comp.graphics' -> 0, ...

print("Documents:", len(X))

# ---------------- STEP 2: SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- STEP 3: TEXT -> NUMBERS ----------------
# Multinomial: word counts
cv_count = CountVectorizer(stop_words='english', max_features=5000)
Xtr_count = cv_count.fit_transform(X_train)
Xte_count = cv_count.transform(X_test)

# Bernoulli: binary (word present = 1, absent = 0)
cv_bin = CountVectorizer(stop_words='english', max_features=5000, binary=True)
Xtr_bin = cv_bin.fit_transform(X_train)
Xte_bin = cv_bin.transform(X_test)

# ---------------- STEP 4: TRAIN + PREDICT ----------------
mnb = MultinomialNB()
mnb.fit(Xtr_count, y_train)
y_pred_m = mnb.predict(Xte_count)

bnb = BernoulliNB()
bnb.fit(Xtr_bin, y_train)
y_pred_b = bnb.predict(Xte_bin)

# ---------------- STEP 5: EVALUATE ----------------
acc_m, f1_m = accuracy_score(y_test, y_pred_m), f1_score(y_test, y_pred_m, average='macro')
acc_b, f1_b = accuracy_score(y_test, y_pred_b), f1_score(y_test, y_pred_b, average='macro')

print("\n=== Multinomial NB ===")
print("Accuracy:", acc_m, " F1:", f1_m)
print(classification_report(y_test, y_pred_m))

print("=== Bernoulli NB ===")
print("Accuracy:", acc_b, " F1:", f1_b)
print(classification_report(y_test, y_pred_b))

# ---------------- STEP 6: PLOT ----------------
results = pd.DataFrame({'Accuracy': [acc_m, acc_b], 'F1-score': [f1_m, f1_b]},
                       index=['Multinomial NB', 'Bernoulli NB'])
results.plot(kind='bar', ylim=(0, 1), rot=0)
plt.title("Multinomial vs Bernoulli Naive Bayes")
plt.show()
