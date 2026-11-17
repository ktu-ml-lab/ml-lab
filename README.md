# Machine Learning Lab (PCCSL508, KTU 2024 scheme) - Exp 1, 2, 3, 7, 8, 11, 13, 17

## Use on any lab PC (no pip install needed)
```
git clone <this repo link>
cd ml-lab
python exp1.py
```
Run the programs from inside the folder so the data files are found.
Only uses numpy, pandas, matplotlib and scikit-learn (already on lab PCs).
No TensorFlow, no seaborn, no internet needed with the default options.

## Programs
| File | Experiment | Option A (library) | Option B (CSV) | Option C (.data) |
|---|---|---|---|---|
| exp1.py | Linear regression: gradient descent + normal equation | fetch_california_housing | housing.csv | housing.data |
| exp2.py | Polynomial vs linear regression | fetch_openml('autoMpg') | auto-mpg.csv | auto-mpg.data |
| exp3.py | Ridge / Lasso vs linear (CV tuning) | load_diabetes | diabetes.csv | diabetes.data |
| exp7.py | Multinomial vs Bernoulli Naive Bayes | fetch_20newsgroups | newsgroups.csv | newsgroups.data |
| exp8.py | KNN with different K | keras fashion_mnist (only if TensorFlow exists) | fashion-mnist_train.csv | fashion.data |
| exp11.py | Linear SVM, boundary + margin | load_iris | iris.csv | iris.data |
| exp13.py | MLP (sklearn MLPClassifier), different architectures | fetch_openml('wine-quality-red') | WineQT.csv | winequality.data |
| exp17.py | K-Means, inertia + silhouette | load_digits | digits.csv | digits.data |

Every program has **STEP 1: LOAD DATA** with OPTION A / B / C.
Keep one option, comment the others (select lines, `Ctrl + /`).
Default: library option for exp3, exp11, exp17 (built in, offline); CSV for the rest.
Option A of exp1, exp2, exp7, exp13 downloads the data, so it needs internet once.

Shorter versions: exp1_short.py, exp2_simple.py, exp3_simple.py, exp8_simple.py,
exp11_simple.py, exp11_memory.py, exp13_simple.py, exp17_simple.py.
exp13_keras.py is the optional TensorFlow/Keras version of Exp 13.

## The .data files
| File | Separator | Header? | Text values / quirks | How it is read |
|---|---|---|---|---|
| housing.data | comma | no | last column `ocean_proximity` is text | `header=None, names=cols` |
| auto-mpg.data | spaces | no | `?` = missing, car name is text | `sep=r'\s+', header=None, names=cols, na_values='?'` |
| diabetes.data | tab | yes | target is `Y`, values NOT scaled | `sep='\t'` then StandardScaler |
| newsgroups.data | tab | no | label is text topic name | `sep='\t', header=None, quoting=3` + LabelEncoder |
| fashion.data | comma | no | first column is text class name | `header=None`, map name -> number |
| iris.data | comma | no | species is text (`Iris-setosa`) | `header=None, names=...`, map name -> number |
| winequality.data | semicolon | yes | no Id column | `sep=';'` |
| digits.data | comma | no | last column is the digit | `header=None`, `iloc[:, :-1]` |

See LOADING_GUIDE.md for the general method for any unknown file.
