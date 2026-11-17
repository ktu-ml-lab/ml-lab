# Machine Learning Lab (PCCSL508, KTU 2024 scheme) - Exp 1, 2, 3, 7, 8, 11, 13, 17

## Setup on a new laptop
```
git clone <this repo link>
cd <repo folder>
pip install -r requirements.txt
python exp1.py
```
Run the programs from inside the repo folder so the data files are found.
If TensorFlow is not available, use `exp13_sklearn.py` for Exp 13.
If the C: drive is full, make a virtual environment on another drive first:
```
python -m venv D:\mlenv
D:\mlenv\Scripts\activate
pip install -r requirements.txt
```

## Programs
| File | Experiment | Option A (library) | Option B (CSV) | Option C (.data) |
|---|---|---|---|---|
| exp1.py | Linear regression: gradient descent + normal equation | fetch_california_housing | housing.csv | housing.data |
| exp2.py | Polynomial vs linear regression | fetch_openml('autoMpg') | auto-mpg.csv | auto-mpg.data |
| exp3.py | Ridge / Lasso vs linear (CV tuning) | load_diabetes | diabetes.csv | diabetes.data |
| exp7.py | Multinomial vs Bernoulli Naive Bayes | fetch_20newsgroups | newsgroups.csv | newsgroups.data |
| exp8.py | KNN with different K | keras fashion_mnist | fashion-mnist_train.csv | fashion.data |
| exp11.py | Linear SVM, boundary + margin | load_iris | iris.csv | iris.data |
| exp13.py | MLP (Keras), different architectures | fetch_openml('wine-quality-red') | WineQT.csv | winequality.data |
| exp17.py | K-Means, inertia + silhouette | load_digits | digits.csv | digits.data |

Every program has **STEP 1: LOAD DATA** with OPTION A / B / C.
Keep one option, comment the others (select lines, `Ctrl + /` in VS Code).
Default active option: library for exp3, exp11, exp17 (built in, no internet);
CSV for the rest (Option A for those needs internet the first time).

Shorter versions (fewer features, easier to memorise): exp1_short.py, exp2_simple.py, exp3_simple.py, exp8_simple.py, exp11_simple.py, exp11_memory.py, exp13_simple.py, exp17_simple.py.

## The .data files (what is special about each)
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
