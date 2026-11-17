# Loading any data file - cheat sheet

## 4 questions (open the file in Notepad first)
1. Separator?    comma -> default | ; -> sep=';' | tab -> sep='\t' | spaces -> sep=r'\s+'
2. Header row?   no -> header=None, names=[...]
3. Missing?      '?' -> na_values='?' then df.dropna()
4. Label column? by name -> df.drop('col', axis=1) | last column -> df.iloc[:, :-1], df.iloc[:, -1]
                 first column -> df.iloc[:, 1:], df.iloc[:, 0]

Then always: print(df.head()); print(df.shape); print(df.isnull().sum())

## Text values
- Text label column  -> y = LabelEncoder().fit_transform(df['col'])
- Or map by hand     -> y = df['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
- Text feature column you don't need -> just don't select it
- Text feature column you need -> pd.get_dummies(df, columns=['col'])  (one-hot)

## Other formats
- Excel -> pd.read_excel("file.xlsx")
- Folder of text files (one subfolder per class) -> sklearn.datasets.load_files("folder", encoding='latin1')
- File in another folder -> pd.read_csv(r"D:\path\file.csv")

## Common errors
| Error | Fix |
|---|---|
| FileNotFoundError | run from the repo folder or use the full path |
| everything in 1 column | wrong separator |
| first row became column names | header=None, names=[...] |
| KeyError: 'col' | print(df.columns) and use the exact name |
| could not convert string to float: '?' | na_values='?' then dropna() |
| could not convert string to float: 'Iris-setosa' | LabelEncoder or map |
| Input contains NaN | df = df.dropna() |
