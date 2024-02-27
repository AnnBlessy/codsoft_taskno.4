## TASK NO : 4
# SPAM SMS DETECTION
![image](https://github.com/AnnBlessy/codsoft_taskno.4/assets/119477835/63e55827-a669-4e58-ac88-d2b5e9ddcdbc)

## CODE
```
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC 

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
```
```
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
```
```
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.4/assets/119477835/6414eef4-0de8-471d-a835-565d2b67d61a)

```
data.info()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.4/assets/119477835/effe1e62-d1e7-472c-8f54-03c027515209)

```
#Data preprocessing
data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"] ,axis = 1 ,inplace = True)
data.head()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.4/assets/119477835/09b86714-4c23-41f7-aeb2-1b3d6270af25)

```
# function to preprocess the data
stopword = set(stopwords.words('english'))
def preprocessing(text):
    text = text.lower()    # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation
    text = re.sub(r'[#@\$]', '', text)    # Remove specific characters #, @, and $
    tokens = text.split()
    text = [token for token in tokens]
    text = [word for word in text if word not in stopword]
    return " ".join(text)
data["cleaned_sms"] =  data["v2"].apply(preprocessing)
data.head()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.4/assets/119477835/acb0b09b-e852-4135-a618-75e47cac7f25)

```
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['encoded_target'] = label_encoder.fit_transform(data['v1'])

class_names= list(label_encoder.classes_)
class_name
data.head()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.4/assets/119477835/f299296d-576d-4e4a-84a7-e219fcb9e772)

```
# Split the data
x = data["cleaned_sms"]
y = data["encoded_target"]

vectorizer = TfidfVectorizer()
x_trans= vectorizer.fit_transform(x)

x_train ,x_test ,y_train ,y_test = train_test_split(x_trans ,y ,test_size = 0.3 ,random_state = 42)
```
### After splitting the data for training and testing, it is trained with different algorithms.

## Logistic Regression
![image](https://github.com/AnnBlessy/codsoft_taskno.4/assets/119477835/7d8f865b-0bce-4903-a1a4-50f9fd4742fb)

## SVC
![image](https://github.com/AnnBlessy/codsoft_taskno.4/assets/119477835/ca7f0ac4-4681-4ea8-99c4-50c438619eab)
