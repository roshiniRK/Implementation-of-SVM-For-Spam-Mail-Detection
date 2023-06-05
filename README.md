# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library packages.
2. Import the dataset to operate on.
3. Split the dataset into required segments.
4. Predict the required output.
5. Run the program

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: ROSHINI R K
RegisterNumber: 212222230123 
*/
import chardet 
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.isnull().sum()
data.head()
data.info()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707332/21b9ff7c-8f49-474b-8127-a0206f1ee89e)
### data.head():
![image](https://github.com/SASIDEVIvenaram/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707332/2e4f16be-3aa6-458b-a8b3-8fe44e06db1a)
### data.info():
![image](https://github.com/SASIDEVIvenaram/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707332/00a6d72d-e64a-4334-b1c8-b978bc5b9335)
### data.isnull().sum():
![image](https://github.com/SASIDEVIvenaram/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707332/a26144eb-819b-4726-8501-1a2db538ee84)
### Y_prediction value:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707332/f4797ac7-09b2-4ece-877a-28619baf0fc8)
### Accuracy value:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707332/4777dd1f-2e78-4db3-8568-1f352d201539)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
