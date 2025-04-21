# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values
## Program And Output:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SURIYA PRAKASH.S
RegisterNumber:  212223100055

```
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()
```
![image](https://github.com/user-attachments/assets/782a8858-e1a6-4f90-bdf6-1f8651a97db3)

```
data1=data.copy()
data1=data1.drop(['salary'],axis=1)
data1.head()
```
![image](https://github.com/user-attachments/assets/beb97d92-90d4-4492-b6a0-5c907a9dbcdb)

```
data1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/86788893-b496-4d55-b1ea-e201f0f32a2c)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```

![image](https://github.com/user-attachments/assets/2e329392-b1a9-4f81-82f7-0eb704df2f59)


```
x=data1.iloc[:,:-1]
x

```
![image](https://github.com/user-attachments/assets/3b6edabe-8c64-491b-8382-10326d6d4322)

```
y=data1["status"]
y
```
![image](https://github.com/user-attachments/assets/cd3420b9-0534-4e55-b756-955485f8eb97)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear')
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```

![image](https://github.com/user-attachments/assets/3fadc001-dea5-4126-b49d-3c4ddf472d06)

```
from sklearn.metrics import confusion_matrix
accuracy=confusion_matrix(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/0268265b-72d1-4f60-9f37-a7425122f496)

```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```
![image](https://github.com/user-attachments/assets/79c1019d-ffbd-4bc6-8c7c-f3a10be8ead9)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
