# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.0Find the accuracy of the model and predict the required values by importing the required module from sklearn.. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ARAVIND SAMY.P
RegisterNumber:  212222230011
*/
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```


## Output:
## Data Head:
![318726910-eaa5731d-9413-4638-92e4-c93c6b1837fd](https://github.com/Aravindsamy04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497037/479d932b-24f4-4c4a-8fa9-1d88e7646979)




## Information:
![318729393-bb81a1fb-5f96-441a-852b-fc791083a78d](https://github.com/Aravindsamy04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497037/2c565d65-ab93-44e2-8a22-444ba4017183)


## NullDataset:
![318729858-30e091bd-2df2-4930-b56f-0f69cf8c0f26](https://github.com/Aravindsamy04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497037/368af0de-81d0-413b-b133-e3b7e5e0efe8)


## ValueCounts:
![318730546-0da66ff6-6869-402a-8418-8ad19e34c562](https://github.com/Aravindsamy04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497037/6998ce4b-1cff-43e3-8cad-3f8e6aeb7713)



## Datahead:

![318730908-9338acd0-9727-4630-8a7c-d6b9ebf93ad7](https://github.com/Aravindsamy04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497037/d0fbe9f9-631c-4b4c-9000-1bd683e095be)


## x.head:
![318731165-10f83be2-a3d1-435e-951d-8f6013f3ea11](https://github.com/Aravindsamy04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497037/7a398791-e8c4-4d24-8f17-c8dea99c7712)


## Accuracy:
![318731747-de857352-0986-44e4-8a96-e42b9dd72deb](https://github.com/Aravindsamy04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497037/0f469dad-1c2c-40b2-bf6b-6bf8b800978a)


## Data Prediction:
![318732140-5ab06940-d549-499d-82ce-2ca8d1b469bd](https://github.com/Aravindsamy04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497037/a77b93ba-894b-4141-a13a-218676f57a60)













## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
