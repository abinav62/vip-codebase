import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
from ISLP import load_data , confusion_table
from sklearn.svm import SVC
from ISLP.svm import plot as plot_svm
from sklearn.metrics import (RocCurveDisplay, classification_report,confusion_matrix, accuracy_score)
from ISLP.models import ModelSpec as MS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("Downloads/master_without_15.csv") #Replace File Path here
df['date'] = pd.to_datetime(df['date'])

class_distribution = df['exit'].value_counts()

#Visualzing the Data

colors = {True: 'green', False: 'red'}

color_mapping = [colors[rank] for rank in df['exit']]
ax = plt.axes()
ax.set_facecolor("#ADD8E6")

plt.scatter(x=df["open"], y=df["rsi_7"],c=color_mapping,)
fig = plt.gcf()
plt.xlabel('Open', fontweight = "bold")
plt.ylabel('RSI_7',fontweight = "bold")
legend_labels = {True: 'Yes', False: 'No'}
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[rank], markersize=10, label=legend_labels[rank]) for rank in sorted(df['exit'].unique())]
plt.legend(handles=legend_elements, title='Exit')
plt.title("Open vs RSI_7",fontweight = "bold")
plt.show()


# Plotting the class distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar', color='skyblue')
plt.title('Class Distribution', fontweight = 'bold')
plt.xlabel('Class',fontweight = 'bold')
plt.ylabel('Count',fontweight = 'bold')
plt.show()

df2 = df.drop(columns=["exit"])
dates = df['date']



X = df2.drop(columns=["date"])
Y = df["exit"]


#Training the Data

(XTrain, XTest, YTrain, YTest) = skm.train_test_split(X,Y, train_size=0.5,random_state=42)



scaling = MinMaxScaler(feature_range=(-1,1)).fit(XTrain)
X_train = scaling.transform(XTrain)
X_test = scaling.transform(XTest)

svm_linear = SVC(kernel="poly",C=0.01, degree = 2) #Change the kernel Here
svm_linear.fit(XTrain,YTrain)
Ypred = svm_linear.predict(XTest)


print(f'Accuracy: {accuracy_score(YTest, Ypred)}')
print(f'Confusion matrix: {confusion_matrix(YTest, Ypred)}')
print(f'Classification report: {classification_report(YTest, Ypred)}')


#Plotting the decision boundry for SVM
fig, ax = plt.subplots(figsize=(8,8)) 
plot_svm(XTrain,YTrain, svm_linear , features = (4,10),ax=ax)
plt.title("C=0.01 Poly, Degree=2",fontweight = "bold")
plt.xlabel("SPXP Open", fontweight = "bold")
plt.ylabel("RSI_7", fontweight = "bold")
plt.show()

#Cross Validation

kfold = skm.KFold(5, random_state = 0, shuffle = True)

gridN = skm.GridSearchCV(svm_linear, {"degree": [1,2,3,4,5,6,7,8,9,10]}, refit = True, cv=kfold, scoring="accuracy")
gridN.fit(X,Y)
print(gridN.cv_results_[('mean_test_score')])
print(gridN.best_params_)







