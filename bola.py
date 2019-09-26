import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

data=pd.read_csv('data.csv')

target=[]
for item1,item2,item3 in zip(data['Age'],data['Overall'],data['Potential']):
    if item1<=25 and item2>=80 and item3>=80:
        target.append(1)
    else:
        target.append(0)
data['Target']=target
x=data[['Age','Overall','Potential']]
y=data['Target']
#Model
from sklearn.neighbors import KNeighborsClassifier
def nilai_k():
    k=round(len(x)**.5)
    if k%2==0:
        return k+1
    else:
        return k
modelKNN=KNeighborsClassifier(
    n_neighbors=nilai_k()
)
modelKNN.fit(x,y)
from sklearn import tree 
modelDT=tree.DecisionTreeClassifier()
modelDT.fit(x,y)
from sklearn.ensemble import RandomForestClassifier
modelRFC=RandomForestClassifier(
    n_estimators=10
    )
modelRFC.fit(x,y)
#Cross Validation
print(np.mean(cross_val_score(
   KNeighborsClassifier(n_neighbors=nilai_k()),x,y))
)
print(np.mean(cross_val_score(
   tree.DecisionTreeClassifier(),x,y))
)
print(np.mean(cross_val_score(
   RandomForestClassifier(n_estimators=10),x,y)
))

#Model terbaik= Random forest classifier 0.9674301092986214

pemain=pd.DataFrame([
    {'Name':'Andik Vermansyah','Age':27,'Overall':87,'Potential':90},
    {'Name':'Awan Setho Raharjo','Age':22,'Overall':75,'Potential':83},
    {'Name':'Bambang Pamungkas','Age':38,'Overall':85,'Potential':75},
    {'Name':'Cristian Gonzales','Age':43,'Overall':90,'Potential':85},
    {'Name':'Egy Maulana Vikri','Age':18,'Overall':88,'Potential':90},
    {'Name':'Evan Dimas','Age':24,'Overall':85,'Potential':87},
    {'Name':'Febri Hariyadi','Age':23,'Overall':77,'Potential':80},
    {'Name':'Hansamu Yama Pranata','Age':24,'Overall':82,'Potential':85},
    {'Name':'Septian David Maulana','Age':22,'Overall':83,'Potential':90},
    {'Name':'Stefano Lilipaly','Age':29,'Overall':88,'Potential':86}]
)
# print(pemain)
#Predict menggunakan Random Forest Classifier
pemain['Target']=modelRFC.predict(pemain.drop('Name',axis=1))
pemain['Target']=pemain['Target'].apply(lambda i: 'Target' if i==1 else 'Non Target')
print(pemain)