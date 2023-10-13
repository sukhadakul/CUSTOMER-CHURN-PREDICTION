#!/usr/bin/env python
# coding: utf-8

# # TASK 2  Machine Learning Intern @Codsoft 

# # CUSTOMER CHURN PREDICTION
# OBJECTIVES:- Develop a model to predict customer churn for a subscription-based service or business. Use historical customer data, including features like usage behavior and customer demographics, and try algorithms like Logistic Regression, Random Forests, or Gradient Boosting to predict churn.

# # 1) Import necessary libraries 🐍 

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sn 
import sklearn.svm as SVM 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score , classification_report  
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# # 2) Load Dataset & Data Preprocessing 🔄🔢

# In[2]:


Data=pd.read_csv("Churn_Modelling.csv") 
Data 


# In[3]:


#check shape of Dataset 
Data.shape 


# In[4]:


Data.columns 


# In[5]:


# Check sample data in our Dataset
Data.sample()  


# # 3) EDA (Exploratory Data Analysis )🚀 

# In[6]:


#Check info of Dataset 
Data.info()  


# In[7]:


#Check null value in our Dataset 
Data.isnull().sum()  


# No null value in our dataset 

# In[8]:


#Check Duplicated values in our dataset 
Data.duplicated().sum()   


# No duplicated value in our dataset 

# In[9]:


# Summary statistics 
Data.describe()   


# In[10]:


# Class distribution
Data['Exited'].value_counts()  


# # 4)Data Visualization 📊📈📉

# In[11]:


plt.figure(figsize=(15,5))
sn.countplot(data=Data,x='Exited')      


# In[12]:


Data.drop(['RowNumber', 'CustomerId', 'Surname','Geography','Gender'],axis=1,inplace=True) 


# In[13]:


Data.head(3) 


# In[14]:


Data.corr() 


# # Heatmap 🌡️

# In[15]:


plt.figure(figsize=(15,5))
sn.heatmap(Data.corr(),annot=True) 


# In[16]:


Data_corr_exit=Data.corr()['Exited'].to_frame()  
Data_corr_exit 


# # Barplot📊 

# In[17]:


plt.figure(figsize=(15,5))
sn.barplot(data=Data_corr_exit,x=Data_corr_exit.index,y='Exited') 


# # 5) Models Training / Building 📈 

# In[18]:


x=Data.drop(['Exited'],axis=1)
y=Data['Exited'] 


# In[24]:


# Standardize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[21]:


#Spliting the Data Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[22]:


#LogisticRegression model 
lr=LogisticRegression(max_iter=500)
lr.fit(x_train,y_train)  


# In[23]:


# Train a Random Forest model
rf_model = RandomForestClassifier() 
rf_model.fit(x_train, y_train) 


# In[25]:


# Train a Gradient Boosting model
gb_model = GradientBoostingClassifier()
gb_model.fit(x_train, y_train)  


# In[26]:


# Evaluate the models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Confusion Matrix:\n", cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


# In[27]:


print("Logistic Regression Model:")
evaluate_model(lr, x_test, y_test)


# In[28]:


print("Random Forest Model:")
evaluate_model(rf_model, x_test, y_test) 


# In[29]:


print("Gradient Boosting Model:")
evaluate_model(gb_model, x_test, y_test) 


# # 6) Accuracy 🎯💯🚀 

# # I got a Accuracy Score: 0.86 (86%)  
