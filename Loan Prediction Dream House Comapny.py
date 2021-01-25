#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\Users\shubham\Downloads\loan_train.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.dtypes


# In[7]:


df['Credit_History'].value_counts()


# In[8]:


#handling the missing values with most frequent categories for categorical data


# In[9]:


def impute_nan(df,variable):
    most_frequent_category=df[variable].mode()[0]
    df[variable].fillna(most_frequent_category,inplace=True)


# In[10]:


for feature in ['Gender','Married','Dependents','Self_Employed','Credit_History','LoanAmount','Loan_Amount_Term']:
    impute_nan(df,feature)


# In[11]:


#handling missing value for numerical value using knn imputer


# In[12]:


X=df.iloc[:,7:10]
X


# In[13]:


from sklearn.impute import KNNImputer


# In[14]:


imputer=KNNImputer(n_neighbors=2)


# In[15]:


WE=imputer.fit_transform(X)


# In[16]:


df_new=pd.DataFrame(WE)


# In[17]:


df_new


# In[18]:


df_new.columns =['CoapplicantIncome_new', 'LoanAmount_new', 'Loan_Amount_Term_new'] 


# In[19]:


df_new


# In[75]:


df['Loan_Status'].unique()


# In[20]:


final_data=pd.concat([df,df_new],axis=1)
final_data


# In[ ]:





# In[21]:


#from sklearn.impute import KNNImputer
#import numpy as np

#X = [ [3, np.NaN, 5], [1, 0, 0], [3, 3, 3] ]
#print("X: ", X)
#print("===========")


#imputer = KNNImputer(n_neighbors= 1)
#impute_with_1 = imputer.fit_transform(X)

#print("\nImpute with 1 Neighbour: \n", impute_with_1)



#imputer = KNNImputer(n_neighbors= 2)
#impute_with_2 = imputer.fit_transform(X)

#print("\n Impute with 2 Neighbours: \n", impute_with_1)


# In[22]:


final_data.isnull().sum()


# In[23]:




final_data


# In[24]:


final_data['Loan_Amount_Term'].unique()


# In[25]:


#categorical variable converted into  dummy varibale


# In[36]:


final_data.Gender=final_data.Gender.map({'Male':1,'Female':0})
final_data.Married=final_data.Married.map({'Yes':1,'No':0})
final_data.Dependents=final_data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
final_data.Education=final_data.Education.map({'Graduate':1,'Not Graduate':0})
final_data.Self_Employed=final_data.Self_Employed.map({'Yes':1,'No':0})
final_data.Property_Area=final_data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
final_data.Loan_Status=final_data.Loan_Status.map({'Y':1,'N':0})


# In[37]:


final_data


# In[38]:


final_data.isnull().sum()


# In[39]:


final_data.drop(['CoapplicantIncome'],axis=1,inplace=True)
final_data.drop(['LoanAmount'],axis=1,inplace=True)
final_data.drop(['Loan_Amount_Term'],axis=1,inplace=True)


# In[40]:


final_data


# In[41]:


y  = final_data[['Loan_Status']]


# In[42]:


x = pd.concat([final_data.iloc[:,1:9],final_data.iloc[:,-3:]],axis = 1)


# In[70]:


X = final_data.iloc[:,1:9]


# In[64]:


x.shape


# In[65]:


### Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(x,y.values.ravel())


# In[66]:


print(model.feature_importances_)


# In[67]:


#top five features
import seaborn as sns
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[71]:


#model creation
from sklearn.model_selection import train_test_split


# In[72]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
logreg = LogisticRegression()
logreg.fit(x_train, y_train.values.ravel())


# In[73]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[81]:


score = logreg.score(X_test, y_test)
print(score)


# In[ ]:


#we user another model random forest classifier


# In[83]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train.values.ravel())

y_pred=clf.predict(X_test)


# In[84]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




