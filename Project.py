#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np


# In[49]:


#import dataset using pandas
diamond_data = pd.read_csv()
diamond_data.head()


# In[50]:


#converting non-numeric values to numeric
diamond_data['Cut'],_ = pd.factorize(diamond_data['Cut'])  
diamond_data['Color'],_ = pd.factorize(diamond_data['Color'])  
diamond_data['Clarity'],_ = pd.factorize(diamond_data['Clarity'])  
diamond_data.head()


# In[51]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data_correlation = diamond_data.drop(columns = ['Multi_Class', 'Binary_Class'])

mask = np.zeros_like(data_correlation.corr())
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots()
fig.set_size_inches(14, 10)

ax = sns.heatmap(data_correlation.corr(), annot = True, mask = mask)


# In[53]:


#Preparing the data set for Classifier
diamond_all = list(diamond_data.shape)[0]
diamond_categories = list(diamond_data['Multi_Class'].value_counts())

print("The dataset has {} diamonds.".format(diamond_all))
print("Category 1 has {} diamonds".format(diamond_categories[0]))
print("Category 2 has {} diamonds".format(diamond_categories[1]))
print("Category 3 has {} diamonds".format(diamond_categories[2]))
print("Category 4 has {} diamonds".format(diamond_categories[3]))
print("Category 5 has {} diamonds".format(diamond_categories[4]))


# In[54]:


#Spliting data into test and training set for uniformity
from sklearn.model_selection import train_test_split

Training_Data ,Test_Data = train_test_split(diamond_data,test_size=0.1)


# In[55]:


#Creating training and test datasets for Classifier
X_train_Clf = Training_Data.drop(["Price","Multi_Class","Binary_Class"], axis = 1)
y_train_Clf = Training_Data.Multi_Class
X_test_Clf = Test_Data.drop(["Price","Multi_Class","Binary_Class"], axis = 1)
y_test_Clf = Test_Data.Multi_Class


# In[59]:


#KNN Classifier algorithm
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix 

classifier_KNN = KNeighborsClassifier(n_neighbors=10)  
classifier_KNN.fit(X_train_Clf, y_train_Clf) 

y_pred_KNN = classifier_KNN.predict(X_test_Clf)  
print('####### KNN Classifier #######')
print('Accuracy: %0.4f '%np.mean(y_pred_KNN == y_test_Clf))
print('')
print('Confusion Matrix:')
print(confusion_matrix(y_test_Clf, y_pred_KNN))  
print('')
print('Classification Report:')
print(classification_report(y_test_Clf, y_pred_KNN))  


# In[57]:


import matplotlib.pyplot as plt

error = []

# Calculating error for K values between 1 and 100
for i in range(1, 100):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_Clf, y_train_Clf)
    pred_i = knn.predict(X_test_Clf)
    error.append(np.mean(pred_i != y_test_Clf))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 

plt.show()


# In[60]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier_RF=RandomForestClassifier(n_estimators=500)
classifier_RF.fit(X_train_Clf,y_train_Clf)

y_pred_RF = classifier_RF.predict(X_test_Clf)  
print('####### Random Forest Classifier #######')
print('Accuracy: %0.4f '%np.mean(y_pred_RF == y_test_Clf))
print('')
print('Confusion Matrix:')
print(confusion_matrix(y_test_Clf, y_pred_RF))  
print('')
print('Classification Report:')
print(classification_report(y_test_Clf, y_pred_RF)) 


# In[39]:


feature_cols = ['Carat', 'Cut', 'Color', 'Clarity','Total_depth','Table','Length','Width','Depth']
feature_imp = pd.Series(classifier_RF.feature_importances_,index=feature_cols).sort_values(ascending=False)
print('Feature Importance:')
print(feature_imp)

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[64]:


X_train_Clf_upd = X_train_Clf.drop(columns = ['Cut', 'Table'])
X_test_Clf_upd = X_test_Clf.drop(columns = ['Cut', 'Table'])

classifier_RF_upd=RandomForestClassifier(n_estimators=600)
classifier_RF_upd.fit(X_train_Clf_upd,y_train_Clf)

y_pred_RF_upd = classifier_RF_upd.predict(X_test_Clf_upd)  
print('####### Random Forest Classifier #######')
print('Accuracy: %0.4f '%np.mean(y_pred_RF_upd == y_test_Clf))
print('')
print('Confusion Matrix:')
print(confusion_matrix(y_test_Clf, y_pred_RF_upd))  
print('')
print('Classification Report:')
print(classification_report(y_test_Clf, y_pred_RF_upd)) 


# In[69]:


#Preparing the data set for Logistic Regression
diamond_all = list(diamond_data.shape)[0]
diamond_categories = list(diamond_data['Binary_Class'].value_counts())

print("The dataset has {} diamonds , {} of 0 and {} of 1.".format(diamond_all, 
                                                                                 diamond_categories[0], 
                                                                                 diamond_categories[1]))


# In[70]:


#Creating training and test datasets for Logistic Regression
X_train_LR = Training_Data.drop(["Price","Multi_Class","Binary_Class"], axis = 1)
y_train_LR = Training_Data.Binary_Class
X_test_LR = Test_Data.drop(["Price","Multi_Class","Binary_Class"], axis = 1)
y_test_LR = Test_Data.Binary_Class


# In[71]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

Regression_Logistic = LogisticRegression(class_weight = 'balanced')
Regression_Logistic.fit(X_train_LR, y_train_LR)

y_pred_LR = Regression_Logistic.predict(X_test_LR)

print('####### Logistic Regression #######')
print('Accuracy : %.4f' % accuracy_score(y_test_LR, y_pred_LR))

print('Coefficients: \n', Regression_Logistic.coef_)
print('')
print('MSE    : %0.2f ' % mean_squared_error(y_test_LR, y_pred_LR))
print('R2     : %0.2f ' % r2_score(y_test_LR, y_pred_LR))
print('Log Loss: %.2f' %log_loss(y_test_LR, y_pred_LR))

print('')
print('Confusion Matrix:')
print(confusion_matrix(y_test_LR, y_pred_LR))  
print('')
print('Classification Report:')
print(classification_report(y_test_LR, y_pred_LR)) 


# In[72]:


#Creating training and test datasets for Regression
X_train_R = Training_Data.drop(["Price","Multi_Class","Binary_Class"], axis = 1)
y_train_R = Training_Data.Price
X_test_R = Test_Data.drop(["Price","Multi_Class","Binary_Class"], axis = 1)
y_test_R = Test_Data.Price


# In[74]:


#Linear Regression
from sklearn.linear_model import LinearRegression

Regression_Linear = LinearRegression()
Regression_Linear.fit(X_train_R , y_train_R)
y_pred_LiR = Regression_Linear.predict(X_test_R)

print('####### Linear Regression #######')
print('Accuracy : %.4f' % Regression_Linear.score(X_test_R, y_test_R))

print('Coefficients: \n', Regression_Linear.coef_)
print('')
print('MSE    : %0.2f ' % mean_squared_error(y_test_R, y_pred_LiR))
print('R2     : %0.2f ' % r2_score(y_test_R, y_pred_LiR))


# In[75]:


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

Regression_RF = RandomForestRegressor()
Regression_RF.fit(X_train_R , y_train_R)

y_pred_RF = Regression_RF.predict(X_test_R)

print('###### Random Forest ######')
print('Accuracy : %.4f' % Regression_RF.score(X_test_R, y_test_R))

print('Coefficients: \n', Regression_Linear.coef_)
print('')
print('MSE    : %0.2f ' % mean_squared_error(y_test_R, y_pred_RF))
print('R2     : %0.2f ' % r2_score(y_test_R, y_pred_RF))


# In[ ]:




