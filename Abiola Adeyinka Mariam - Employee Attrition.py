#!/usr/bin/env python
# coding: utf-8

# # Step 1 : Import required python libraries


# Importing python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp


# # Step 2: Import the dataset for employee attrition given by Company X


#importing dataset
excel_file = 'Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx'
left_emp = pd.read_excel(excel_file, sheet_name = 'Employees who have left')
exist_emp = pd.read_excel(excel_file, sheet_name = 'Existing employees')



left_emp.head()

exist_emp.head()


#adding attrition column for whether the employee left or not 
# 1 for the employees that left and 0 for the employees that still exist
left_emp['attrition'] = 1
exist_emp['attrition'] = 0


#concatenating the two dataset together by the rows to form 1 dataset
employee = pd.concat([left_emp, exist_emp], axis = 0)
employee.describe()


# # Step 3: Data Analysis : Explorative Analysis And Visualizations

#renaming columns
employee.columns = employee.columns.str.strip().str.lower().str.replace(' ', '_')
employee.head()


left_emp['salary'].value_counts() #count of employees that left based on their salary


#salary countplot for employees that left
sns.countplot(left_emp['salary'])
plt.title('Salaries of Employees who left')
plt.savefig('salary_left plot.png', bbox_inches='tight')
plt.show()


##### Employees with low and medium salaries left the most


#salary countplot for employees existing
sns.countplot(exist_emp['salary'])
plt.title('Salaries of Employees existing')
plt.savefig('salary_exist plot.png', bbox_inches='tight')
plt.show()


##### Most of the existing employees have a low or medium salary


left_emp['dept'].value_counts() #count of employees that left based on their departments

#department countplot for employees that left
plt.figure(figsize=(14,7))
sns.countplot(left_emp['dept'])
plt.title('Departments of Employees who left')
plt.savefig('dept_left plot.png', bbox_inches='tight')
plt.show()


# #### Employees in sales, technical and support department left the most 
# #### Management department employees were the least to leave


#deptartment countplot for employees existing
plt.figure(figsize=(14,7))
sns.countplot(exist_emp['dept'])
plt.title('Departments of Employees existing')
plt.savefig('dept_exist plot.png', bbox_inches='tight')
plt.show()


#description of columns and rows,the shape of the dataframe and their datatypes
employee.info()



#Employyees by attrition status
sns.countplot(employee['attrition'])
plt.title('Count of employees that left and exist')
plt.savefig('Exist and non exist employees.png', bbox_inches='tight')


#percentage of exployeees that left and are existing
print(round((employee['attrition'].value_counts()/14999)*100,2)) 


# ### The percentage of employees that left is about 24%


#label encoding of the categorical features to numerical discrete values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
employee['salary']= le.fit_transform(employee['salary'])
employee['dept']= le.fit_transform(employee['dept'])
employee.head()


# salary: 
# low = 1, 
# medium =2, 
# high = 0.

#visualization of the correlation between features and target value
sns.set(font_scale=1)
plt.figure(figsize=(12,10))
sns.heatmap(employee.corr(),annot=True)
plt.xticks(rotation=90)
plt.title('Correlation of features and target value', fontsize = 18)
plt.savefig('corr.png', bbox_inches='tight')
plt.show()


# ##### Correlation shows that satisfaction level has major role in employee attrition. 
# The next step is to figure out why the employees are not satisfied.
# ###### Used tableau to visualize why employees are not satisfied

# # Step 4: MODEL SELECTION AND EVALUATION


#dividing dataset into independent (X) variables and dependent (Y) variables
# X = employee.iloc[:, 1:-1]
X = employee.drop(['emp_id','attrition'],axis=1)
y = employee['attrition']


X.head()


y.head()


#spliting the dataset to train and test set using 30% of data for the test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)


#importing machine learning models
from sklearn.linear_model import LogisticRegression # 1. Logistic Regression Model
from sklearn.naive_bayes import GaussianNB #2. Naive Bayes Model
from sklearn.ensemble import RandomForestClassifier #3. Random Forest Model
from sklearn.svm import SVC #4. Support Vector Machine model
from sklearn.neighbors import KNeighborsClassifier #5. K - Nearest Neighbors Model
from sklearn.tree import DecisionTreeClassifier #6. Decision Tree Model


# ### Building the model and selecting the best model with highest accuracy

# ### 1. LOGISTIC REGRESSION


# Fitting Logistic Regression to Training set
lr = LogisticRegression(random_state=0, solver='lbfgs')
lr.fit(X_train, y_train)

# accuracy for logistic regression model
accuracy_lr = lr.score(X_test, y_test)
print('The accuracy for logistic regression model on the dataset is ', accuracy_lr)

# Confusion Matrix for logistic regression
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test, lr.predict(X_test))
print('The confusion matrix for logistic regression :')
print(cm_lr)


# ### 2. NAIVE BAYES


#fitting naive bayes to training set
nb = GaussianNB()
nb.fit(X_train, y_train)

# accuracy for naive bayes model
accuracy_nb = nb.score(X_test, y_test) 

# Confusion Matrix for naive bayes
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(y_test, nb.predict(X_test))

print('The accuracy for naive bayes model on the dataset is ', accuracy_nb)
print('The confusion matrix for naive bayes :')
print(cm_nb)


# ### 3. RANDOM FOREST


#fitting random forest model to training set

rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rfc.fit(X_train, y_train)

# accuracy for random forest model
accuracy_rfc = rfc.score(X_test, y_test)

# Confusion Matrix for random forest
from sklearn.metrics import confusion_matrix
cm_rfc = confusion_matrix(y_test, rfc.predict(X_test))

print('The accuracy for random forest model on the dataset is ', accuracy_rfc)
print('The confusion matrix for random forest :')
print(cm_rfc)


# ### 4. SUPPORT VECTOR MACHINE


#fitting support vector classifier to training set

svm = SVC(kernel='linear', random_state=0)
svm.fit(X_train, y_train)

#accuracy for svc model
accuracy_svm = svm.score(X_test, y_test)

# Confusion Matrix for svc
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, svm.predict(X_test))

print('The accuracy for support vector machine model on the dataset is ', accuracy_svm)
print('The confusion matrix for support vector machine :')
print(cm_svm)


# ### 5. K - NEAREST NEIGHBORS


#fitting knn model to training set

knc = KNeighborsClassifier(n_neighbors=15)
knc.fit(X_train, y_train)

#accuracy for knn model
accuracy_knc = knc.score(X_test, y_test)

# Confusion Matrix for knn
from sklearn.metrics import confusion_matrix
cm_knc = confusion_matrix(y_test, knc.predict(X_test))

print('The accuracy for k-nearest neighbor model on the dataset is ', accuracy_knc)
print('The confusion matrix for k-nearest neighbor :')
print(cm_knc)


# ### 6. DECISION TREE


#fitting decision tree classifier to training set

dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(X_train, y_train)

#accuracy for decision tree model
accuracy_dtree = dtree.score(X_test, y_test)

# Confusion Matrix for decision tree
from sklearn.metrics import confusion_matrix
cm_dtree = confusion_matrix(y_test, dtree.predict(X_test))

print('The accuracy for decision tree model on the dataset is ', accuracy_dtree)
print('The confusion matrix for decision tree :')
print(cm_dtree)


# ### After comparing the accuracy for all the models, Random forest is the best fit for the problem with an accuracy of about 98.9%


# Model evaluation for Random forest
from sklearn.metrics import classification_report
evaluation  = classification_report(y_test, rfc.predict(X_test))
y_pred_rfc = rfc.predict(X_test) 
y_pred_rfc


# ### Feature Importance

#feature importance to show the features that affect the target value using the random forest model
feature_importance = pd.Series(rfc.feature_importances_,index=X.columns)
plt.figure(figsize=(14,7))
feature_importance = feature_importance.nlargest(9)
feature_importance.plot(kind='bar')
plt.title('Feature Importance', fontsize=20)
plt.savefig('FeatureImportance.png', bbox_inches='tight')


# ### From the plot the top 3 features that have importance in the random forest model are satisfaction level, time spent in company, number of projects

# # Step 5. Determine the employees that are prone to leave next uaing Random Forest Model


#dividing dataset into independent (X) variables and dependent (Y) variables
# X = employee.iloc[:, 1:-1]
X = employee.drop(['attrition'],axis=1)
y = employee['attrition']

#spliting the dataset to train and test set using 30% of data for the test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

#fitting random forest model to training set

rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rfc.fit(X_train.drop('emp_id',axis=1), y_train)
y_pred_rfc = rfc.predict(X_test.drop('emp_id', axis=1))

# accuracy for random forest model
accuracy_rfc = rfc.score(X_test.drop('emp_id', axis=1), y_test)

# Confusion Matrix for random forest
from sklearn.metrics import confusion_matrix
cm_rfc = confusion_matrix(y_test, y_pred_rfc)

print('The accuracy for random forest model on the dataset is ', accuracy_rfc)
print('The confusion matrix for random forest :')
print(cm_rfc)


# Finding employees who are prone to leave from employees that still exist

test_set = pd.concat([y_test,X_test], axis=1)
y_pred = pd.DataFrame(y_pred_rfc) #change the predicted variable from array to a dataframe

y_pred.reset_index(inplace=True, drop=True) 

prediction = pd.concat([test.reset_index(),y_pred], axis = 1) #merge the test set and the predicted variables

Emp = prediction[prediction.attrition==0] #store all the employees that still exist in variable emp
Emp = Emp.drop('index', axis=1)

Emp.columns=['attrition','emp_id','PredictedAttrition'] #change the column names

Employees_prone_to_leave=Emp[Emp['PredictedAttrition']==1] #store employees that were predicted to leave by the model but still exist
Employees_prone_to_leave=Employees_prone_to_leave.reset_index()
Employees_prone_to_leave=Employees_prone_to_leave.drop(['attrition','PredictedAttrition','index'],axis=1) #remove all features except employee id
Employees_prone_to_leave #employees that are predicted to leave in the future.


#Details of employees that were predicted to leave
emp_prone_to leave_details = pd.merge(Employees_prone_to_leave, employee, how='inner', on='emp_id')
emp_prone_to leave_details 






