#!/usr/bin/env python
# coding: utf-8

# ## *Import Libraries*

# In[1]:


#!pip install lightgbm


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## *Load Dataset*

# In[3]:


tc=pd.read_csv('telecommunications_churn.csv')
tc


# ## ***EDA***

# In[4]:


# Overview of the dataset

tc.info()


# In[5]:


tc.columns


# In[6]:


#Display Summary

tc.describe()


# In[7]:


# Checking missingvalues

tc.isnull().sum()


# In[8]:


#checking datatypes

tc.dtypes


# In[9]:


# Checking Duplicates

tc.duplicated().sum()


# ## ***Data Visualization***

# In[10]:


numerical_cols = tc.select_dtypes(include=np.number).columns


# In[11]:


#Histogram
selected_features=['account_length','day_mins','evening_mins','night_mins','international_mins','day_calls','day_charge','evening_calls','evening_charge',
                  'night_calls','night_charge','international_charge','total_charge']
for col in selected_features:
    plt.figure()
    sns.histplot(tc[col], kde=True)
    plt.title(f"Histogram of {col}")
    plt.show()


# In[12]:


# Bar Chart
sns.countplot(x='churn', data=tc)
plt.title('Churn Count Plot')
plt.xlabel('churn')
plt.ylabel('count')


# In[13]:


# correlation

tc.corr()


# In[14]:


# calculates the number of outliers per column using IQR method
outliers_count = ((tc < (tc.quantile(0.25) - 1.5 * (tc.quantile(0.75) - tc.quantile(0.25)))) | (tc > (tc.quantile(0.75) + 1.5 * (tc.quantile(0.75) - tc.quantile(0.25))))).sum()
outliers_count


# In[15]:


# Box Plot

for col in numerical_cols:
  plt.figure()
  sns.boxplot(tc[col])
  plt.title(f"Boxplot of {col}")
  plt.show()


# In[16]:


# Treating Outlier
def outlier_Detection(tc,columns):
    for col in columns:
        Q1=tc[col].quantile(0.25)
        Q3=tc[col].quantile(0.75)
        iqr=Q3-Q1
        lower=Q1-1.5*iqr
        upper=Q3+1.5*iqr
        tc[col]=np.where(tc[col]>upper,upper,np.where(tc[col]<lower,lower,tc[col]))
    return tc  


# In[17]:


outlier_Detection(tc,['account_length', 'voice_mail_messages', 'day_mins', 'evening_mins', 'night_mins', 'international_mins','customer_service_calls', 'day_calls','day_charge', 'evening_calls', 'evening_charge', 'night_calls', 'night_charge', 'international_calls', 'international_charge', 'total_charge'])


# In[18]:


# Box Plot after removing outliers

for col in numerical_cols:
  plt.figure()
  sns.boxplot(tc[col])
  plt.title(f"Boxplot of {col}")
  plt.show()


# In[19]:


# calculates the number of outliers per column using IQR method
outliers_count = ((tc < (tc.quantile(0.25) - 1.5 * (tc.quantile(0.75) - tc.quantile(0.25)))) | (tc > (tc.quantile(0.75) + 1.5 * (tc.quantile(0.75) - tc.quantile(0.25))))).sum()
outliers_count


# In[20]:


# Scatter plot
target= 'churn'
features= [col for col in tc.columns if col !=target]
for feature in features:
    sns.scatterplot(data=tc, x=feature, y=target)
    plt.title(f"{feature} vs {target}")
    plt.show()


# In[21]:


# Heatmap

plt.figure(figsize=(18,10))
sns.heatmap(tc.corr(),annot=True,cmap='coolwarm')
plt.show()


# In[22]:


# Pair plot

selected_features=['day_mins','evening_mins','night_mins','international_mins','churn']
sns.pairplot(tc[selected_features],hue='churn')
plt.show()


# ## *Feature Engineering*

# In[23]:


# Create the interaction feature

tc['avg_day_call_charge'] = tc['day_mins'] + tc['day_calls']+tc['day_charge']
tc['avg_evening_call_charge']=tc['evening_mins']+tc['evening_calls']+tc['evening_charge']
tc['avg_night_call_charge']=tc['night_mins']+tc['night_calls']+tc['night_charge']


# In[24]:


tc.head()


# In[25]:


# Normalization

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
ts = pd.DataFrame(scalar.fit_transform(tc.drop(columns=['churn'])),columns=tc.columns[:-1])
ts['churn']=tc['churn']


# In[26]:


ts


# ## *Import Libraries*

# In[27]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.cluster import KMeans


# In[28]:


x=ts.drop(['churn'],axis=1)
y=ts['churn']


# In[29]:


x.head()


# In[30]:


y.head()


# In[31]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size = 0.2 , random_state = 1)


# ## *Model Building* 

# In[32]:


# Automatic Model Building

def prediction(model):
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    
    train_acc = model.score(xtrain, ytrain)
    test_acc = model.score(xtest, ytest)
    print(f"Train Accuracy:{model.score(xtrain,ytrain)}")
    print(f"Test Accuracy:{model.score(xtrain,ytrain)}")
    print(classification_report(ytest,y_pred))

    return test_acc


# In[33]:


# Using Logistic Regression

prediction(LogisticRegression())


# In[34]:


# Using Support Vector Machine

prediction(SVC())


# In[35]:


# Using Decision Tree Classifier

prediction(DecisionTreeClassifier())


# In[36]:


# Using Random Forest Classifier

prediction(RandomForestClassifier())


# In[37]:


# Using KNN

prediction( KNeighborsClassifier())


# In[38]:


models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier()
}

# Variable to store the best model and its accuracy
best_model = None
best_accuracy = 0
model_accuracies = {}

# Evaluate each model
for name, model in models.items():
    print(f"Evaluating {name}...")
    acc = prediction(model)  # Assuming prediction function gives accuracy for the model
    model_accuracies[name] = acc
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model  # Store the actual model, not the name
print(f"Best Model: {type(best_model).__name__}")
print(f"with accuracy: {best_accuracy}")


# # *Boost the accuracy*

# In[39]:


# Cross-validation for LightGBM

lgbm_model = lgb.LGBMClassifier()


# In[40]:


lgbm_cv_scores = cross_val_score(lgbm_model, x, y, cv=5, scoring='accuracy')


# In[41]:


# Print the mean cross-validation score
print(f"LightGBM Mean Cross-Validation Accuracy: {lgbm_cv_scores.mean():.4f}")


# # *Deployment*

# In[42]:


import joblib

joblib.dump(best_model, "Tele-communication_model.pkl")
print ("Model saved as 'Tele-communication_model.pkl'.")


# In[43]:


import nbformat
from nbconvert import PythonExporter


# In[48]:


#Load the notebook

notebook_filename =" Tele-communication (5).ipynb"
with open (notebook_filename, 'r', encoding='utf-8') as notebook_file:
    notebook_content = nbformat.read( notebook_file, as_version=4 )


# In[ ]:


# Convert to Python script

python_exporter = PythonExporter()
python_code, _ = python_exporter.from_notebook_node(notebook_content)


# In[ ]:


# Saving to a '.py' file

python_filename = notebook_filename.replace('.ipynb','.py')
with open(python_filename, 'w', encoding='utf-8') as python_file:
    python_file.write(python_code)

print(f" Notebook converted to {python_filename}")


# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib


# In[ ]:


#Load the trained model

Model_File = 'Tele-communication_model.pkl' # Replacing it with model file name
model = joblib.load(Model_File)


# In[ ]:


# Streamlit App Title
st.title('Telecommunication Model Churn Prediction')
st.write("Enter the Values to predict the churn")


# In[ ]:


# Input fields for churn 

account_length = st.number_input("Account Length", min_value=0)
voice_mail_plan = st.selectbox("Voice Mail Plan", options=[0, 1])  # 0 = No, 1 = Yes
voice_mail_messages = st.number_input("Voice Mail Messages", min_value=0)
day_mins = st.number_input("Day Minutes", min_value=0.0)
evening_mins = st.number_input("Evening Minutes", min_value=0.0)
night_mins = st.number_input("Night Minutes", min_value=0.0)
international_mins = st.number_input("International Minutes", min_value=0)
customer_service_calls = st.number_input("Customer Service Calls", min_value=0)
international_plan = st.selectbox("International Plan", options=[0, 1])  # 0 = No, 1 = Yes
day_calls = st.number_input("Day Calls", min_value=0)
day_charge = st.number_input("Day Charge", min_value=0.0)
evening_calls = st.number_input("Evening Calls", min_value=0)
evening_charge = st.number_input("Evening Charge", min_value=0.0)
night_calls = st.number_input("Night Calls", min_value=0)
night_charge = st.number_input("Night Charge", min_value=0.0)
international_calls = st.number_input("International Calls", min_value=0)
international_charge = st.number_input("International Charge", min_value=0.0)
total_charge = st.number_input("Total Charge", min_value=0.0)


# In[ ]:


if st.button(" Predict "):
    # Creating input array

    missing_feature_1 = 0  # Replace with actual missing feature
    missing_feature_2 = 0  # Replace with actual missing feature
    
    input_features = np.array([account_length, voice_mail_plan, voice_mail_messages, day_mins, evening_mins, night_mins,
                               international_mins, customer_service_calls, international_plan, day_calls, day_charge, evening_calls,
                               evening_charge, night_calls, night_charge, international_calls, international_charge, total_charge,
                              missing_feature_1,missing_feature_2])
    
   # Reshape the input features to be a 2D array (1 sample with 19 features)
    input_features = input_features.reshape(1, -1)
    
    # Making a Prediction
    predicted_churn = model.predict(input_features)[0]

    # Displaying the result
    st.success(f"The Predicted Churn is : {predicted_churn:.2f}")

