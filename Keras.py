#!/usr/bin/env python
# coding: utf-8

# In[309]:


import pandas as pd
df = pd.read_csv('ks-projects-201801.csv')
dforig = pd.read_csv('ks-projects-201801.csv')


# In[310]:


dforig.head()


# ### Data Cleansing
# 
# Removing nulls, checking types, checking row uniqueness, checking data values, getting timeframe differrence, checking if some data is invalid(I dont' see a rule in this last point)

# In[311]:


df.dropna(inplace=True)
df.shape


# In[312]:


df.dtypes


# ###### the numbered data are really numbers, no need to convert any type

# In[313]:


df['state'].value_counts()


# ###### Above seems we will need to keep only the "failed" and "succesfful" states. But first let us get rid of some of unseful columns

# In[314]:


df.drop(['ID','category','currency'],axis=1,inplace=True)


# ###### generating the timeframe columns then get rid of the deadline and launched columns

# In[315]:


from datetime import datetime
def days_between(dd1, dd2):
    ddres=[]
    for d1,d2 in zip(dd1,dd2):
        d1 = datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
        ddres.append(abs((d2 - d1).days))
    return ddres

df['timeframe']=days_between(df['deadline'],df['launched'])


# In[316]:


df.drop(['deadline','launched'],axis=1, inplace=True)
df.drop(['country','name'],axis=1,inplace=True)
df.head()


# In[317]:


df=df[df.state.isin(['failed','successful'])]


# In[318]:


df.shape


# In[319]:


df


# chaning state column to 0 or 1

# In[320]:


df.loc[df['state']=='failed','result']=0
df.loc[df['state']=='successful','result']=1


# In[321]:


df.drop(['state'],axis=1,inplace=True)
df.drop_duplicates(inplace=True)


# In[322]:


import matplotlib.pyplot as plt
#fig, ax = plt.subplots(num=None, figsize=(10, 4), dpi=80, facecolor='w', edgecolor='k')
#ax.boxplot(df['backers'])
#ax.hist(df['goal'].T)
#plt.show()
df[['backers','goal','timeframe','pledged']].hist(bins=5)


# Above shows that timeframe has a normal distribution, while others are very centric

# In[323]:


features=df.drop('result',axis=1)
labels=df['result']


# In[324]:


from sklearn.model_selection import train_test_split
X_train, X_tv, y_train, y_tv = train_test_split(features, labels, test_size=0.33, random_state=42)
X_test, X_validate, y_test, y_validate = train_test_split(X_tv, y_tv, test_size=0.5, random_state=42)


# # Centring data 
#    

# In[325]:


# hot encoding and keepying the lb_style for prediction
from sklearn.preprocessing import LabelBinarizer
lb_style = LabelBinarizer()
lb_style.fit(X_train["main_category"],)
x_train=lb_style.transform(X_train['main_category'])
x_test=lb_style.transform(X_test['main_category'])
x_validate=lb_style.transform(X_validate['main_category'])
#dfd=df.join(pd.DataFrame(lb_results, columns=lb_style.classes_, index = df.index))
#df = dfd.drop('main_category',axis=1)


# In[326]:


X_train=X_train.drop('main_category',axis=1).join(pd.DataFrame(x_train,columns=lb_style.classes_,index=X_train.index))
X_test=X_test.drop('main_category',axis=1).join(pd.DataFrame(x_test,columns=lb_style.classes_,index=X_test.index))
X_validate=X_validate.drop('main_category',axis=1).join(pd.DataFrame(x_validate,columns=lb_style.classes_,index=X_validate.index))

X_train
X_test


# In[327]:


str(X_validate.shape)+str(X_test.shape)+str(X_train.shape)


# ### Standardization of all

# In[328]:


from sklearn import preprocessing
# Get column names first
names = X_train.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaler.fit(X_train)
train=scaler.transform(X_train)
test=scaler.transform(X_test)
validate=scaler.transform(X_validate)
validate=scaler.transform(X_validate)
X_train= pd.DataFrame(train, columns=names)
X_test= pd.DataFrame(test, columns=names)
X_validate= pd.DataFrame(validate, columns=names)
X_test.head()


# In[329]:


X_test.shape[0]+X_train.shape[0]+X_validate.shape[0]


# # finding if lot of columns are correlated

# In[330]:


X_train.corr()


# ..

# In[331]:


str(X_train.shape) + str(X_test.shape)


# In[332]:


from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(X_train)
train=pca.transform(X_train)
test=pca.transform(X_test)
validate=pca.transform(X_validate)
X_train= pd.DataFrame(data = train)
X_test = pd.DataFrame(data = test)
X_validate = pd.DataFrame(data = validate)


# In[333]:


X_test


# In[334]:


from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots(num=None, figsize=(14,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
ax.scatter(X_train[0],X_train[1],X_train[2],c=y_train)


# In[335]:


from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots(num=None, figsize=(14,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
ax.scatter(X_train[0],X_train[3],X_train[4],c=y_train)


# In[336]:


from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots(num=None, figsize=(14,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
ax.scatter(X_train[1],X_train[3],X_train[4],c=y_train)


# In[337]:


from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots(num=None, figsize=(14,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
ax.scatter(X_train[2],X_train[3],X_train[4],c=y_train)


# In[338]:


from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots(num=None, figsize=(14,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
ax.scatter(X_train[1],X_train[2],X_train[3],c=y_train)


# In[339]:


from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots(num=None, figsize=(14,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
ax.scatter(X_train[0],X_train[1],X_train[3],c=y_train)


# In[340]:


from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots(num=None, figsize=(14,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
ax.scatter(X_train[0],X_train[2],X_train[4],c=y_train)


# From above scatters we can see that we can train data as we see mostly separation between failed an succeeded proejcts

# # Training with Logistics regression

# In[341]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,y_train)
trainscore=clf.score(X_train,y_train)
testscore=clf.score(X_test,y_test)
print('training score: ',trainscore,'  , test score: ',testscore)


# # Training with Keras

# In[342]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import sys
from tensorflow.keras.layers import LSTM, Dense, Activation


# ## defining layers, epocs and dimension.. converting pandas sets to numpy

# In[343]:


number_of_neurons_layer1 = 150
number_of_neurons_layer2 = 120
number_of_neurons_layer3 = 40
number_of_neurons_layer4 = 40
number_of_neurons_layer5 = 1
number_of_epochs = 100
dim = X_train.shape[1]
X_train,y_train,X_test,y_test,X_validate,y_validate = X_train.to_numpy(),y_train.to_numpy(),X_test.to_numpy(),y_test.to_numpy(),X_validate.to_numpy(),y_validate.to_numpy()


# In[344]:


# design network
from tensorflow.keras import optimizers
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

model = Sequential()
model.add(Dense(number_of_neurons_layer1,input_shape=(dim, ), activation='relu'))
model.add(Dense(number_of_neurons_layer2, activation='relu'))
model.add(Dense(number_of_neurons_layer5, activation='sigmoid'))
model.compile(loss='mean_squared_error',
        #optimizer='rmsprop',
        optimizer = 'adam',
        metrics=['accuracy','mae'])


# In[345]:


model.summary()


# In[346]:


model.fit(X_train, y_train,
        batch_size=144,
        epochs=number_of_epochs,
        verbose=1,
        validation_data=(X_validate, y_validate))
        


# In[347]:


testscore = model.evaluate(X_test, y_test, verbose=0)
trainscore = model.evaluate(X_train, y_train, verbose=0)
print('training MSE,accuracy,MAE: ',trainscore,'  , test MSE,accuracy,MAE: ',testscore)


# ## Save model and weights

# In[348]:


model_json = model.to_json()
with open("StarupStudy.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("StartupStudy.h5")
print("Saved model to disk")


# In[ ]:




