import numpy as np
import pandas as pd 
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from pickle import load
from datetime import datetime

def days_between(dd1,dd2):
    ddres = []
    for d1,d2 in zip(dd1, dd2):
        d1 = datetime.strptime(d1, '%Y-%m-%d')
        d2 = datetime.strptime(d2, '%Y-%m-%d')
        ddres.append(abs((d2 - d1).days))
    return ddres

########### Loading keras model and weights
json_file = open('StartupStudy.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
print('Loaded model from disk')
loaded_model.load_weights('StartupStudy.h5')
########### loading data preparation scalers
lb_style = load(open('lb_style.pkl','rb'))
scaler = load(open('scaler.pkl','rb'))
pca = load(open('pca.pkl','rb'))
############################### Sample record
#reco = {'main_category':['Publishing'], 'deadline':['2015-10-09'], 'goal': [1000.0], 'launched': ['2015-08-11'], 'pledged': [0.0], 'backers': [0], 'usd pledged': [0.0], 'usd_pledged_real': [0.0], 'usd_goal_real': [1533.95] }
#df= pd.DataFrame.from_dict(reco)
#df.to_csv('Thestartup.csv',index=False)
df=pd.read_csv('Thestartup.csv')
print(df)
################################ Prepare data
df['timeframe'] = days_between(df['deadline'], df['launched'])
df.drop(['deadline', 'launched'],axis=1, inplace=True)
dfcat = lb_style.transform(df['main_category'])
df=df.drop('main_category',axis=1).join(pd.DataFrame(dfcat,columns=lb_style.classes_,index=df.index))
dfscale=scaler.transform(df)
names=df.columns
df=pd.DataFrame(dfscale,columns=names)
dfpca=pca.transform(df)
df=pd.DataFrame(data=dfpca)
print('the Startup is expected to be : ', 'successful' if np.around(loaded_model.predict(dfpca)) else 'failed')
