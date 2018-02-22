
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import Counter


# In[2]:


products=pd.read_csv('C:/Users/Gautam/Desktop/Big_bazaalarge/a.csv',parse_dates=True)
tender=pd.read_csv('C:/Users/Gautam/Desktop/Big_bazaalarge/tendermodes.csv')


# In[89]:


products.head(100)


# In[8]:


pred_new=pd.DataFrame(columns=['customerID','products'],dtype=str)

abcd=products
m=0
    
k=0    
for i in abcd['customerID'].unique():
    pred_list=[]
    for j in abcd['customerID']:
        if(j==i):
            pred_list.append(abcd['product_code'][k])
            
            k=k+1
    pred_2=[int(i) for i in pred_list]
    pred_2=[str(i) for i in pred_2]
    
    
    c = Counter(pred_2)
    l=c.most_common()
    sorted_d=[]
    sl = [(v,n) for (v, n) in l]
    
    
    points = [(n) for (v, n) in l]
    
    
    for u in range(len(sl)):
        sorted_d.append((str)(sl[u][0]))
    
    if(len(sorted_d)<3):
        for v in range(3-len(sorted_d)):
            sorted_d.append("None")
    
    if (len(sorted_d)>=3):
        sorted_d=sorted_d[0:3]
    for o in range(17):
        sorted_d.append("None")
    
    pred_temp=sorted_d[0]+","
    for l in range(1,len(sorted_d)):
        pred_temp=pred_temp+ "%s"%(sorted_d[l]) 
        if(l!=(len(sorted_d)-1)):
            pred_temp=pred_temp+ ","
            
      
    pred_temp_df = pd.DataFrame.from_records([{'customerID':i, 'products':pred_temp}])  
    pred_new=pred_new.append(pred_temp_df)
    m=m+1
    #if(m%250==0):
    print(i)


# In[64]:


pred.to_csv('C:/Users/Gautam/Desktop/Big_bazaalarge/pred.csv',index=False)


# In[7]:


products['product_code'].fillna("0000", inplace=True)


# In[208]:



products['transactionDate'] = pd.to_datetime(products['transactionDate'],format='%Y-%m-%d')
products['trx_year']=(products['transactionDate'].dt.year).astype(int)

products['trx_month']=(products['transactionDate'].dt.month).astype(int)


# In[22]:


len(pred_new)


# In[13]:


sample=pd.read_csv('C:/Users/Gautam/Desktop/Big_bazaalarge/sampleSubmission.csv')


# In[14]:


pred_custom=pred_new['customerID'].values
    


# In[75]:


Large_products=pd.read_csv('C:/Users/Gautam/Desktop/Big_bazaalarge/products.csv',parse_dates=True)


# In[76]:


Large_products[Large_products['customerID']=='BBID_2041218']


# In[15]:


sample_custom=sample['customerID'].values


# In[149]:


sample_custom


# In[16]:


diff=np.setdiff1d(sample_custom,pred_custom,assume_unique=True)


# In[17]:


len(diff)


# In[85]:


len(pred_custom)


# In[19]:


noneArray=["None","None","None","None","None","None","None","None","None","None","None","None","None","None","None","None","None","None","None","None",]
noneArray


# In[18]:


for i in diff:
    diff_temp_df = pd.DataFrame.from_records([{'customerID':i, 'products':noneArray}])  
    pred_new=pred_new.append(diff_temp_df)


# In[ ]:


pred_new=pd.DataFrame(columns=['customerID','products'],dtype=str)

abcd=products
k=0
m=0
abcd['points']=0
for k in range(len(abcd)):
    if(abcd['trx_month'][k] in range(5,7)):
        abcd['points'][k]+=1
    
    
k=0    
for i in abcd['customerID'].unique():
    pred_list=[]
    for j in abcd['customerID']:
        if(j==i):
            pred_list.append(abcd['product_code'][k])
            
            k=k+1
    pred_2=[int(i) for i in pred_list]
    pred_2=[str(i) for i in pred_2]
    
    
    c = Counter(pred_2)
    l=c.most_common()
    sorted_d=[]
    sl = [(v,n) for (v, n) in l]
    
    
    points = [(n) for (v, n) in l]
    
    
    for u in range(len(sl)):
        sorted_d.append((str)(sl[u][0]))
    
    if(len(sorted_d)<3):
        for v in range(3-len(sorted_d)):
            sorted_d.append("None")
    
    if (len(sorted_d)>=3):
        sorted_d=sorted_d[0:3]
    for o in range(17):
        sorted_d.append("None")
    
    pred_temp=sorted_d[0]+","
    for l in range(1,len(sorted_d)):
        pred_temp=pred_temp+ "%s"%(sorted_d[l]) 
        if(l!=(len(sorted_d)-1)):
            pred_temp=pred_temp+ ","
            
    print(pred_temp)    
    pred_temp_df = pd.DataFrame.from_records([{'customerID':i, 'products':pred_temp}])  
    pred_new=pred_new.append(pred_temp_df)
    m=m+1
    #if(m%250==0):
    print(i)


# In[223]:


abcd


# In[24]:


pred_new.to_csv('C:/Users/Gautam/Desktop/Big_bazaalarge/pred1.csv',index=False)


# In[193]:


diff_1=np.setdiff1d(sample_custom,p['customerID'].values,assume_unique=True)
len(diff_1)


# In[21]:


pred_temp_none="None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None"

for i in diff:
    diff_temp_df = pd.DataFrame.from_records([{'customerID':i, 'products':pred_temp_none}])  
    pred_new=pred_new.append(diff_temp_df)


# In[204]:


p=pred[-28622:]
len(prednew)


# In[195]:


p.to_csv('C:/Users/Gautam/Desktop/Big_bazaalarge/p.csv',index=False)


# In[198]:


q=pd.DataFrame(columns=['customerID','products'],dtype=str)
for i in sample['customerID']:
    diff_temp_df = pd.DataFrame.from_records([{'customerID':i, 'products':pred_temp_none}])  
    q=q.append(diff_temp_df)


# In[199]:


q.to_csv('C:/Users/Gautam/Desktop/Big_bazaalarge/q.csv',index=False)

