from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# reading the datasets

products=pd.read_csv('C:/Users/Gautam/Desktop/BigBazaa_Small/cproducts.csv',parse_dates=True)
tender=pd.read_csv('C:/Users/Gautam/Desktop/BigBazaa_Small/ctender.csv')





products['promotion_description'].fillna('no_promo', inplace=True)
products['Gender'].fillna('no_gender', inplace=True)
products['State'].fillna('no_state', inplace=True)
products['PinCode'].fillna(-1, inplace=True)
products['DOB'].fillna("2111-11-11", inplace=True)

products.loc[products['DOB']=='NANA',['DOB']] = '2111-11-11'

products1=products
tender['Gender'].fillna('no_gender', inplace=True)
tender['State'].fillna('no_state', inplace=True)
tender['PinCode'].fillna(-1, inplace=True)
tender['DOB'].fillna("1", inplace=True)
tender['PaymentUsed'].fillna("no_pay", inplace=True)

#making new columns for transaction year and month


products['transactionDate'] = pd.to_datetime(products['transactionDate'],format='%Y-%m-%d')
products['trx_year']=(products['transactionDate'].dt.year).astype(int)

products['trx_month']=(products['transactionDate'].dt.month).astype(int)




for c in products1.columns:
    lbl = LabelEncoder()
    if products1[c].dtype == 'object' and c not in ['store_description','customerID','transactionDate']:
        products1[c] = lbl.fit_transform(products1[c])
tender1=tender
for c in tender1.columns:
    lbl = LabelEncoder()
    if tender1[c].dtype == 'object' and c not in ['store_description','customerID','transactionDate']:
        tender1[c] = lbl.fit_transform(tender1[c])


#new feature based on type of method of payment

cash=[6,11]                  #1
wallets=[34,40,27,38,15,26,28,5,43,36,29,30,37,3,42,35,39,13,10,25,16,24,44,14]               #2
cards=[41,18,21,23,20,22,19,33,0,4,17,32,12,31,2,1]                 #3
             
empty=[7,9,8]                 #0
custom_id=products1['customerID'].values
sLength = len(products1['customerID'])
products1['custom_type_mode'] = 5


for i in custom_id:
    
    use_cols=tender1[tender1['customerID']==i]
    if (use_cols.empty==False):
        if(use_cols['tender_type'].value_counts().idxmax() in cash):
            products1.loc[products1['customerID'] == i, ['custom_type_mode']] = 1
        elif(use_cols['tender_type'].value_counts().idxmax() in wallets):
            products1.loc[products1['customerID'] == i, ['custom_type_mode']] = 2
        elif(use_cols['tender_type'].value_counts().idxmax() in cards):
            products1.loc[products1['customerID'] == i, ['custom_type_mode']] = 3
       
        elif(use_cols['tender_type'].value_counts().idxmax() in empty):
            products1.loc[products1['customerID'] == i, ['custom_type_mode']] = 0
            

#making new feature based on price of products

products1['custom_price_type']=0

products1.loc[products1['sale_price_after_promo']<250, ['custom_price_type']] = 1

products1.loc[products1['sale_price_after_promo'] >250, ['custom_price_type']] = 2

products1.loc[products1['sale_price_after_promo']>550, ['custom_price_type']] = 3

products1.loc[products1['sale_price_after_promo']>1250, ['custom_price_type']] = 4

#feature based on promotional offer on a product

products1.loc[products1['promo_code']==2213, 'promo_type']=1
products1['promo_type'].fillna(2, inplace=True)



## scaling, creating matrix and running k-means


stores = list(set(products1['store_code']))

cluster_labels = []
cluster_store = []
cluster_data = []
cluster_customers = []
cluster_score = []

for x in stores:
    cld = products1[products1['store_code'] == x]
    cluster_customers.append(cld['customerID'])
    cld.drop(['customerID','DOB','Gender','State','PinCode','transactionDate','store_code','store_description','till_no','transaction_number_by_till','promotion_description','product_description','sale_price_after_promo','promo_code'], axis=1, inplace=True)
    
    rbs = RobustScaler()
    cld2 = rbs.fit_transform(cld)
    
    km1 = KMeans(n_clusters=4)
    km2 = km1.fit(cld2)
    label = km2.predict(cld2)
    
    s_score = silhouette_score(cld2, label)
    print(x,s_score)
    cluster_score.append(s_score)
    
    cluster_labels.append(label)
    cluster_store.append(np.repeat(x, cld.shape[0]))
    cluster_data.append(cld2)




cluster_data = np.concatenate(cluster_data)

## convert nested lists as 1d array
cluster_customers = np.concatenate(cluster_customers)
cluster_store = np.concatenate(cluster_store)
cluster_labels = np.concatenate(cluster_labels)
print(cluster_labels)




subm = pd.DataFrame({'customerID':cluster_customers, 'store_code':cluster_store, 'cluster':cluster_labels})



#Saving the result to files

np.savetxt('C:/Users/Gautam/Desktop/BigBazaa_Small/subtxttest.txt', cluster_data)
subm.to_csv('C:/Users/Gautam/Desktop/BigBazaa_Small/submtest.csv', index=False)








