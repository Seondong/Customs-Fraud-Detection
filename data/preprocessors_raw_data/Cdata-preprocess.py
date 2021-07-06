import pandas as pd
import numpy as np

df_raw = pd.read_csv('./CAM_2016-2019_Annoymized.csv')
df = df_raw.copy()

# List data in ascending order of REGDATE. 
df = df_raw.sort_values('SGD.DATE', ascending=True).reset_index(drop=True)

# Delete unnecessary columns
del df['File']

# Manage NaN values
df['IMPORTER.ID'] = np.where(df['IMPORTER.ID'].isna(), 'unknown_importer', df['IMPORTER.ID'])
df['COUNTRY'] = np.where(df['COUNTRY'].isna(), 'unknown_country', df['COUNTRY'])


# Clearing up edge cases
'''
In [1]: df['ILLICIT'].value_counts()

0    1875482
1      27148
Name: ILLICIT, dtype: int64
'''

# Case 1:
# There are quite a lot of edge cases where revenue is nan (25%). 
# Among them, very small portion (0.2%) of them are marked as illicit.
'''
In [2]: df[df['REVENUE'].isna()]['ILLICIT'].value_counts()

0    420755
1      1090
Name: ILLICIT, dtype: int64
'''

# Handle these edge cases where revenue is nan, but illicit == 0
# Force revenue value to be 0 since they are not illicit.
df['REVENUE'] = np.where((df['REVENUE'].isna()) & (df['ILLICIT'] == 0), 0, df['REVENUE'])


# For these cases with ILLICIT == 0 and REVENUE is nan, we cannot fill na in my own way.
# These 1090 rows are illicit, thus having these items would probably be helpful to classify illicit items. 
# However, they don't provide enough evidence to train our classifiers. So we remove them.
df = df.dropna(subset=['REVENUE'])    
    

# Case 2:
# There are some edge cases where revenue < 0. Among them, 75% are labeled as ILLICIT.
df[df['REVENUE']<0]['ILLICIT'].value_counts()
'''
1    3354
0    1201
Name: ILLICIT, dtype: int64
'''

# The distributions seem okay...
'''
import math
df[(df['ILLICIT'] == 0) & (df['REVENUE'] < 0)]['REVENUE'].apply(lambda x: math.log(-x, 10)).hist()
df[(df['ILLICIT'] == 1) & (df['REVENUE'] < 0)]['REVENUE'].apply(lambda x: math.log(-x, 10)).hist()
'''

### But, we do not want to count troublesome situations, unless we are given any overinvoicing problems. So we decided to drop these inputs.
df = df.drop(df[df['REVENUE']<0].index)

# Case 3:
# There are some edge cases where revenue == 0, but illicit == 1.
'''
df[df['REVENUE']==0]['ILLICIT'].value_counts()
0    1873894
1       3048
Name: ILLICIT, dtype: int64
'''

# Handle these edge cases where revenue == 0, but illicit == 1
# Since those are marked as illicit, I decided to force illicit value to be a very smal1 number.
df['REVENUE'] = np.where((df['REVENUE']==0) & (df['ILLICIT'] == 1), 0.1, df['REVENUE'])

# Case 4:
# There are some edge cases where revenue > 0, but illicit == 0
'''
df[df['REVENUE']>0]['ILLICIT'].value_counts()
1    22704
0      387
Name: ILLICIT, dtype: int64
'''
# Handle these edge cases where revenue > 0, but illicit == 0, we can force illicit value to be 1
df['ILLICIT'] = np.where((df['REVENUE']>0) & (df['ILLICIT'] == 0), 1, df['ILLICIT'])



# We can omit this code block. 
# This code block is to remove top 0.1% transactions in 'TOPUPTAX'.
# 2016 real import data has some outliers which have extremely high values in 'TOPUPTAX'.
# We remove them as the perfomance of XGBoost model in revenue collection may be significantly affected by
# (distorted by) whether such outliers are detected or not.  
upper_bound=df[df['ILLICIT']==1]['REVENUE'].quantile(0.995)
df = df.copy()[df['REVENUE']<upper_bound]

# Replace some columns' names with those used in the DATE model
df.rename(columns={'SGD.ID':'sgd.id',
                   'SGD.DATE':'sgd.date',
                   'IMPORTER.ID':'importer.id',
                   'DECLARANT.ID':'declarant.id',
                   'COUNTRY':'country',
                   'OFFICE.ID':'office.id',
                   'TARIFF.CODE':'tariff.code',
                   'QUANTITY':'quantity',
                   'GROSS.WEIGHT':'gross.weight',
                   'FOB.VALUE':'fob.value',
                   'CIF.VALUE':'cif.value',
                   'TOTAL.TAXES':'total.taxes',
                   'ILLICIT':'illicit',
                   'REVENUE':'revenue',
                  }, inplace=True)

# Define columns to use
columns_to_use = df.columns

df = df[columns_to_use]

# Set data format
df['sgd.date'] = df['sgd.date'].astype(str)
df['tariff.code'] = df['tariff.code'].astype(int)

df=df.sort_values('sgd.date', ascending=True).reset_index(drop=True)
df.to_csv('../../cdata.csv', encoding = "ISO-8859-1", index=False)

