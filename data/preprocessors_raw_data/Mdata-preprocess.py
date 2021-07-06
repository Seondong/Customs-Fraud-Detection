import pandas as pd
import numpy as np

df_raw = pd.read_csv('./M.data.cleaned.anonymized.csv')

# List data in ascending order of REGDATE. 
df = df_raw.sort_values('REGDATE', ascending=True).reset_index(drop=True)

# Define illicit imports
# The data owner provided a separate table composed of the following three features only for illicit imports.
# The following three features have positive numeric values only when the imports are fraudulent.
# Replace NAs with 0s.
df.ASSESSEDTAX.fillna(0, inplace=True)
df.ORIGTAXPAID.fillna(0, inplace=True)
df.TOPUPTAX.fillna(0, inplace=True)

# Define 'illicit' (value 1 if any of additional taxes and fines have values)
df['illicit'] = np.where(df['TOPUPTAX']>0, 1, 0)

# Clean data
# The last 20% of data had no illicit imports. 
# We assume that the last 20% have not been properly labelled. 
# Remove the last 20% of data
df = df.iloc[:int(df.shape[0]*0.8),:]


# Generate 'year' to check the number of imports per year
df['year']=pd.to_datetime(df['REGDATE'], format= '%Y-%m-%d').dt.year

print(df['year'].value_counts())

# Numbers of imports from 2007 to 2012 is abnormally small.
# We assume that those are only subsamples not randomly selected. 
# Remove data from 2007 to 2012
df = df[df['year']>2012]

# Define total.taxes
df['total.taxes'] = df['DUTY']+df['EXCISE']+df['VAT']

# The data owner did not informed whether total.taxes are initial values or final values (adjusted values after Customs intervention).
# Check the number of cases where imports are illicit and ;
print('illicit case 1. total.taxes > ORIGTAXPAID: ',((df['illicit'] == 1)&(df['total.taxes']>df['ORIGTAXPAID'])).sum())
print('illicit case 2. total.taxes = ORIGTAXPAID: ', ((df['illicit'] == 1)&(df['total.taxes']==df['ORIGTAXPAID'])).sum())
print('illicit case 3. total.taxes < ORIGTAXPAID: ', ((df['illicit'] == 1)&(df['total.taxes']<df['ORIGTAXPAID'])).sum())

# Considering the above analysis, 
# total.taxes (the sum of DUTY, EXCISE and VAT) is more likely to be the final value of total taxes after Customs intervention rather than initial values. 
# Replace total.taxes in frauds with respective ORIGINTAXPAID, as we will define total.taxes as the initial values.
df['total.taxes'] = np.where(df['illicit'] == 1, df['ORIGTAXPAID'], df['total.taxes'])

# We can omit this code block. 
# This code block is to remove top 1% transactions in 'TOPUPTAX'.
# 2016 real import data has some outliers which have extremely high values in 'TOPUPTAX'.
# We remove them as the perfomance of XGBoost model in revenue collection may be significantly affected by
# (distorted by) whether such outliers are detected or not.  
upper_bound=df[df['illicit']==1]['TOPUPTAX'].quantile(0.995)
df = df.copy()[df['TOPUPTAX']<upper_bound]

# Replace some columns' names with those used in the DATE model
df.rename(columns={'TOPUPTAX':'revenue',
                   'ITEMNO':'item.number',
                   'REGNO':'sgd.number',
                   'ORIGIN':'country',
                   'HSCODE':'tariff.code',
                   'TPIN':'importer.id',
                   'VDP.AMOUNTS':'cif.value',
                   'NETWEIGHT':'gross.weight',
                   'QTY':'quantity',
                   'REGDATE':'sgd.date',
                   'AGENTCODE':'declarant.id',
                   'EXCRATE':'exchange.rate',
                   'OFFICE':'office.id',
                   'EXPORTER.NAME':'exporter.name',
                   'EXPCTY':'expcty',
                  }, inplace=True)


# Define columns to use
columns_to_use = [
    'sgd.date', 'sgd.number', 'office.id', 'importer.id', #'IMPORTER.NAME',
    'declarant.id', #'DECLARANT.NAME', 
    'exporter.name',
    'expcty',
    'item.number', 'tariff.code',
    'country', 'cif.value', 'exchange.rate', #'CIF_USD_EQUIVALENT',
    'quantity', 'gross.weight', 'total.taxes', #'total.taxes.USD',
    'revenue', 
    #'RAISED_TAX_AMOUNT_USD', 
    'illicit', #'Source'
    ]

df = df[columns_to_use]

# Remove NAs
df = df.dropna(subset=['quantity', 'exchange.rate'])
df['sgd.date'] = pd.to_datetime(df['sgd.date'],format='%Y-%m-%d').dt.strftime('%y-%m-%d')

# Set data format
df['sgd.date'] = df['sgd.date'].astype(str)
df['tariff.code'] = df['tariff.code'].astype(int)

df=df.sort_values('sgd.date')
df.to_csv('../../mdata.csv', encoding = "ISO-8859-1", index=False)

