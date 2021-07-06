import pandas as pd
import numpy as np

df_1 = pd.read_csv('./data_01.txt',sep="|", header=0, encoding='latin-1')
df_2 = pd.read_csv('./data_02.txt',sep="|", header=0, encoding='latin-1')
df_3 = pd.read_csv('./data_03.txt',sep="|", header=0, encoding='latin-1')
df_4 = pd.read_csv('./data_04.txt',sep="|", header=0, encoding='latin-1')
df_5 = pd.read_csv('./data_05.txt',sep="|", header=0, encoding='latin-1')

df_raw = df_1.append([df_2,df_3,df_4,df_5])

df = df_raw.copy()

# Define 'illicit' (value 1 if any of additional taxes and fines have values)
df['illicit'] = np.where(df.taxe627.isnull() & 
                         df.taxe630.isnull() & 
                         df.taxe635.isnull() & 
                         df.Mt_Redressement.isnull(),
                         0, 1)

# Reference) If Mt_redressement> 0 or tax 635> 0 then we can define them in the majority of cases as under evaluation Taxes627 and taxes630 are fines that can be linked to a false declaration of value or other offenses

# Currency expression harmonization: Convert "xxx,xx" to "xxx.xx"
# Define numeric variables
num_var = ['n°article', 'activité', 'valeur',
           'Cours_TND_USD', 'Valeur en USD', 'montant', 'poids net', 'qcs',
           'taxe627', 'taxe630',
           'taxe635', 'Mt_Redressement']
# Conversion
for var in num_var:
    df[var] = df[var].astype(str)
    df[var] = df[var].str.replace(',','.').astype(float)
    df[var] = df[var].fillna(0)
    
df['revenue'] = df.taxe627 + df.taxe630 + df.taxe635 + df.Mt_Redressement

column_to_use = ['date_validation','Bureau','code opérateur',
                 'ndp','org','provenance','achat', 'valeur',
                 'montant', 'poids net', 'qcs',
                 'revenue','illicit']

df = df[column_to_use]
df = df.rename(columns={'date_validation':'sgd.date', 
                   'Bureau':'office.id',
                   'code opérateur':'importer.id',
                   'ndp':'tariff.code',
                   'org':'country',
                   'provenance':'last.departure.code',
                   'achat':'contract.party.code',
                   'valeur':'cif.value',
                   'montant':'total.taxes',
                   'poids net':'gross.weight',
                   'qcs':'quantity',
                   'revenue':'revenue',
                   'illicit':'illicit'})

# Set data format
df['sgd.date'] = df['sgd.date'].astype(str)
df = df[df['tariff.code'] != 's']
df['tariff.code'] = df['tariff.code'].astype(int)

df=df.sort_values('sgd.date')
df['sgd.date'] = pd.to_datetime(df['sgd.date'],format='%Y%m%d').dt.strftime('%y-%m-%d')
df.to_csv('../../tdata.csv', encoding = "ISO-8859-1", index=False)

