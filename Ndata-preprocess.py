import pandas as pd

df = pd.read_csv('./data/Ndata.merged.anonymized.single_tax_relabeled.csv', encoding = "ISO-8859-1") 
df = df.rename(columns={'SGD.DATE':'sgd.date', 'IMPORTER.TIN':'importer.id', 'DECLARANT.CODE':'declarant.id', 'ORIGIN.CODE':'country','OFFICE':'office.id','TARIFF.CODE':'tariff.code', 'QUANTITY':'quantity', 'GROSS.WEIGHT':'gross.weight', 'FOB.VALUE':'fob.value','CIF.VALUE':'cif.value','TOTAL.TAXES':'total.taxes','RAISED_TAX_AMOUNT':'revenue'})

df['sgd.date'] = pd.to_datetime(df['sgd.date'],format='%d-%b-%y').dt.strftime('%y-%m-%d')
df.to_csv('./data/ndata.csv', encoding = "ISO-8859-1", index=False)


