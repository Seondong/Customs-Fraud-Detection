import pandas as pd
import preprocess_data


class Import_declarations():
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path, encoding = "ISO-8859-1")
        self.profile_candidates = None
        self.firstcheck()
    
    def firstcheck(self):
        self.df = self.df.dropna(subset=["illicit"])
        self.df = self.df.sort_values("sgd.date")
        self.df = self.df.reset_index(drop=True)
        

class Syntheticdata(Import_declarations):
    def __init__(self, path):
        super(Syntheticdata, self).__init__(path)
        self.profile_candidates = ['importer.id', 'declarant.id', 'tariff.code', 'quantity', 'HS6', 'HS4', 'HS2', 'office.id']
        
        
        
class Ndata(Import_declarations):
    def __init__(self, path):
        super(Ndata, self).__init__(path)
        self.profile_candidates = ['importer.id', 'declarant.id', 'tariff.code', 'quantity', 'HS6', 'HS4', 'HS2', 'office.id']

        
class Mdata(Import_declarations):
    def __init__(self, path):
        super(Mdata, self).__init__(path)
        self.profile_candidates = ['importer.id', 'exporter.name', 'expcty', 'country', 'declarant.id', 'tariff.code', 'quantity', 'HS6', 'HS4', 'HS2', 'office.id']
        
        
class Tdata(Import_declarations):
    def __init__(self, path):
        super(Tdata, self).__init__(path)
        self.profile_candidates = ['importer.id', 'country', 'last.departure.code', 'contract.party.code',
                      'tariff.code', 'quantity', 'HS6', 'HS4', 'HS2', 'office.id']


class Cdata(Import_declarations):
    def __init__(self, path):
        super(Cdata, self).__init__(path)


class Kdata(Import_declarations):
    def __init__(self, path):
        super(Kdata, self).__init__(path)

        




        
    

