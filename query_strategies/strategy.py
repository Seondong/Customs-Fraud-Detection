import numpy as np

class Strategy:
    def __init__(self, args):  # model_path, test_data, test_loader
        self.args = args
        self.rev_func = args.rev_func
        self.name = args.sampling
    
    
    def set_name(self, name):
        self.name = name
        
        
    def set_uncertainty_module(self, uncertainty_module):
        self.uncertainty_module = uncertainty_module
        

    def set_available_indices(self, unavailable):
        self.available_indices = np.delete(np.arange(self.num_data), unavailable)

    
    def set_data(self, data):
        self.data = data
        self.num_data = len(self.data.test)
        self.available_indices = np.arange(self.num_data)
    
    
    def query(self, k):
        pass