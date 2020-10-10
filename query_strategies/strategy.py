import numpy as np

class Strategy:
    def __init__(self, data, args):  # model_path, test_data, test_loader
        self.data = data
        self.args = args
        self.num_data = len(self.data.test)
        self.available_indices = np.arange(self.num_data)
        self.rev_func = args.rev_func 
        

    def set_available_indices(self, unavailable):
        self.available_indices = np.delete(np.arange(self.num_data), unavailable)

        
    def query(self, k, **kwargs):
        pass