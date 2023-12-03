import pickle
import os

'''
Class used to allow the loss landscape metrics to inherit basic methods and template
'''
class Metric:
    
    def __init__(self, model=None, data_loader=None, name="metric") -> None:
        
        assert model != None
        
        self.model = model
        self.data_loader = data_loader
        self.name = name
        self.results = None
    
    def save_on_file(self, path="./"):
        print('Storing the result...')
        f = open(os.path.join(path, self.name + ".pkl"), "wb")
        pickle.dump({self.name: self.results}, f)
        f.close()
        print('Complete')
        
    def load_from_file(self, path="./"):
        print('Loading the result...')
        f = open(os.path.join(path, self.name + ".pkl"), "rb")
        data = pickle.load(f)
        f.close()
        self.result = data[self.name]  
        print('Loading complete')
        
        return self.result
