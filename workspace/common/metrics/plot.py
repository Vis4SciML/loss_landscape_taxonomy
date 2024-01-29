from __future__ import print_function

import warnings
from metric import Metric

import loss_landscapes
import loss_landscapes.metrics

# ---------------------------------------------------------------------------- #
#                               Neural efficiency                              #
# ---------------------------------------------------------------------------- #

class Plot(Metric):
    def __init__(self, 
                 model=None, 
                 data_loader=None, 
                 name="plot"):
        super().__init__(model, data_loader, name)
        
    def compute(self, steps=150, distance=100, normalization='filter'):
        '''
        Compute the points to plot the approximantion of the loss landscape
        '''
        print("Plotting the loss landscape...")
        self.name += f'_{normalization}_{steps}_{distance}'
        
        criterion = self.model.loss
        x, y = iter(self.data_loader).__next__()
        metric = loss_landscapes.metrics.Loss(criterion, x, y)

        print(f'distance:\t{distance}\nsteps:\t{steps}\n')
        loss_data = loss_landscapes.random_plane(self.model,
                                                 metric,
                                                 distance,
                                                 steps,
                                                 normalization=normalization,
                                                 deepcopy_model=False)
        
        self.results = {
            'points': loss_data
        }
        
        return self.results