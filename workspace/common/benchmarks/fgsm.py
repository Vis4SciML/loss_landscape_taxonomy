import torch

class FSGM:
    def __init__(self, model, dataset, epsilons=[0, .05, .1, .15, .2, .25, .3]):
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.epsilons = espsilons
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = 'cuda'
        
    @staticmethod
    def fsgm_attack(input, epsilon, data_grad, min=0, max=1):
        sign = data_grad.sign()
        noisy_input = input + epsilon * sign 
        
        noisy_input = torch.clamp(noisy_input, min, max)
        
        return noisy_input
    
    def test():
        