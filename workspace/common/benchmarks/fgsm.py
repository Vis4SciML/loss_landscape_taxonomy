import torch


# test
module_path = os.path.abspath(os.path.join('../../../workspace/models/rn08/code/')) # or the path to your source code
sys.path.insert(0, module_path)
import rn08


class FSGM:
    def __init__(self, model, dataset, normalize, denormalize):
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.normalize = normalize
        self.denormalize = denormalize
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
    
    def test(self, epsilon):
        correct = 0
        adv_example = []
        
        for input, target in self.dataset:
            input, target = input.to(self.device), target.to(self.device)
            
            input.require_grad = True
            
            output = self.model(input)
            init_pred =output.max(1, keepdim=True)[1]
            
            # if the model already miss predict continue
            if init_pred.item() != target.item():
                continue
            
            loss = self.model.loss(output, target)
            
            self.model.zero_grad()
            loss.backward()

            input_grad = input.grad.data
            
            input_denorm = self.denormalize(input)
            
            perturbed_input = FSGM.fsgm_attack(input_denorm, epsilon, input_grad)
            
            perturbed_input_norm = self.normalize(perturbed_input)
            
            output = self.model(perturbed_input_norm)
            
            final_pred = output.max(1, keepdim=True)[1]
            
            if final_pred.item() == target.item():
                correct += 1
                
            adv_ex = perturbed_input.squeeze().detach().cpu().numpy()
            adv_example.append(adv_ex)

        final_acc = correct/float(len(self.dataset))
        print(f"Epsilon: {epsilon}\tTest Accuracy = {final_acc}")
        
        return final_acc, adv_example