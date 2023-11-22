import torch
import re
from model import JetTagger


input_shapes = {
    "JT": (1,16),
    "RN07": (1,3,32,32),
}

def get_new_model(args):
    model_arch = args.arch.split('_')[0]
    if model_arch == 'RN07':
        return get_rn07(args)
    if args.weight_precision is 32:
        print("using FP32 model")
        model = get_jettagger(args)#.cuda()
    else:
        print("using quant model")
        model = get_quantized_jettagger(args)#.cuda()
    return model


class ArgFake:
    def __init__(self, w=32, b=32, a=32, bn=False, d=False, arch='JT'):
        self.weight_precision = w
        self.bias_precision = b
        self.act_precision = a
        self.batch_norm = bn
        self.dropout = d
        self.arch = arch

def update_fc_size(model, ckp, dense_layers=['dense_1', 'dense_2', 'dense_3', 'dense_4']):
    for dl in dense_layers:
        layer = getattr(model, dl)
        ckp[f"{dl}.fc_scaling_factor"] = torch.ones(layer.fc_scaling_factor.shape) * ckp[f"{dl}.fc_scaling_factor"]
    return ckp

def load_checkpoint(args, file_name): 
   # checkpoint = torch.load(file_name)
   model_arch = args.arch.split('_')[0]
   result = re.search(f'{model_arch}_(.*)b', args.arch)
   quant = int(result.group(1))
   fake = ArgFake(w=quant,b=quant,a=quant+3, arch=model_arch)
   model = get_new_model(fake)
   
   checkpoint = torch.load(file_name, map_location=torch.device("cpu"))
   model(torch.randn(input_shapes[model_arch]))
   model.load_state_dict(checkpoint)
   return model

#    if quant is 32:
#         model.load_state_dict(checkpoint)
#         #model.train()
#         return model

#     #set_bit_config(model, checkpoint["bit_config"], args)
#     model = get_new_model(fake)
#     #checkpoint = checkpoint["state_dict"]
#     model_dict = model.state_dict()
#     modified_dict = {}
    
#     updated_ckp = update_fc_size(model, checkpoint)   
    
#     for key, value in checkpoint.items():
#         if model_dict[key].shape != value.shape:
#             print(f"mismatch: {key}: {value}")
#             value = torch.tensor([value], dtype=torch.float64)
#         modified_dict[key] = value
    
#     model.load_state_dict(modified_dict, strict=False)
    
    
    # return model
