import torch
from thop import profile, clever_format

from mobilemlp import MobileMLP_Small, MobileMLP_Large


if __name__=="__main__":
    input = torch.randn(1, 3, 224, 224)
    model = MobileMLP_Small()
    model = MobileMLP_Large()
    model.eval()

    print(model)

    macs, params = profile(model, inputs=(input, ), custom_ops={})
    macs, params = clever_format([macs, params], "%.3f")

    print('Flops:  ', macs)
    print('Params: ', params)
