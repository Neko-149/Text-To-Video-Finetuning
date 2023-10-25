import torch
device=torch.device("cpu")
model_weight_path="/home/yuxueqing/models/zeroscope/text2video_pytorch_model.pth"
zeroscope=torch.load(model_weight_path,map_location=device)
model_weight_path="/home/yuxueqing/models/modelscope/text2video_pytorch_model.pth"
modelscope=torch.load(model_weight_path,map_location=device)

def custom_equal(tensor1, tensor2):
    diff = torch.abs(tensor1 - tensor2)
    return torch.all(diff < 1e-3)
#device=torch.device("cuda:0")
#modelscope=modelscope.to(device)
#zeroscope=zeroscope.to(torch.device("cpu"))

for key, value in modelscope.items():
    if key in zeroscope and not custom_equal(value, zeroscope[key]):
        print("Key '{}' differs".format(key))
    