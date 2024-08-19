import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()
script_model = torch.jit.script(model)
from torch.utils.mobile_optimizer import optimize_for_mobile
script_model_vulkan = optimize_for_mobile(script_model, backend='Vulkan')
torch.jit.save(script_model_vulkan, "frcnn.pth")