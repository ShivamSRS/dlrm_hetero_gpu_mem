import torch
import torchvision.models as models
  
# Making the code device-agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
# Instantiating a pre-trained model
model = models.resnet18(pretrained=True)
  
# Transferring the model to a CUDA enabled GPU
model = model.to(device)
  
# Now the reader can continue the rest of the workflow
# including training, cross validation, etc!