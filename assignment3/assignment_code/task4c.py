
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
from torch import nn
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)

new_model = nn.Sequential(*list(model.children())[:-2]) #Extrct the model's sturcture until the last Conv. layer
#print(new_model) You can see the sturcture if you want.


# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation_last=new_model(image)
print("Activation shape:", activation_last.shape)

for i in range(10):
    image2=torch_image_to_numpy(activation_last[0,i,:,:])
    plt.imshow(image2)
    plt.savefig('Activation_Last_'+str(i)+'.png')


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image

