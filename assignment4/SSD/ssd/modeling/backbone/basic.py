import torch


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
    #Define the structure
    #1.Classical VGG
        base=[]
        base.append(torch.nn.Conv2d(in_channels=image_channels, out_channels=64,kernel_size=3,stride=1,padding=1))
        base.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        base.append(torch.nn.ReLU())
        base.append(torch.nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3,stride=1,padding=1))
        base.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        base.append(torch.nn.ReLU())
        base.append(torch.nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3,stride=1,padding=1)) 
        base.append(torch.nn.ReLU())
        base.append(torch.nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3,stride=2,padding=1)) #output[0]
        base.append(torch.nn.ReLU())
        self.base = torch.nn.Sequential(*base)
    #2. extra_layers (ReLU will be used in the foward function)
        extra_layer=[]
        extra_layer.append(torch.nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3,stride=1,padding=1))
        #need ReLU
        extra_layer.append(torch.nn.Conv2d(in_channels=256, out_channels=512,kernel_size=3,stride=2,padding=1)) #ouput[1]
        #need ReLU
        extra_layer.append(torch.nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3,stride=1,padding=1))
        #need ReLU
        extra_layer.append(torch.nn.Conv2d(in_channels=512, out_channels=256,kernel_size=3,stride=2,padding=1)) #output[2]
        #need ReLU
        extra_layer.append(torch.nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3,stride=1,padding=1))
        #need ReLU
        extra_layer.append(torch.nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3,stride=2,padding=1)) #output[3]
        #need ReLU
        extra_layer.append(torch.nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3,stride=1,padding=1))
        #need ReLU
        extra_layer.append(torch.nn.Conv2d(in_channels=256, out_channels=128,kernel_size=3,stride=2,padding=1))  #output[4]
        #need ReLU
        extra_layer.append(torch.nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3,stride=1,padding=1))
        #need ReLU
        extra_layer.append(torch.nn.Conv2d(in_channels=256, out_channels=128,kernel_size=3,stride=2,padding=0))  #output[5]
        #need ReLU
        self.extra_layer=torch.nn.Sequential(*extra_layer)
        
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        #The output from the base,i.e. output[0]
        x=self.base(x)
        out_features.append(x)
        #For other outputs:
        for i, f in enumerate(self.extra_layer): #i means the index and f means the function of the layer
            if i== len(self.extra_layer):
                x= f(x)
            else:
                x= torch.nn.functional.relu(f(x),inplace=True)
            if i % 2 == 1:
                out_features.append(x)
            
        for idx, feature in enumerate(out_features):
            out_channel = self.output_channels[idx]
            feature_map_size=self.output_feature_size[idx]
            #print('out_channel',feature.shape[1:])
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

