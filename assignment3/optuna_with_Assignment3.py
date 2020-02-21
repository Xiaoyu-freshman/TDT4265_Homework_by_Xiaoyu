
"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.
We have the following two ways to execute this example:
(1) Execute this code directly.
    $ python pytorch_simple.py
(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize pytorch_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.nn.init as init
import utils
import time
import typing
import collections
from torchvision import datasets
from torchvision import transforms
from dataloaders import load_cifar10

import optuna
#device_ids = [0,1]

def compute_loss_and_accuracy(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss):
    """
    Computes the average loss and the accuracy over the whole dataset
    in dataloader.
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: torch.nn.CrossEntropyLoss()
    Returns:
        [average_loss, accuracy]: both scalar.
    """
    average_loss = 0
    accuracy = 0
    running_corrects = 0.0
    size=0
    #print('dataloder',len(dataloader)) #len(dataloader) express the total number of minibatch. eg. len=156
                                       #and batch_size=64, which means that total number of samples in each 
                                       #epoch is 156*64=9984
    with torch.no_grad():
        for (X_batch, Y_batch) in dataloader:
            # Transfer images/labels to GPU VRAM, if possible
            X_batch = utils.to_cuda(X_batch)
            Y_batch = utils.to_cuda(Y_batch)
            bs=Y_batch.size(0)
            # Forward pass the images through our model
            output_probs = model(X_batch)

            # Compute Loss and Accuracy
            average_loss = loss_criterion(output_probs,Y_batch)
            _,index = torch.max(output_probs,1) #torch.max returns the maximum value and corresponding index location
                                                #in each row. Here we just need the index
            running_corrects += torch.sum(index == Y_batch)
            size+=bs
        accuracy=running_corrects.item() / size
    return average_loss, accuracy

def weights_init(m):  #This weight_init is based on several blogs and materials, By Xiaoyu Zhu.
    if isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

class ExampleModel(nn.Module):

    def __init__(self,
                 trial,
                 image_channels, 
                 num_classes,
                 num_ConvLayer,
                 num_filters,
                 size_kernel,
                 num_DenseLayer,
                 num_units
                ):
        input_size=32
        
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_classes = num_classes
        self.loss_criterion = nn.CrossEntropyLoss()
        #Define the network, Firstly we define the AlexNet:
        self.activation = get_activation(trial)
        # Define the convolutional layers
        Conv_struc=[]
        #self.feature_extractor = nn.Sequential(
        #1. The first layer
        Conv_struc.append(nn.Conv2d(
            in_channels=image_channels,
            out_channels=num_filters[0],
            kernel_size=size_kernel[0]))                 
        self.output_size=input_size - size_kernel[0] +1
        Conv_struc.append(self.activation)  
        Conv_struc.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.output_size=int(self.output_size / 2)
    #2. The next several layers:
        for i in range(1,num_ConvLayer):
            Conv_struc.append(nn.Conv2d(
                in_channels= num_filters[i-1],
                out_channels=num_filters[i],
                kernel_size=size_kernel[i],
                stride=1,
                padding=int(size_kernel[i]/2)
            ))
            Conv_struc.append(self.activation)
        Conv_struc.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.output_size=int(self.output_size / 2)
        self.feature_extractor = nn.Sequential(*Conv_struc) #Define the part of feature_extractor.
        
        #FC layers
        FC_struc=[]
        self.output_size=self.output_size*self.output_size*num_filters[num_ConvLayer-1]
        FC_struc.append(nn.Linear(in_features=self.output_size,out_features=num_units[0]))
        FC_struc.append(self.activation)
        for i in range(1,num_DenseLayer):
            FC_struc.append(nn.Linear(in_features=num_units[i-1],out_features=num_units[i]))
            FC_struc.append(self.activation)
        FC_struc.append(nn.Linear(in_features=num_units[i],out_features=num_classes))
        self.classifier = nn.Sequential(*FC_struc) #Define the part of classifier


    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        conv_out=self.feature_extractor(x)
        res=conv_out.view(conv_out.size(0), -1)
        out=self.classifier(res)
        return out

class Trainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 model: torch.nn.Module,
                 dataloaders: typing.List[torch.utils.data.DataLoader],
                optimizer):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the model
        self.model = model
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = optimizer

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = dataloaders

        # Validate our model everytime we pass through 50% of the dataset
        self.num_steps_per_val = len(self.dataloader_train) // 2
        self.global_step = 0
        self.start_time = time.time()

        # Tracking variables
        self.VALIDATION_LOSS = collections.OrderedDict()
        self.TEST_LOSS = collections.OrderedDict()
        self.TRAIN_LOSS = collections.OrderedDict()
        self.VALIDATION_ACC = collections.OrderedDict()
        self.TEST_ACC = collections.OrderedDict()

        #self.checkpoint_dir = pathlib.Path("checkpoints")

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC[self.global_step] = validation_acc
        self.VALIDATION_LOSS[self.global_step] = validation_loss
        used_time = time.time() - self.start_time
        print(
            f"Epoch: {self.epoch:>2}",
            f"Batches per seconds: {self.global_step / used_time:.2f}",
            f"Global step: {self.global_step:>6}",
            f"Validation Loss: {validation_loss:.2f},",
            f"Validation Accuracy: {validation_acc:.3f}",
            sep="\t")
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC[self.global_step] = test_acc
        self.TEST_LOSS[self.global_step] = test_loss

        self.model.train()

    def should_early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(self.VALIDATION_LOSS.values())[-self.early_stop_count:]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            # Perform a full pass through all the training samples
            for X_batch, Y_batch in self.dataloader_train:
                X_batch = utils.to_cuda(X_batch)
                Y_batch = utils.to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)
                self.TRAIN_LOSS[self.global_step] = loss.detach().cpu().item()

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()

                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                self.global_step += 1
                 # Compute loss/accuracy for all three datasets.
                if should_validate_model():
                    self.validation_epoch()
                    #self.save_model()
                    if self.should_early_stop():
                        print("Early stopping.")
                        return
    


def get_activation(trial):
    activation_names = ['ReLU','Hardtanh','Sigmoid']
    activation_name = trial.suggest_categorical('activation', activation_names)
    
    if activation_name == activation_names[0]:
        activation = nn.ReLU()
    elif activation_name == activation_names[1]:
        activation = nn.Hardtanh()
    else:
        activation = nn.Sigmoid()
    
    return activation

def objective(trial):
    #Set GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #Several hyperparameters:
    # 1. Number of Conv Layers and Number of filter
    num_ConvLayer = trial.suggest_int('num_layer', 3, 7)
    num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 16, 128, 16)) for i in range(num_ConvLayer)]
    size_kernel = [int(trial.suggest_discrete_uniform('kernel_size_'+str(i), 3,11,2)) for i in range(num_ConvLayer)]
    # 2. Number of Dense Layers and number of units in each layer
    num_DenseLayer = trial.suggest_int('num_layer', 2, 4)
    num_units = [int(trial.suggest_discrete_uniform("mid_units_"+str(i), 100, 500, 100)) for i in range(num_DenseLayer)]
    # 3. Generate the optimizers.
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
    lr = trial.suggest_discrete_uniform('lr', 1e-5, 5e-1,1e-5) #Learning rate
    model = ExampleModel(trial,3, 10,num_ConvLayer,num_filters,size_kernel,num_DenseLayer,num_units)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    print(model)
    
    
    
    model.apply(weights_init) #Weight initialization

    
    # Training of the model.
    #model.train()
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        optimizer
    )
    trainer.train()
    
    loss_val_f,acc_val_f=compute_loss_and_accuracy(
            dataloader_val, model, torch.nn.CrossEntropyLoss())
    
#     for step in range(EPOCH):
#         train(model, device, train_loader, optimizer)
#         error_rate = test(model, device, test_loader)
#     return error_rate#accuracy
    return acc_val_f


if __name__ == '__main__':
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    dataloader_train, dataloader_val, dataloader_test = dataloaders

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10000)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
