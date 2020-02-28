import os
import pathlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import utils
import time
import typing
import collections
from torch import nn
from dataloaders import load_cifar10
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

class Task3_structure(nn.Module):

    def __init__(self,#trial,
                 image_channels, 
                 num_classes,
                 sturcture_choice,
                 drop_out:bool,
                 ):
        #image_channels=3
        #num_classes=10
        
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_classes = num_classes
        if sturcture_choice == 'a':
            # Define the convolutional layers
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(
                    in_channels= 32,
                    out_channels=192,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(
                    in_channels=192,
                    out_channels=224,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(
                    in_channels=224,
                    out_channels=192,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

            )
            

            
            self.num_output_features = 192*2*2
            # Initialize our last fully connected layer
            # Inputs all extracted features from the convolutional layers
            # Outputs num_classes predictions, 1 for each class.
            # There is no need for softmax activation function, as this is
            # included with nn.CrossEntropyLoss
            self.classifier = nn.Sequential(
                nn.Linear(in_features=self.num_output_features,out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64,out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64,out_features=96),
                nn.ReLU(),
                nn.Linear(in_features=96, out_features=num_classes),
            )
        elif sturcture_choice == 'b':
            
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=224,
                    kernel_size=3,
                    stride=1,
                    padding=1, 
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=224,
                    out_channels=224,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(
                    in_channels= 224,
                    out_channels=224,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels= 224,
                    out_channels=224,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(
                    in_channels=224,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            
            self.num_output_features = 64*4*4
            # Initialize our last fully connected layer
            # Inputs all extracted features from the convolutional layers
            # Outputs num_classes predictions, 1 for each class.
            # There is no need for softmax activation function, as this is
            # included with nn.CrossEntropyLoss
            self.classifier = nn.Sequential(
                nn.Linear(in_features=self.num_output_features,out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64,out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64,out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=num_classes),
            )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        conv_out=self.feature_extractor(x)
        res=conv_out.view(conv_out.size(0), -1)
        out=self.classifier(res)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


class Trainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 l2_lambda: float,
                 early_stop_count: int,
                 epochs: int,
                 model: torch.nn.Module,
                 dataloaders: typing.List[torch.utils.data.DataLoader],
                 optimizer,
                 regularization
                ):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.early_stop_count = early_stop_count
        self.epochs = epochs
            
        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the model
        self.model = model
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)
        print(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        if optimizer == 'SGD':
            if regularization:
                self.optimizer = torch.optim.SGD(self.model.parameters(),self.learning_rate,weight_decay=self.l2_lambda)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(),self.learning_rate)
        elif optimizer == 'Adam':
            if regularization:
                self.optimizer = torch.optim.Adam(self.model.parameters(),self.learning_rate,weight_decay=self.l2_lambda)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(),self.learning_rate)
        elif optimizer == 'RMSprop':
            if regularization:
                self.optimizer = torch.optim.Adam(self.model.parameters(),self.learning_rate,weight_decay=self.l2_lambda)
            else:
                self.optimizer = torch.optim.RMSprop(self.model.parameters(),self.learning_rate)

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

        self.checkpoint_dir = pathlib.Path("checkpoints")

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
                    self.save_model()
                    if self.should_early_stop():
                        print("Early stopping.")
                        return

    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            validation_losses = list(self.VALIDATION_LOSS.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.TRAIN_LOSS, label="Training loss")
    utils.plot_loss(trainer.VALIDATION_LOSS, label="Validation loss")
    utils.plot_loss(trainer.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.VALIDATION_ACC, label="Validation Accuracy")
    utils.plot_loss(trainer.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    epochs = 10
    batch_size = 64
    learning_rate = 0.00034
    l2_lambda= 0.0001
    early_stop_count = 6
    optimizer='Adam'
    regularization =True
    dataloaders = load_cifar10(batch_size)

    model = Task3_structure(image_channels=3, num_classes=10,sturcture_choice='b', drop_out=False)
    model.apply(weights_init)
    trainer = Trainer(
        batch_size,
        learning_rate,
        l2_lambda,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        optimizer,
        regularization
    )
    trainer.train()
    trainer.load_best_model()
    create_plots(trainer, "task3")
    #Output the ACC and Loss
    dataloader_train, dataloader_val, dataloader_test = dataloaders
    loss_train_f,acc_train_f=compute_loss_and_accuracy(
            dataloader_train, model, torch.nn.CrossEntropyLoss())    
    loss_val_f,acc_val_f=compute_loss_and_accuracy(
        dataloader_val, model, torch.nn.CrossEntropyLoss())
    loss_test_f,acc_test_f=compute_loss_and_accuracy(
        dataloader_test, model, torch.nn.CrossEntropyLoss())
    print('Final Loss on Training Dataset=',loss_train_f)
    print('Final Acc on Training Dataset=',acc_train_f)
    print('Final Loss on Validation Dataset=',loss_val_f)
    print('Final Acc on Validation Dataset=',acc_val_f)
    print('Final Loss on Test Dataset=',loss_test_f)
    print('Final Acc on Test Dataset=',acc_test_f)