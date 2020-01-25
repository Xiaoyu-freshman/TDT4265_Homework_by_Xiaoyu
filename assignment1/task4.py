import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from task4a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float: 
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    accuracy = 0
    return accuracy


def train(
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        l2_reg_lambda: float # Task 3 hyperparameter
        ):
    global X_train, X_val, X_test
    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}

    # Intialize our model
    model = SoftmaxModel(l2_reg_lambda)

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            _train_loss = 0
            train_loss[global_step] = _train_loss
            
            # Track training loss continuously
            _train_loss = 0
            train_loss[global_step] = _train_loss
            # Track validation loss / accuracy every time we progress 20% through the dataset
            if global_step % num_steps_per_val == 0:
                _val_loss = 0
                val_loss[global_step] = _val_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1
    return model, train_loss, val_loss, train_accuracy, val_accuracy


# Load dataset
validation_percentage = 0.1
X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
    validation_percentage)


# Hyperparameters
num_epochs = 50
learning_rate = .3
batch_size = 128
l2_reg_lambda = 0.001

model, train_loss, val_loss, train_accuracy, val_accuracy = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=l2_reg_lambda)

print("Final Train Cross Entropy Loss:",
      cross_entropy_loss(Y_train, model.forward(X_train)))
print("Final Validation Cross Entropy Loss:",
      cross_entropy_loss(Y_test, model.forward(X_test)))
print("Final Test Cross Entropy Loss:",
      cross_entropy_loss(Y_val, model.forward(X_val)))


print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
print("Final Test accuracy:", calculate_accuracy(X_test, Y_test, model))


# Plot loss
#plt.ylim([0.01, .2])
utils.plot_loss(train_loss, "Training Loss")
utils.plot_loss(val_loss, "Validation Loss")
plt.legend()
plt.savefig("softmax_train_loss.png")
plt.show()


# Plot accuracy
#plt.ylim([0.8, .95])
utils.plot_loss(train_accuracy, "Training Accuracy")
utils.plot_loss(val_accuracy, "Validation Accuracy")
plt.legend()
plt.show()
