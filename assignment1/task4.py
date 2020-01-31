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
    output=model.forward(X)
    accuracy=(output.argmax(axis=1) == targets.argmax(axis=1)).mean()
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

            #core-code of training
            output=model.forward(X_batch)            #forward
            model.backward(X_batch,output,Y_batch)   #backward and gain the gradient
            model.w=model.w-learning_rate*model.grad #update the gradient
            
            # Track training loss continuously
            _train_loss = cross_entropy_loss(Y_batch,output)
            train_loss[global_step] = _train_loss
            # Track validation loss / accuracy every time we progress 20% through the dataset
            if global_step % num_steps_per_val == 0:
                _val_loss = cross_entropy_loss(Y_val,model.forward(X_val))
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
X_train=pre_process_images(X_train)
X_val=pre_process_images(X_val)
X_test=pre_process_images(X_test)
print('X_train',X_train.shape,'X_val',X_val.shape,'X_test',X_test.shape)
Y_train = one_hot_encode(Y_train, 10)
Y_val = one_hot_encode(Y_val, 10)
Y_test= one_hot_encode(Y_test,10)

# Hyperparameters
num_epochs = 50
learning_rate = .3
batch_size = 128
l2_reg_lambda = [0.0,0.001]


model, train_loss, val_loss, train_accuracy, val_accuracy = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=0) #For the task4a-c, the condition of no l2-regularization

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
plt.ylim([0.01, .2])
utils.plot_loss(train_loss, "Training Loss")
utils.plot_loss(val_loss, "Validation Loss")
plt.xlabel('Gradient Step')
plt.ylabel('Loss')
plt.legend()
plt.savefig("softmax_loss.png")
plt.show()


# Plot accuracy
plt.ylim([0.8, .95])
utils.plot_loss(train_accuracy, "Training Accuracy")
utils.plot_loss(val_accuracy, "Validation Accuracy")
plt.xlabel('Gradient Step')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("softmax_acc.png")
plt.show()

#---------For Task4e to f
train_loss_total={}
val_loss_total={}
train_accuracy_total={}
val_accuracy_total={}
model={}
for i in range(len(l2_reg_lambda)):
    model[i], train_loss_total[i], val_loss_total[i], train_accuracy_total[i], val_accuracy_total[i] = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=l2_reg_lambda[i])
for i in range(len(l2_reg_lambda)):
    for j in range(10):
        plt.imshow((model[i].w[:-1,j]).reshape(28,28))
        plt.savefig('Weight_lambda_'+str(l2_reg_lambda[i])+'Figure_'+str(j)+'.png')
