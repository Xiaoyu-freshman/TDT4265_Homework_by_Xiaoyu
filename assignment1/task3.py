import numpy as np
import utils
import matplotlib.pyplot as plt
from task3a import cross_entropy_loss, BinaryModel, pre_process_images
np.random.seed(0)

def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: BinaryModel) -> float: 
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 1]
        model: model of class BinaryModel
    Returns:
        Accuracy (float)
    """
    # calculating the accuracy
    output=model.forward(X)
    mark=abs(output-targets)<=0.5
    count=np.zeros(output.shape)
    count[mark]=1
    accuracy = np.sum(count)/len(output)
    return accuracy


def train(
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        l2_reg_lambda: float # Task 3 hyperparameter. Can be ignored before this.
        ):
    """
        Function that implements logistic regression through mini-batch
        gradient descent for the given hyperparameters
    """
    global X_train, X_val, X_test
    
    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    model = BinaryModel(l2_reg_lambda)
    
    model.w=np.zeros([(X_train.shape[1]),1])
    
    
    
    global_step = 0
    for epoch in range(num_epochs):
        
        for step in range(num_batches_per_epoch):
            # Select our mini-batch of images / labels
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]
            
            #core-code of training
            output=model.forward(X_batch)             #forward
            model.backward(X_batch,output,Y_batch)    #backward and gain the gradient
            model.w=model.w-learning_rate*model.grad  #update the gradient
            #xiaoyu

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
            #In this task, we do not need the early-stopping.
    return model, train_loss, val_loss, train_accuracy, val_accuracy
# Load dataset
category1, category2 = 2, 3
validation_percentage = 0.1
X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_binary_dataset(
    category1, category2, validation_percentage)
#preprocessing
X_train=pre_process_images(X_train)
X_val=pre_process_images(X_val)
X_test=pre_process_images(X_test)
    
# hyperparameters
num_epochs = 80
learning_rate = 0.2
batch_size = 128
l2_reg_lambda = np.array([1,0.1,0.01,0.001])

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
    
# Plot Accuracy of different values of lambda
for i in range(len(l2_reg_lambda)):
    utils.plot_loss(val_accuracy_total[i], "lamba="+str(l2_reg_lambda[i]))
plt.xlabel('Gradient Step')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("Accuracy_with_different_lamba.png")
plt.show()

#Plot the L2 Length
L2_length=np.zeros(l2_reg_lambda.shape)
for i in range(len(l2_reg_lambda)):
    L2_length[i]=np.dot(model[i].w.transpose(),model[i].w)
print(L2_length)
plt.plot(l2_reg_lambda,L2_length)
plt.xlabel('Lamba')
plt.ylabel('L2 Norm')
plt.title('L2 Norm VS. Lamba')
plt.savefig('L2_Norm_vs_Lamba.png')
plt.show()

#Plot the weight of different lambda
for i in range(len(l2_reg_lambda)):
    plt.imshow((model[i].w[:-1]).reshape(28,28))
    plt.savefig('Weight_lambda_'+str(l2_reg_lambda[i])+'.png')