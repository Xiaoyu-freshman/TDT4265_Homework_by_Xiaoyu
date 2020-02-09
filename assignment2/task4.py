import datetime
starttime = datetime.datetime.now()
import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images

np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    output=model.forward(X)[0][-1]
    accuracy=(output.argmax(axis=1) == targets.argmax(axis=1)).mean()
    return accuracy


def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_momentum: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets
    for i in range(len(model.neurons_per_layer)):
        if model.use_improved_weight_init:
            model.ws[i]=np.random.normal(0,1/model.ws[i].shape[0],model.ws[i].shape)
        else:
            model.ws[i]=np.random.uniform(-1,1,model.ws[i].shape)
    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    v=[]

    global_step = 0
    for epoch in range(num_epochs):
        if use_momentum:
            for i in range(len(model.neurons_per_layer)):
                v_tmp=np.zeros(model.ws[i].shape)
                v.append(v_tmp)
            learning_rate=0.02
        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]
            
            #core-code of training
            output=model.forward(X_batch)[0][-1]            #forward
            model.backward(X_batch,output,Y_batch)          #backward and gain the gradient
            for i in range(len(model.neurons_per_layer)):
                if not use_momentum:
                    model.ws[i]=model.ws[i]-learning_rate*model.grads[i] #update the gradient
                else:
                    v[i]=momentum_gamma*v[i]-learning_rate*model.grads[i]
                    model.ws[i]=model.ws[i]+v[i]
            
            
            # Track train / validation loss / accuracy
            if (global_step % num_steps_per_val) == 0:
                _val_loss = _val_loss = cross_entropy_loss(Y_val,model.forward(X_val)[0][-1])
                val_loss[global_step] = _val_loss

                _train_loss = cross_entropy_loss(Y_batch,output)
                train_loss[global_step] = _train_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)
            
            global_step += 1
        #for task3a: Shuffle data.
        if use_shuffle:
            index=np.arange(X_train.shape[0])
            np.random.shuffle(index)
            X_train=X_train[index]
            Y_train=Y_train[index]
    return model, train_loss, val_loss, train_accuracy, val_accuracy


if __name__ == "__main__":
    # Load dataset
    validation_percentage = 0.2
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
        validation_percentage)
    #Preprocessing_by_Xiaoyu
    X_train=pre_process_images(X_train)
    X_val=pre_process_images(X_val)
    X_test=pre_process_images(X_test)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    Y_test= one_hot_encode(Y_test,10)
    
    
    # Hyperparameters
    num_epochs = 20
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [60,60,10]#[64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    # Settings for task 3. Keep all to false for task 2.
    use_shuffle = [True]
    use_improved_sigmoid = [True]
    use_improved_weight_init = [True]
    use_momentum = [True]
    
    #comparing for different tricks
    train_loss={}
    val_loss={}
    train_accuracy={}
    val_accuracy={}
    #The command below is the same as that in task3.py. If you want to test different tricks, please keep the
    #format of command as this.
    for i in range(len(use_shuffle)):  
        print('use_shuffle=',use_shuffle[i],' use_improved_sigmoid=',use_improved_sigmoid[i],' use_improved_weight_init='
             ,use_improved_weight_init[i],' use_momentum=',use_momentum[i])
        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid[i],
            use_improved_weight_init[i])
        print('model.use_improved_sigmoid',model.use_improved_sigmoid)
        print('model.use_improved_weight_init',model.use_improved_weight_init)
        model, train_loss[i], val_loss[i], train_accuracy[i], val_accuracy[i] = train(
            model,
            [X_train, Y_train, X_val, Y_val, X_test, Y_test],
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_shuffle=use_shuffle[i],
            use_momentum=use_momentum[i],
            momentum_gamma=momentum_gamma)
        print("Final Train Cross Entropy Loss:",
              cross_entropy_loss(Y_train, model.forward(X_train)[0][-1]))
        print("Final Validation Cross Entropy Loss:",
              cross_entropy_loss(Y_val, model.forward(X_val)[0][-1]))
        print("Final Test Cross Entropy Loss:",
              cross_entropy_loss(Y_test, model.forward(X_test)[0][-1]))
        print("Final Train accuracy:",
              calculate_accuracy(X_train, Y_train, model))
        print("Final Validation accuracy:",
              calculate_accuracy(X_val, Y_val, model))
        print("Final Test accuracy:",
              calculate_accuracy(X_test, Y_test, model))
        
    # Plot loss
    plt.figure(figsize=(40, 16))
    plt.subplot(1, 2, 1)
    plt.ylim([0.05, .4])
    utils.plot_loss(train_loss[0], "Training Loss")
    utils.plot_loss(val_loss[0], "Validation Loss")
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    # Plot accuracy
    plt.ylim([0.9, 1.0])
    utils.plot_loss(train_accuracy[0], "Training Accuracy")
    utils.plot_loss(val_accuracy[0], "Validation Accuracy")
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    plt.savefig("task4d.png")
    plt.show()


endtime = datetime.datetime.now()
print ('Running Time ',endtime - starttime)