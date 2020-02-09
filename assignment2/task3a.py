import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785]
    """
    X_std=np.std(X)
    X_mean=np.mean(X)
    X=(X-X_mean)/X_std
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    X1=np.ones((X.shape[0],1))
    X=np.append(X,X1,axis=1)
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    ce = targets * np.log(outputs)
    return -1*np.sum(ce)/targets.shape[0] 


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Define number of input nodes
        self.I = 784+1
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_improved_weight_init = use_improved_weight_init
        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        self.grads = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            g = np.zeros(w_shape)
            self.ws.append(w)
            self.grads.append(g)
            prev = size
            
    def forward(self, X: np.ndarray) -> np.ndarray: #improved sigmoid是隐藏层的激活函数，softmax是输出层的激活函数
        """                                         
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        Y=[]
        zs=[]
        for i in range(len(self.neurons_per_layer)):
            if i ==0:
                z=np.dot(X,self.ws[i])
                zs.append(z) #save intermediate values
                if self.use_improved_sigmoid:
                    y=1.7159*(np.exp(2.0/3.0*z)-np.exp(-2.0/3.0*z))/(np.exp(2.0/3.0*z)+np.exp(-2.0/3.0*z)) #improved sigmoid
                else:
                    y=1.0/(1.0+np.exp(-1*z)) #standard sigmoid
                Y.append(y)

            else:
                z=np.dot(np.array(Y[i-1]),self.ws[i])
                zs.append(z)
                if i != len(self.neurons_per_layer)-1: #For the middle layers
                    if self.use_improved_sigmoid:
                        y=1.7159*(np.exp(2.0/3.0*z)-np.exp(-2.0/3.0*z))/(np.exp(2.0/3.0*z)+np.exp(-2.0/3.0*z)) #improved sigmoid
                    else:
                        y=1.0/(1.0+np.exp(-1*z)) #standard sigmoid
                    Y.append(y)
                else: #for the output layer
                    y=np.exp(z)
                    y=y/np.sum(y,axis=1,keepdims=True) #Softmax function, for output layer     
                    Y.append(y)
        return Y,zs

    def backward(self, X: np.ndarray, outputs: np.ndarray,  #对于是否存储中间变量，暂且观望。
                 targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
        Y,zs=self.forward(X)
        delta=[]
        for i in range(-1,-1*(len(self.neurons_per_layer)+1),-1):
            if self.use_improved_sigmoid: #for improved sigmoid function
                if i==-1:    
                    #The Last Layer (the last hidden layer and output layer)
                    a_l_before=np.array(Y[i-1])
                    delta_tmp=-(targets-outputs)
                    delta.append(delta_tmp)
                    self.grads[i]=np.dot(a_l_before.transpose(),delta[-1*(i+1)])/X.shape[0]
                elif abs(i)== len(self.neurons_per_layer):
                    #The First Layer (input layer and the first hidden layer)
                    d_sigmoid=1.7159*(2.0/3.0)*(1.0-(np.array(Y[i])/1.7159)**2)
                    delta_tmp=np.dot(np.array(delta[-1*(i+2)]),self.ws[i+1].transpose())*d_sigmoid
                    delta.append(delta_tmp)
                    self.grads[i]=np.dot(X.transpose(),np.array(delta[-1*(i+1)]))/X.shape[0]
                else:
                    #The other layers
                    d_sigmoid=1.7159*(2.0/3.0)*(1.0-(np.array(Y[i])/1.7159)**2)
                    delta_tmp=np.dot(np.array(delta[-1*(i+2)]),self.ws[i+1].transpose())*d_sigmoid
                    delta.append(delta_tmp)
                    self.grads[i]=np.dot(np.array(Y[i-1]).transpose(),delta[-1*(i+1)])/X.shape[0]
                
            else: #for standard sigmoid function
                if i==-1:
                    a_l_before=np.array(Y[i-1])
                    #The Last Layer (the last hidden layer and output layer)
                    delta_tmp=-(targets-outputs)   
                    delta.append(delta_tmp)
                    self.grads[i]=np.dot(a_l_before.transpose(),delta[-1*(i+1)])/X.shape[0]
                elif abs(i)== len(self.neurons_per_layer):    
                    #The First Layer (input layer and the first hidden layer)
                    d_sigmoid=np.array(Y[i])*(1-np.array(Y[i]))
                    delta_tmp=np.dot(np.array(delta[-1*(i+2)]),self.ws[i+1].transpose())*d_sigmoid
                    delta.append(delta_tmp)
                    self.grads[i]=np.dot(X.transpose(),np.array(delta[-1*(i+1)]))/X.shape[0]
                else:
                    #The other layers
                    d_sigmoid=np.array(Y[i])*(1-np.array(Y[i]))
                    delta_tmp=np.dot(np.array(delta[-1*(i+2)]),self.ws[i+1].transpose())*d_sigmoid
                    delta.append(delta_tmp)
                    self.grads[i]=np.dot(np.array(Y[i-1]).transpose(),delta[-1*(i+1)])/X.shape[0]
        
        return self.grads
        

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    one_hot=np.zeros((Y.shape[0],num_classes))
    for i in range(Y.shape[0]):
        one_hot[i,Y[i]]=1
    return one_hot


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws): 
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = np.array(model.forward(X)[0][-1]) #Return item is a list consisting of activations in each layer._By Xiaoyu 
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = np.array(model.forward(X)[0][-1]) #Return item is a list consisting of activations in each layer._By Xiaoyu
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = np.array(model.forward(X)[0][-1]) #Return item is a list consisting of activations in each layer._By Xiaoyu
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist(0.1)
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64,10]# To test the extending performance, you can set this parameter like: [64, 32, 16,10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
