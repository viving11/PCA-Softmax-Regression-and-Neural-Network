"""
    Extra credit: direction (3)
        -> most changes are in:
            1) __inint__(): new variables for momentums
            2) update(): new update rules
            
        -> Implement AMSGrad: https://www.ruder.io/deep-learning-optimization-2017/#improvingadam
            1) Compute gradients
            2) Compute 1st order momentum
            3) Compute 2nd order momentum
            4) Compute max 2nd order momentum
            5) Update weights using 1st & 2nd order momentum
"""

import numpy as np

class NN_amsgrad:
    def __init__(self, activation_function, loss_function, hidden_layers=[1024], input_d=784, output_d=10):
        # Model weights by layers
        self.weights = []
        self.biases = []
        # 1st-order gradient momentum by layers
        self.m_weights = []
        self.m_biases = []
        # 2nd-order gradient momentum by layers
        self.v_weights = []
        self.v_biases = []
        # bias-corrected 2nd-order gradient momentum by layers
        self.vhat_weights = []
        self.vhat_biases = []

        self.activation_function = activation_function
        self.loss_function = loss_function

        # Initialization of weights and biases and their 1st and 2nd order momentums
        d1 = input_d
        hidden_layers.append(output_d)
        for d2 in hidden_layers:
            self.weights.append(np.random.randn(d2, d1)*np.sqrt(2.0/d1))
            self.biases.append(np.zeros((d2,1)))
            self.m_weights.append(np.zeros((d2, d1)))
            self.m_biases.append(np.zeros((d2,1)))
            self.v_weights.append(np.zeros((d2, d1)))
            self.v_biases.append(np.zeros((d2,1)))
            self.vhat_weights.append(np.zeros((d2, d1)))
            self.vhat_biases.append(np.zeros((d2,1)))
            d1 = d2

    def print_model(self):
        """
        This function prints the shapes of weights and biases for each layer.
        """
        print("activation:{}".format(self.activation_function.__class__.__name__))
        print("loss function:{}".format(self.loss_function.__class__.__name__))
        for idx,(w,b) in enumerate(zip(self.weights, self.biases),1):
            print("Layer {}\tw:{}\tb:{}".format(idx, w.shape, b.shape))

    def predict(self, X):
        D = X
        ws = self.weights
        bs = self.biases
        for w,b in zip(ws[:-1], bs[:-1]):
            D = self.activation_function.activate(np.matmul(w,D)+b) 
            # Be careful of the broadcasting here: (d,N) + (d,1) -> (d,N).
        Yhat = np.matmul(ws[-1], D)+bs[-1]
        return np.argmax(Yhat, axis=0)

    def compute_gradients(self, X, Y):
        ws = self.weights
        bs = self.biases
        D_stack = []

        D = X
        D_stack.append(D)
        num_layers = len(ws)
        for idx in range(num_layers-1):
            # TODO 2: Calculate D for forward pass (which is similar to self.predict). 
            # This intermediate results too will then be stored to D_stack.

            ### YOUR CODE HERE ###
            D = np.matmul(self.weights[idx], D_stack[-1]) + bs[idx]
            D = self.activation_function.activate(D)
            D_stack.append(D)

        Yhat = np.matmul(ws[-1], D) + bs[-1]
        training_loss = self.loss_function.loss(Y, Yhat)
        '''
        '''
        grad_bs = []
        grad_Ws = []

        grad = self.loss_function.lossGradient(Y,Yhat)
        grad_b = np.sum(grad, axis=1, keepdims=1)
        grad_W = np.matmul(grad, D_stack[num_layers-1].transpose())
        grad_bs.append(grad_b)
        grad_Ws.append(grad_W)
        for idx in range(num_layers-2, -1, -1):
            # TODO 3: Calculate grad_bs and grad_Ws, which are lists of gradients for b's and w's of each layer. 
            # Take a look at the update step if you are not sure about the format. Notice that we first store the
            # gradients for each layer in a reversed order. The two lists are reversed before returned.

            #1. Update grad for the current layer 
            ### YOUR CODE HERE ###
            grad = np.matmul(ws[idx+1].T, grad) * self.activation_function.backprop_grad(D_stack[idx+1])

            #2. Calculate grad_b (gradient with respect to b of the current layer)
            ### YOUR CODE HERE ###
            grad_b = np.sum(grad, axis=1, keepdims=1)
        
            #3. Calculate grad_W (gradient with respect to W of the current layer)
            ### YOUR CODE HERE ###
            grad_W = np.matmul(grad, D_stack[idx].transpose())
            
            grad_bs.append(grad_b)
            grad_Ws.append(grad_W)

        grad_bs, grad_Ws = grad_bs[::-1], grad_Ws[::-1] # Reverse the gradient lists
        return training_loss, grad_Ws, grad_bs

    def update(self, grad_Ws, grad_bs, learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-6):
        # Update the weights and biases
        num_layers = len(grad_Ws)
        ws = self.weights
        bs = self.biases
        m_ws = self.m_weights
        m_bs = self.m_biases
        v_ws = self.v_weights
        v_bs = self.v_biases
        vhat_ws = self.vhat_weights
        vhat_bs = self.vhat_biases

        for idx in range(num_layers):
            # 1st-order momentum
            m_ws[idx] = beta1 * m_ws[idx] + (1-beta1) * grad_Ws[idx]
            m_bs[idx] = beta1 * m_bs[idx] + (1-beta1) * grad_bs[idx]
            # 2nd-order momentum
            v_ws[idx] = beta2 * v_ws[idx] + (1-beta2) * np.power(grad_Ws[idx], 2)
            v_bs[idx] = beta2 * v_bs[idx] + (1-beta2) * np.power(grad_bs[idx], 2)
            # bias-corrected 2-nd order momentum
            vhat_ws[idx] = np.maximum(vhat_ws[idx], v_ws[idx])
            vhat_bs[idx] = np.maximum(vhat_bs[idx], v_bs[idx])

            # Update weights & biases
            ws[idx] -= (learning_rate * m_ws[idx]) / (np.sqrt(vhat_ws[idx]) + epsilon)
            bs[idx] -= (learning_rate * m_bs[idx]) / (np.sqrt(vhat_bs[idx]) + epsilon)
        
        self.weights = ws
        self.biases = bs
        self.m_weights = m_ws
        self.m_biases = m_bs
        self.v_weights = v_ws
        self.v_biases = v_bs
        self.vhat_weights = vhat_ws
        self.vhat_biases = vhat_bs
        
        return 

class activationFunction:
    def activate(self,X):
        """
        The output of activate should have the same shape as X
        """
        raise NotImplementedError("Abstract class.")

    def backprop_grad(self, grad):
        """
        The output of backprop_grad should have the same shape as X
        """
        raise NotImplementedError("Abstract class.")

class Relu(activationFunction):
    def activate(self,X):
        """
        The output of activate should have the same shape as X
        """
        return X*(X>0)

    def backprop_grad(self, X):
        """
        The output of backprop_grad should have the same shape as X
        """
        return (X>0).astype(np.float64)

class Linear(activationFunction):
    def activate(self,X):
        """
        The output of activate should have the same shape as X
        """
        return X
    def backprop_grad(self,X):
        """
        The output of backprop_grad should have the same shape as X
        """
        return np.ones(X.shape, dtype=np.float64)

class LossFunction:
    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """
        raise NotImplementedError("Abstract class.")

    def lossGradient(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are in 
        Yhat; compute the gradient of the loss with respect to Yhat
        """
        raise NotImplementedError("Abstract class.")

class SquaredLoss(LossFunction):
    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """
        # TODO 0: loss function for squared loss.

        ### YOUR CODE HERE ###
        K, N = Yhat.shape
        return 1/(2*N) * np.power(Yhat-Y, 2).sum()

    def lossGradient(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are in 
        Yhat; compute the gradient of the loss with respect to Yhat
        """
        #TODO 1: gradient for squared loss.

        ### YOUR CODE HERE ###
        K, N = Yhat.shape
        return 1/N * (Yhat - Y)


class CELoss(LossFunction):
    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """
        #TODO 4: loss function for cross-entropy loss.

        ### NOT REQUIRED FOR THIS PROJ, YOU CAN DO IT FOR FUN ###
        raise NotImplementedError("Implement CELoss.")

    def lossGradient(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are in 
        Yhat; compute the gradient of the loss with respect to Yhat, which
        has the same shape of Yhat and Y.
        """
        #TODO 5: gradient for cross-entropy loss.

        ### NOT REQUIRED FOR THIS PROJ, YOU CAN DO IT FOR FUN ###
        raise NotImplementedError("Implement CELoss")
