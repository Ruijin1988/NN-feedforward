import numpy as np

def tanh(x):
    return np.tanh(x)
    
def tanh_deriv(x):
    return (1.0 - np.tanh(x)*np.tanh(x))

class NeuralNetwork:
    def __init__(self, layers):

        self.activation = tanh
        self.activation_deriv = tanh_deriv
            
        self.weights = []
#        for i in range(1, len(layers) - 1):
        for i in range(1, len(layers) - 1):
            #### the weights is from -0.5 to 0.5 ####
            self.weights.append(np.random.random((layers[i-1]+1, layers[i]+1))-0.5)
#            self.weights.append((np.random.random((layers[i]+1, layers[i+1]))-0.5))
#            self.weights.append(np.random.random((layers[i], layers[i+1]))-0.5)
        self.weights.append((np.random.random((layers[-2]+1, layers[-1]))-0.5))

    def train(self, X, y, learning_rate=0.2, epochs=1):

        ####bias setting####
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = k%len(X)
            a = [X[i]]

            #### feedforward ####
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))  # a is the output
#            print a
#            a[-2]=[

            #### last layer error gradient ####
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            #### hidden layer error gradient ####
            for l in range(len(a) - 2, 0, -1):
#                print self.weights
                deltas.append(np.dot(self.weights[l],deltas[-1])*self.activation_deriv(a[1]))

            #### backpropagation ####
            for i in range(len(self.weights)):

                layer = np.array([a[len(self.weights)-i-1]])
                delta = np.array([deltas[i]])

                self.weights[len(self.weights)-i-1] += learning_rate * np.dot(layer.T,delta)


    def predict(self, x):
        x = np.array(x)
        temp = np.ones([len(x)+1])
        temp[0:-1] = x
        a = temp

        for l in range(len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    #Test
nn = NeuralNetwork([2,4,4,4,4,4,1])
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])
nn.train(X,y)
for i in [[0,0], [0,1], [1,0], [1,1]]:
    print(i, nn.predict(i))