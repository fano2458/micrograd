import random
from src.grad import Value


class Module:
    """
    This is a base class for neural network modules.
    It provides functionalities for zeroing gradients and retrieving parameters.
    """
    def zero_grad(self):
        """
        Zeros the gradients of all parameters within the module.
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """
        Returns a list of all parameters (Value objects) within the module.
        """
        return []


class Neuron(Module):
    """
    This class represents a single neuron in a neural network.
    It has weights, a bias, and performs a tanh activation function.
    """
    def __init__(self, n):
        """
        Initializes a Neuron object.
        Args:
            n: The number of input features (dimensions of the input vector).
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n)]  # List of weights (Value objects)
        self.b = Value(random.uniform(-1, 1))  # Bias term (Value object)

    def __call__(self, x):
        """
        Performs the forward pass for the neuron.
        Args:
            x: The input vector (Value object).
        Returns:
            The output of the neuron after applying the tanh activation function (Value object).
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)  # Weighted sum of inputs and bias
        out = act.tanh()  # Apply tanh activation
        return out

    def parameters(self):
        """
        Returns a list containing the weights and bias of the neuron (Value objects).
        """
        return self.w + [self.b]


class Layer(Module):
    """
    This class represents a layer in a neural network.
    It contains multiple neurons and performs the forward pass for the entire layer.
    """
    def __init__(self, nin, nout):
        """
        Initializes a Layer object.
        Args:
            nin: The number of input features (dimensions of the input vector).
            nout: The number of output features (dimensions of the output vector).
        """
        self.neurons = [Neuron(nin) for _ in range(nout)]  # List of Neuron objects

    def __call__(self, x):
        """
        Performs the forward pass for the layer.
        Args:
            x: The input vector (Value object).
        Returns:
            The output of the layer (a list of Value objects for multiple outputs, or a single Value object for single output).
        """
        outs = [n(x) for n in self.neurons]  # Get outputs from each neuron
        return outs[0] if len(outs) == 1 else outs  # Return single output or list of outputs

    def parameters(self):
        """
        Returns a list containing all parameters (Value objects) within the layer (from all neurons).
        """
        return [params for n in self.neurons for params in n.parameters()]


class MLP(Module):
    """
    This class represents a Multi-Layer Perceptron (MLP) neural network.
    It consists of a sequence of Layers.
    """
    def __init__(self, nin, nouts):
        """
        Initializes an MLP object.
        Args:
            nin: The number of input features (dimensions of the input vector).
            nouts: The number of output features (dimensions of the output vector).
        """
        sz = [nin] + nouts  # List of layer sizes (including input and output)
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]  # List of Layer objects

    def __call__(self, x):
        """
        Performs the forward pass for the entire MLP.
        Args:
            x: The input vector (Value object).
        Returns:
            The output of the MLP (Value object).
        """
        for layer in self.layers:
            x = layer(x)  # Pass the input through each layer sequentially
        return x

    def parameters(self):
        """
        Returns a list containing all parameters (Value objects) within the MLP (from all layers and neurons).
        """
        return [params for l in self.layers for params in l.parameters()]
    
    
def L1_loss(model, alpha = 1e-4):
    """
    Calculates L1 regularization penalty
    """
    reg_loss = alpha * sum((abs(p) for p in model.parameters()))
    
    return reg_loss
    
    
def L2_loss(model, alpha = 1e-4):
    """
    Calculates L2 regularization penalty
    """
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    
    return reg_loss
    