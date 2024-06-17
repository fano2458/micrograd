import math


class Value:
    """
    This class represents a computational node in a computational graph.
    It holds data, gradient, and performs automatic differentiation.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        """
        Initializes a Value object.
        Args:
            data: The numerical data of the node.
            _children: A tuple of Value objects, the children of this node in the computation graph (internal use).
            _op: The operation that created this node (internal use).
            label: A label for the node (optional).
        """
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        """
        Returns a string representation of the Value object.
        """
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        """
        Performs addition between two Value objects or a Value and a number.
        Args:
            other: The other operand (Value object or number).
        Returns:
            A new Value object representing the sum.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        """
        Performs multiplication between two Value objects or a Value and a number.
        Args:
            other: The other operand (Value object or number).
        Returns:
            A new Value object representing the product.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        """
        Calculates the power of the Value object (** operator).
        Args:
            other: The exponent (only integer or float supported).
        Raises:
            AssertionError: If the exponent is not an integer or float.
        Returns:
            A new Value object representing the result of the exponentiation.
        """
        assert isinstance(other, (int, float)), 'only supported for int/float'
        out = Value(self.data**other, (self, ), f'**{other}')
        
        def _backward():
            self.grad += other * self.data ** (other-1) * out.grad   
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        """
        Performs division using multiplication by the inverse (other**-1).
        """
        return self * other**-1
    
    def __neg__(self):
        """
        Returns the negative of the Value object.
        """
        return self * -1
    
    def __sub__(self, other):
        """
        Performs subtraction using addition with the negative of the other operand.
        """
        return self + (-other)
    
    def exp(self):
        """
        Calculates the exponential of the Value object.
        Returns:
            A new Value object representing the exponential.
        """
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def log(self):
        """
        Calculates the log of the Value object
        Returns:
            A new Value object representing the log
        """
        x = self.data
        out = Value(math.log(x), (self, ), 'log')
        
        def _backward():
            self.grad += out.grad * (1 / x)
        self._backward = _backward
        
        return out
    
    def tanh(self):
        """
        Calculates the hyperbolic tangent of the Value object.
        Returns:
            A new Value object representing the tanh.
        """
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        """
        Calculates Rectified Linear Unit of the Value object.
        Returns:
            A new Value object representing the ReLU
        """
        x = self.data
        t = x if x > 0 else 0
        out = Value(t, (self, ), 'relu')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out

    def backward(self):
        """
        Performs the backward pass to compute gradients of all nodes in the computational graph.
        """
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        
    def __radd__(self, other):
        """
        Supports addition where the Value object is on the right side (e.g., 3 + x).
        """
        return self + other
        
    def __rmul__(self, other):
        """
        Supports multiplication where the Value object is on the right side (e.g., 2 * x).
        """
        return self * other
    
    def __rsub__(self, other):
        """
        Supports subtraction where the Value object is on the right side (e.g., x - 3).
        """
        return -self + other
    