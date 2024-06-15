import numpy as np
import math


class Value:
    def __init__(self, data, grad=0.0, _children=(), _op='', label=''):
        self.data = data
        self.grad = grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad = out.grad * 1.0
            other.grad = out.grad * 1.0
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        
        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward
        out = Value(t, (self, ), 'tanh')
        return out

