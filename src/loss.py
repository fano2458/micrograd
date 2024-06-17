from src.grad import Value


def MSE_loss(ys, yp):
    loss = sum((y_p - y_t)**2 for y_t, y_p in zip(ys, yp))
                
    return loss


def hinge_loss(ys, yp):    
    total_loss = [(1 + -yi*scorei).relu() for yi, scorei in zip(ys, yp)]
    loss = sum(total_loss) * (1.0/len(total_loss))
    
    return loss


def softmax(yp):
    exp_values = [val.exp() for val in yp]
    exp_values_sum = sum(exp_values)
 
    return [val/exp_values_sum for val in exp_values]


def encode(ys):
    y_onehot = [0] * 10 # assuming 10 classes
    y_onehot[ys] = 1
    
    return y_onehot


def cross_entropy_loss(ys, yp):
    y_pred = [softmax(y_p) for y_p in yp]
    ys = [encode(y_s) for y_s in ys]

    losses = []
    for y_t, y_p in zip(ys, y_pred):
        losses.append(sum((-1 * y_t[i] * y_p[i].log() for i in range(len(y_p)))))
    
    return sum(losses) / len(losses)


# import torch.nn as nn
# import torch
 
# loss = nn.CrossEntropyLoss()
 
# y_pred = torch.tensor([[1.4, 0.4, 1.1, 0.1, 2.3], 
#                        [2, 0.2, 4, 0.1, 0.7]])
# y_true = torch.tensor([0, 1])
 
# cross_entropy = loss(y_pred, y_true)
 
# print("Loss: ", cross_entropy.item())

# y_p = [[Value(1.4), Value(0.4), Value(1.1), Value(0.1), Value(2.3)], 
#        [Value(2), Value(0.2), Value(4), Value(0.1), Value(0.7)]]
# y_t = [0, 1]

# print(cross_entropy_loss(y_t, y_p))