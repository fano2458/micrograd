from src.grad import Value


def MSE_loss(ys, yp):
    loss = sum((y_p - y_t)**2 for y_t, y_p in zip(ys, yp))
                
    return loss


def hinge_loss(ys, yp):    
    total_loss = [(1 + -yi*scorei).relu() for yi, scorei in zip(ys, yp)]
    loss = sum(total_loss) * (1.0/len(total_loss))
    
    return loss