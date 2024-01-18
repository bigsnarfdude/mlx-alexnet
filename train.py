import time
import pickle

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from alex_net import AlexNet

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count



def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict


def loss_fn(model, X, y):
    p = model(X)
    xe = nn.losses.cross_entropy(p, y)
    mx.simplify(xe)
    return mx.mean(xe)


def main():
    seed = 1337 
    num_layers = 2
    hidden_dim = 16
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    #np.random.seed(seed)

    # Load the data
    trainingData = unpickle('./cifar100/train')
    train_images = map(mx.array, trainingData['data']) # needs split 80-90 percent
    train_labels = map(mx.array, trainingData['fine_labels']) # needs split 80-90 percent
    #test_images, test_labels  = validation stage
    #eval 
    #for item in trainingData:
    #    print(item, type(trainingData[item]))
    #print(trainingData['data'][0])

    # Load the model
    model = AlexNet()
    print(model)
    mx.eval(model.parameters())
    

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)
    print(loss_and_grad_fn)
    print(optimizer)

    for row in range(500):
        data = trainingData['data'][row]
        filenames = trainingData['filenames'][row]
        fine_labels = trainingData['fine_labels'][row]
        coarse_labels = trainingData['coarse_labels'][row]

        
        print(data)
        print(filenames)
        print(fine_labels)
        print(coarse_labels)
        print('<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>')

    for batch in dataset:
        loss, grad = value_and_grad_fn(model, batch)
        optimizer.update(model, grad)
        # Evaluate the loss and the new parameters which will
        # run the full gradient computation and optimizer update
        mx.eval(loss, model.parameters())

    '''
    for e in range(num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
        accuracy = eval_fn(model, test_images, test_labels)
        toc = time.perf_counter()
        print(
            f"Epoch {e}: Test accuracy {accuracy.item():.3f},"
            f" Time {toc - tic:.3f} (s)"
        )
    '''
main()

