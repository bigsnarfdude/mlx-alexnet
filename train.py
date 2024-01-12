import pickle
from alex_net import AlexNet
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


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
    
main()
