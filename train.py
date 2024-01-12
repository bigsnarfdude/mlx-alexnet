import pickle
from alex_net import AlexNet
import mlx.core as mx
import mlx.nn as nn

def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict



def main():
    seed = 1337 
    num_layers = 2
    hidden_dim = 16
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1

    #np.random.seed(seed)
    trainingData = unpickle('./cifar100/train')

    # Load the data
    #train_images, train_labels, test_images, test_labels = map(mx.array, mnist())

    model = AlexNet()
    print(model)

    # Load the model
    mx.eval(model.parameters())
    #for item in trainingData:
    #    print(item, type(trainingData[item]))
    #print(trainingData['data'][0])

    train_images = map(mx.array, trainingData['data'])
    train_labels = map(mx.array, trainingData['fine_labels'])
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)

    #for e in range(num_epochs):
    #    tic = time.perf_counter()
    #    for X, y in batch_iterate(batch_size, train_images, train_labels):
    #        loss, grads = loss_and_grad_fn(model, X, y)
    #        optimizer.update(model, grads)
    #        mx.eval(model.parameters(), optimizer.state)
    #    accuracy = eval_fn(model, test_images, test_labels)
    #    toc = time.perf_counter()
    #    print(
    #        f"Epoch {e}: Test accuracy {accuracy.item():.3f},"
    #        f" Time {toc - tic:.3f} (s)"
    #    )
    
main()
