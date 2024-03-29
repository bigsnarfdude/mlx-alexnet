import mlx.core as mx
import mlx.nn as nn
from max_pool import MaxPool2d


class AlexNet(nn.Module):

  def __init__(self, classes=10, dropout=0.5):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
      nn.ReLU(),
      MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False),
      nn.ReLU(),
      MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=False),
      nn.ReLU(),
      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
      nn.ReLU(),
      MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(p=dropout),
      nn.Linear(256 * 1 * 1, 4096),
      nn.ReLU(),
      nn.Dropout(p=dropout),
      nn.Linear(4096, 4096),
      nn.ReLU(),
      nn.Linear(4096, classes),
    )

  def __call__(self, x): # mx.array
    x = self.features(x)
    x = x.reshape(x.shape[0], -1)
    x = self.classifier(x)
    return x


def alexnet(**kwargs):
  """
  AlexNet model from paper
  `"One weird trick for parallelizing convolutional neural networks" <https://arxiv.org/abs/1404.5997>`_ paper.
  """
  model = AlexNet(**kwargs)
  return model
