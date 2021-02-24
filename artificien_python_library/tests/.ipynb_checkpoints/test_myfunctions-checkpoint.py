from artificienlib import syftfunctions
from artificienlib import constants

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

model = Net()

def test_mse():
    logits = torch.tensor((), dtype=torch.float64)
    logits.new_ones((2, 3))
    target = torch.tensor((), dtype=torch.float64)
    target.new_zeros((2,3))
    assert syftfunctions.mse_with_logits() == (logits - targets).sum() / batch_size

