## 1 Introduction
Physical Information Neural Network implemented using pytorch, Mainly to facilitate the solution of ordinary differential equations.

## 2 External dependencies for pypinn
torch, numpy, and tqdm.

## 3 Usage

### 3.1 step 1 
Define a class that inherits from `pypinn.Pinn`, such as `Net`. Write the equation to be solved in the `get_f` function of `Net`. Here is an example:
```python
import pypinn
import torch
import torch.nn as nn

class Net(pypinn.Pinn):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, seed=0):
        super().__init__(input_size, hidden_sizes, output_size, seed)

    def get_f(self,x):
        y, dy = self.get_y_and_dy(x)
        f1 = dy[0]-y[1]
        f2 = dy[1]-(-y[1]-(2+torch.sin(x))*y[0])
        f3 = self.forward(torch.tensor([[0.0]])) - torch.tensor([[0,1]])
        return f1,f2,f3
```
The above example is used to solve a system of equations:
$$
\begin{cases}
    \frac{\mathrm{d}y_1}{\mathrm{d}x} = y_2,\\
    \frac{\mathrm{d}y_2}{\mathrm{d}x} = -y_2-(2+\sin x)y_1,\\
    y_1(0)=0,y_2(0)=1.
\end{cases}
$$

Tips: ` dy[0], dy[1] ` represent $\frac{dy_1}{dx}, \frac{dy_2}{dx}$, ` y[0], y[1]` represent $y_1, y_2$. The number of equations can theoretically be arbitrarily large.

### 3.2 step 2
Instantiate the class we defined, for example, set the number of input neurons to 1, the hidden layer to 20*20, and the number of output neurons to 2: `mdl = Net(1,[20,20],2)`.

### 3.3 step 3
Configure the training data, loss function, optimizer, learning rate, number of iterations, and then use `mdl.train(**settings)` to train.
```python
settings={
    'x': torch.linspace(0,6,301,requires_grad=True).view(-1,1),
    'loss_fn': nn.MSELoss(),
    'optimizer': torch.optim.Adam,
    'lr': 1e-3,
    'epochs': 5000
}
mdl.train(**settings)
```

### 3.4 step 4
After the model is trained, it can be visualized. Here is a simple example:
```python
import matplotlib.pyplot as plt
t = torch.linspace(0,6,500).view(-1,1)
plt.plot(t,mdl.predict(t))
plt.show()
```



## Update log
`1.0.1` More detailed instructions have been added
`1.0.0` first updata