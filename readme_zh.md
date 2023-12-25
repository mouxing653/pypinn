## 1 简介
物理信息神经网络的 pytorch 实现，主要是为了方便求解常微分方程组。

## 2 pypinn 依赖的第三方库
torch, numpy, and tqdm.

## 3 用法

### 3.1 step 1 
定义一个继承 pypinn.Pinn 类，如 `Net`. 将要求解的方程写入 `Net` 中的 `get_f` 函数，下面是一个示例：
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
上面的例子用于求解方程组：
$$
\begin{cases}
    \frac{\mathrm{d}y_1}{\mathrm{d}x} = y_2,\\
    \frac{\mathrm{d}y_2}{\mathrm{d}x} = -y_2-(2+\sin x)y_1,\\
    y_1(0)=0,y_2(0)=1.
\end{cases}
$$

其中 `dy[0],dy[1]` 代表 $\frac{dy_1}{dx},\frac{dy_2}{dx}$, `y[0],y[1]` 代表 $y_1,y_2$.

### 3.2 step 2
对我们定义的类进行实例化，比如，设置输入神经元数量为 1,隐藏层为 20*20,输出神经元数量为 2: `mdl = Net(1,[20,20],2)`

### 3.3 step 3
配置训练数据，损失函数，优化器，学习率，迭代次数，然后使用 `mdl.train(**settings)` 进行训练.
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
模型训练好后可以进行可视化，下面是个简单的示例：
```python
import matplotlib.pyplot as plt
t = torch.linspace(0,6,500).view(-1,1)
plt.plot(t,mdl.predict(t))
plt.show()
```



## 更新日志
`1.0.1` More detailed instructions have been added
`1.0.0` first updata