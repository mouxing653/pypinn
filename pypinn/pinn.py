import torch
import torch.nn as nn
from torch import Tensor
from typing import Union
from numpy import ndarray
from tqdm import tqdm

'''搭建神经网络'''
class Pinn(nn.Module):
    """
        物理信息网络(PINN).

        Args:
        - input_size: int, Size of the input feature.
        - hidden_sizes: list, Dimensions of the hidden layers.
        - output_size: int, Size of the output feature.
        - seed: int, Random seed, default 0.

        Funs:
        - forward: 获取网络的输出
        - get_y_and_dy: 获取网络的输出及其导数
        - get_f: 需要求解的方程，右端保持为 0，该部分需要根据具体方程进行重写
        - predict: 用于网络的预测

        Examples:
        - `mdl = pinn.PINN(1,[20,20],3)`
    """

    def __init__(self, input_size:int, hidden_sizes:list, output_size:int,seed=0):
        super(Pinn, self).__init__()
        torch.manual_seed(seed)
        # 网络搭建部分
        self.layers = nn.ModuleList() # 使用ModuleList来存储网络层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0])) # 添加输入层到第一个隐藏层的全连接层
        self.layers.append(nn.Tanh())
        for i in range(1, len(hidden_sizes)): # 添加多个隐藏层和激活函数
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size)) # 添加输出层的全连接层

        self.initialize_weights() # 初始化权重

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=5.0/3)


    '''方程部分'''
    def get_y_and_dy(self,x:Tensor):
        '''
        获取函数值及其导数值，y[0] 对应数学含义上的 y1，dy[0] 对应数学含义上的 y1_x.

        Args:
        - x: tensor, 输入数据，需要求导，二维 tensor.

        Returns:
        - y: lsit, y[0] 对应数学含义上的 y1，y 中的每个元素都是 tensor.
        - dy: list, dy[0] 对应数学含义上的 y1_x，dy 中的每个元素都是 tensor.

        Eaxmple:
        - `y, dy = self.get_y_and_dy(x)`
        '''
        y = self.forward(x)
        n = y.shape[1]
        ones = torch.ones_like(y[:,0].view(-1,1))
        dy = []
        for i in range(n):
            yi_x = torch.autograd.grad(y[:,i].view(-1,1),x,ones,create_graph=True)[0]
            dy.append(yi_x)
        return (y.T).unsqueeze_(2), dy
    
    
    def get_f(self,x:Tensor):
        '''
        描述需要求解的方程

        ```markdown
        y1_x = y1, 
        y2_x = cos(x), 
        y1(0) = 1, 
        y2(0) = 0. 
        ```

        Example:
        ```python
        def get_f(self,x):
            y, dy = self.get_y_and_dy(x)
            f1 = dy[0]-y[0]
            f2 = dy[1]-torch.cos(x)
            f3 = self.forward(torch.tensor([[0.0]])) - torch.tensor([[1,0.0]])
            return f1,f2,f3
        ```

        这里 dy[1] 是函数 y2 对 自变量 x 的导数，同理，dy[7] 代表 函数 y8 对 自变量 x 的导数. 如需求解偏微分，需要自行实现偏导数和相应的方程(需要返回对应的 f1, f2...). 
        '''
        pass

    def train(self,x:Tensor,loss_fn,optimizer,lr=1e-3,epochs=5):
        '''
        训练函数，用于网络参数的更新

        Args:
        - x: tensor, 输入数据，需要求导，二维tensor.
        - loss_fn: 损失函数，如 `nn.MSELoss()`.
        - optimizer: 优化器，如 `torch.optim.Adam`.
        - lr: float, 学习率，如 1e-3，会传入 optimizer.
        - epochs: int, 训练次数.

        Return:
        - loss.item(): float, 经过 epochs 次训练后得到的损失函数值.

        Example:
        ```python
        settings={
            'x': torch.linspace(0,1,101,requires_grad=True).view(-1,1),
            'loss_fn': nn.MSELoss(),
            'optimizer': torch.optim.Adam,
            'lr': 1e-3,
            'epochs': 2000
        }
        mdl.train(**settings)
        ```
        '''

        opt = optimizer(self.parameters(),lr=lr)
        for i in tqdm(range(epochs)):
            f = self.get_f(x)
            loss = 0
            for fi in f:
                loss = loss + loss_fn(fi,torch.zeros_like(fi))
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()

    def predict(self, x: Union[ndarray,Tensor]) -> Tensor:
        with torch.no_grad():
            try:
                if torch.is_tensor(x) is not True:
                    x = torch.from_numpy(x).to(torch.float32)
                return self.forward(x)
            except:
                print("接收二维 ndarry 或 tensor")
