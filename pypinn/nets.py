import torch
import torch.nn  as nn
from tqdm import tqdm
from typing import Union
from numpy import ndarray

class Ann(nn.Module):
    '''
    用于有监督学习的全连接神经网络

    Args:
    - input_size: int, 输入特征的数量
    - hidden_sizes: list, 隐藏层的尺寸
    - output_size: int, 输出特征的数量
    - seed: int, 随机种子，默认 0

    Example:
    `mdl = Ann(1,[20,20],3)`
    '''
    def __init__(self, input_size, hidden_sizes, output_size, seed=0):
        super(Ann, self).__init__()
        torch.manual_seed(seed)
        # 使用ModuleList来存储网络层
        self.layers = nn.ModuleList()
        # 添加输入层到第一个隐藏层的全连接层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.Tanh())
        # 添加多个隐藏层和激活函数
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.Tanh())
        # 添加输出层的全连接层
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        # 初始化权重
        self.initialize_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=5.0/3)
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
    

    def train(self,input:torch.Tensor,target:torch.Tensor,loss_fn,optimizer,lr:float,epochs:int=5) -> float:
        '''
        模型的训练

        Args:
        - input: Tensor, 输入数据
        - target: Tensor, 数据的标签，模型的期望输出
        - loss_fn: 损失函数, 如 nn.MSEloss()
        - optimizer: 优化器, 如 torch.optim.Adam
        - lr: float, 学习率, 会被传入 optimizer
        - epochs: int, 训练次数

        Return:
        - loss:float, 模型经过 epochs 次训练后得到的损失函数值

        Example:
        ```python
        x = torch.linspace(0,7,100).view(-1,1)
        settings = {
            'input': x,
            'target':torch.sin(x),
            'loss_fn': nn.MSELoss(),
            'optimizer': torch.optim.Adam,
            'lr': 1e-3,
            'epochs': 1000
        }
        # 开始训练
        mdl.train(**settings)
        ```
        '''
        opt = optimizer(self.parameters(),lr=lr)
        for i in tqdm(range(epochs)):
            out = self.forward(input)
            loss = loss_fn(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()

    # 用于推理
    def predict(self, x: Union[ndarray,torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            try:
                if torch.is_tensor(x) is not True:
                    x = torch.from_numpy(x).to(torch.float32)
                return self.forward(x)
            except:
                print("接收二维 ndarry 或 tensor")

