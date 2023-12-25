import numpy as np
from typing import Callable

def rk4(dy: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray, t0: float, tn: float, h: float) -> np.ndarray:
    """
    使用四阶龙格-库塔法（RK4）进行数值积分

    Args:
    - dy: 由导数组成的 np.array([y1_t,y2_t,...,yn_t])
    - y0: 初始条件 y(t0)，类型为 NumPy 数组（一维）
    - t0: 初始时间
    - tn: 结束时间
    - h: 步长

    Return:
    数值积分结果的 NumPy 数组，包含从 t0 到 tn 的时间点上的 y 值

    Example:
    ```python
    def dy(t: float, y: np.ndarray) -> np.ndarray:
        y1, y2 = y
        y1_t = y2
        y2_t = -y2-(2+np.sin(t))*y1
        return np.array([y1_t,y2_t])
    y0 = np.array([0.0, 1.0])  # 初始条件
    t0 = 0.0  # 初始时间
    tn = 10.0  # 结束时间
    h = 0.1  # 步长
    res = rk4(dy, y0, t0, tn, h)
    # 对应的时间
    t = np.linspace(t0,tn,len(res))
    ```
    """
    num_steps = int((tn - t0) / h)  # 计算步数
    t = np.linspace(t0, tn, num_steps + 1)  # 生成时间点数组
    y = np.zeros((num_steps + 1, len(y0)))  # 初始化结果数组
    y[0] = y0  # 设置初始条件

    for i in range(num_steps):
        k1 = h * dy(t[i], y[i])
        k2 = h * dy(t[i] + h/2, y[i] + k1/2)
        k3 = h * dy(t[i] + h/2, y[i] + k2/2)
        k4 = h * dy(t[i] + h, y[i] + k3)

        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def dy(t: float, y: np.ndarray) -> np.ndarray:
            y1, y2 = y
            y1_t = y2
            y2_t = -y2-(2+np.sin(t))*y1
            return np.array([y1_t,y2_t])
    y0 = np.array([0.0, 1.0])  # 初始条件
    t0 = 0.0  # 初始时间
    tn = 10.0  # 结束时间
    h = 0.05  # 步长

    result = rk4(dy, y0, t0, tn, h)
    t = np.linspace(t0,tn,len(result))
    plt.plot(t,result)
    plt.show()