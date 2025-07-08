import pickle
import numpy as np
import matplotlib.pyplot as plt

# 替换 'your_file.pkl' 为你的 PKL 文件路径
file_path = 'gnn_flattening_fnc_2018.pkl'

with open(file_path, 'rb') as file:
    interp_func = pickle.load(file)

# 查看插值函数的属性
print("x values:", interp_func.x)
print("y values:", interp_func.y)

# 使用插值函数在特定点进行插值
x_new = np.linspace(min(interp_func.x), max(interp_func.x), 100)
y_new = interp_func(x_new)

# 绘制插值结果
plt.plot(interp_func.x, interp_func.y, 'o', label='Original data')
plt.plot(x_new, y_new, '-', label='Interpolated function')
plt.legend()
plt.savefig("ddd.png")
