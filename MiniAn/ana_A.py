import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, simpledialog, filedialog
from matplotlib.colors import ListedColormap
import numpy as np

# 定义一个颜色映射表，A值不为0时使用的颜色
cmap = ListedColormap(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow'])

# 指定支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建Tk窗口对象并隐藏
root = Tk()
root.withdraw()

# 弹出窗口选择CSV文件
file_path = filedialog.askopenfilename(title='选择CSV文件', filetypes=[("CSV files", "*.csv")])
if not file_path:
    print("未选择文件，程序将退出。")
    exit()

# 读取CSV文件
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"读取CSV文件时出错：{e}")
    exit()

# 弹出窗口输入unit_id
unit_ids_input = simpledialog.askstring("输入", "请输入unit_id，用逗号分隔:", parent=root)
if not unit_ids_input:
    print("未输入unit_id，程序将退出。")
    exit()
unit_ids = [int(uid.strip()) for uid in unit_ids_input.split(',')]

# 筛选数据并绘制热点图
fig, ax = plt.subplots()
data = df[df['unit_id'].isin(unit_ids)].pivot_table(index='height', columns='width', values='A')

# 将A值为0的部分设置为白色
data = data.fillna(0)
data[data == 0] = np.nan  # 将A值为0的部分设置为NaN，以便在热点图中不显示

print(data)

# 使用imshow绘制热点图
cax = ax.imshow(data, cmap=cmap, interpolation='nearest')

# 设置图表标题和坐标轴标签
ax.set_title('热点图')
ax.set_xlabel('宽度')
ax.set_ylabel('高度')

# 添加颜色条
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('A值')

# 显示图表，使用block=True确保不阻塞
plt.show(block=True)

# 弹出窗口显示图表
simpledialog.showinfo("完成", "热点图已绘制完成，请查看。")