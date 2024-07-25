import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 定义一个函数用于绘制数据
def plot_data():
    # 从 unit_id_entry 输入框获取 unit_id
    unit_id = unit_id_entry.get()
    column=column_entry.get()
    if not unit_id:
        messagebox.showwarning("Warning", "Please enter a unit_id.")
        return

    try:
        # 根据 unit_id 筛选数据
        filtered_df = df[df['unit_id'] == int(unit_id)]  # 假设 unit_id 是整数类型
        if filtered_df.empty:
            messagebox.showwarning("Warning", "No data found for the given unit_id.")
            return

        # 清除旧图形
        ax.clear()
        # 绘制新曲线
        ax.plot(filtered_df.index, filtered_df[column], label=f'unit_id: {unit_id}')
        ax.legend()
        # 重绘画布
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

# 选择 CSV 文件
def select_csv_file():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            file_path_entry.delete(0, tk.END)
            file_path_entry.insert(0, file_path)
            # 可选：自动绘制数据
            # plot_data()  # 注释掉如果你希望用户手动触发绘制
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read the file:\n{e}")

# 创建 Tkinter 主窗口
window = tk.Tk()
window.title("Data Plotter")

# 创建用于存储 CSV 文件路径的 Entry
file_path_label = tk.Label(window, text="File path:")
file_path_label.pack(side='top', fill='x')

file_path_entry = tk.Entry(window, width=50)
file_path_entry.pack(side='top', fill='x')

# 创建用于选择 CSV 文件的按钮
load_button = tk.Button(window, text="Load CSV", command=select_csv_file)
load_button.pack(side='top', fill='x')

# 创建用于存储 column 的输入框
column_label = tk.Label(window, text="Enter column:")
column_label.pack(side='top', fill='x')

column_entry = tk.Entry(window, width=10)
column_entry.pack(side='top', fill='x')

# 创建用于存储 unit_id 的输入框
unit_id_label = tk.Label(window, text="Enter unit_id:")
unit_id_label.pack(side='top', fill='x')

unit_id_entry = tk.Entry(window, width=10)
unit_id_entry.pack(side='top', fill='x')

# 创建用于绘制数据的按钮
plot_button = tk.Button(window, text="Plot", command=plot_data)
plot_button.pack(side='top', fill='x')

# 创建绘图所需的 matplotlib 图形和轴对象
fig, ax = plt.subplots(figsize=(10, 6))

# 创建 FigureCanvasTkAgg 并添加到主窗口
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

# 运行 Tkinter 事件循环
window.mainloop()