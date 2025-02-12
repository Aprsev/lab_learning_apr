from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.graph_objects as go
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/heatmap', methods=['POST'])
def heatmap():
    # 获取上传的CSV文件和unit_ids
    file = request.files['file']
    unit_ids = request.form.get('unit_ids', default='', type=str)
    unit_ids = [int(uid) for uid in unit_ids.split(',')] if unit_ids else []

    # 读取CSV文件
    df = pd.read_csv(file)

    # 筛选数据并绘制热点图
    data = df[df['unit_id'].isin(unit_ids)].pivot_table(index='height', columns='width', values='A')

    # 将A值为0的部分设置为白色
    data = data.fillna(0)
    data[data == 0] = np.nan  # 将A值为0的部分设置为NaN，以便在热点图中不显示

    print(data)
    
    # 创建Plotly热点图
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='Viridis',
        zmin=0, zmax=1,  # 设置颜色映射的范围
        showscale=False
    ))

    fig.update_layout(
        title='Heatmap',
        xaxis_title='Width',
        yaxis_title='Height',
        width=1200,
        height=800
    )
    fig.show()
    # 将图表转换为JSON格式以在网页上显示
    plot_json = fig.to_json()
    return jsonify(plot_json=plot_json)

@app.route('/heatmap_all', methods=['POST'])
def heatmap_all():
    # 获取上传的CSV文件
    file = request.files['file_all']

    # 读取CSV文件
    df = pd.read_csv(file)

    # 绘制包含所有unit_id的热点图
    data = df.pivot_table(index='height', columns='width', values='A')

    # 将A值为0的部分设置为白色
    data = data.fillna(0)
    data[data == 0] = np.nan  # 将A值为0的部分设置为NaN，以便在热点图中不显示

    print(data)
    
    # 创建Plotly热点图
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='Viridis',
        zmin=0, zmax=1,  # 设置颜色映射的范围
        showscale=False
    ))

    fig.update_layout(
        title='Heatmap for All Units',
        xaxis_title='Width',
        yaxis_title='Height',
        width=1200,
        height=800
    )
    fig.show()
    # 将图表转换为JSON格式以在网页上显示
    plot_json = fig.to_json()
    return jsonify(plot_json=plot_json)

if __name__ == '__main__':
    app.run(debug=True)