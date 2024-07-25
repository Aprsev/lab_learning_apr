from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.graph_objects as go

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

    # 筛选指定unit_id的数据
    df_filtered = df[df['unit_id'].isin(unit_ids)]

    # 创建热点图数据
    heatmap_data = df_filtered.pivot_table(index='height', columns='width', values='A', fill_value=0)

    # 将0值映射为白色
    heatmap_data[heatmap_data == 0] = None

    # 创建Plotly热点图
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        zmin=0, zmax=1,  # 设置颜色映射的范围
        showscale=False
    ))

    fig.update_layout(
        title='热点图',
        xaxis_title='Width',
        yaxis_title='Height',
        width=600,
        height=400
    )

    # 将图表转换为JSON格式以在网页上显示
    plot_json = fig.to_json()
    return jsonify(plot_json=plot_json)

if __name__ == '__main__':
    app.run(debug=True)