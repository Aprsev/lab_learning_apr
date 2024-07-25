
#加载数据集

import mnist 
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf

def preprocess_image(image): # 缩放图像 
    image = tf.image.resize(image, [224, 224]) # 归一化图像 
    image = tf.image.perimage_whitening(image) 
    return image 

def build_deconvolutional_neural_network(input_shape): # 定义卷积层 
    conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape) # 定义反卷积层 
    deconv_layer = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu') # 定义全连接层 
    fc_layer = tf.keras.layers.Dense(units=10, activation='softmax') # 构建模型 
    model = tf.keras.Sequential([conv_layer, deconv_layer, fc_layer]) 
    return model 
def train_deconvolutional_neural_network(model, input_data, labels, epochs=10, batch_size=32): # 编译模型 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # 训练模型 
    model.fit(input_data, labels, epochs=epochs, batch_size=batch_size) 
    return model 

def process_image(image, model): # 预处理图像 
    image = preprocess_image(image) # 使用模型进行图像处理 
    processed_image = model.predict(image) 
    return processed_image 

#构建反卷积神经网络
(inputdata, labels) = mnist.loaddata()

model = build_deconvolutional_neural_network(inputshape=(28, 28, 1))
#训练反卷积神经网络

train_deconvolutional_neural_network(model, inputdata, labels, epochs=10, batch_size=32)
#使用反卷积神经网络进行图像处理

image = inputdata[0] 
processed_image = process_image(image, model)
#显示原始图像和处理后的图像


plt.subplot(1, 2, 1) 
plt.imshow(image, cmap='gray') 
plt.title('Original Image') 
plt.subplot(1, 2, 2) 
plt.imshow(processed_image, cmap='gray') 
plt.title('Processed Image') 
plt.show() 