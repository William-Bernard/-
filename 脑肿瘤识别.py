#!/usr/bin/env python
# coding: utf-8

# ## 导入必要的库

# In[ ]:


# 系统库
import os 
import itertools 
from PIL import Image
# 数据处理库
import cv2 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix,classification_report 
# 深度学习库 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam ,Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras import regularizers
import warnings 
warnings.filterwarnings('ignore')


# ## 数据处理

# In[ ]:


train_data_dir = r'C:\Users\Administrator\OneDrive\Desktop\脑肿瘤识别\脑肿瘤识别\brain-tumor-mri-dataset\Training'
filepaths = []#列表用于存储图像文件的路径。
labels = []#列表用于存储对应图像的标签

folds = os.listdir(train_data_dir)
for fold in folds:
    foldpath = os.path.join(train_data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        #通过两层循环（外层循环遍历每个子文件夹，内层循环遍历子文件夹中的所有文件），构建图像的完整路径，并将其添加到 `filepaths` 列表中。
        filepaths.append(fpath)
        labels.append(fold)
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')#使用 `pd.Series` 创建两个Pandas序列，分别包含图像路径和对应的标签。
data = pd.concat([Fseries, Lseries], axis= 1)#输出包含了所有图像的路径和对应的标签。
data 


# In[ ]:


test_data_dir = r"C:\Users\Administrator\OneDrive\Desktop\脑肿瘤识别\脑肿瘤识别\brain-tumor-mri-dataset\Training"
filepaths = []
labels = []
folds = os.listdir(test_data_dir)

for fold in folds:
    foldpath = os.path.join(test_data_dir,fold)
    filelist = os.listdir(foldpath) 
    for file in filelist:
        fpath = os.path.join(foldpath,file)
        filepaths.append (fpath)
        labels.append(fold)
Fseries = pd.Series(filepaths,name = 'filepaths')
Lseries = pd.Series(labels, name = 'labels')
dt = pd.concat([Fseries,Lseries],axis='columns')
dt 


# ## 划分数据集为验证集和测试集

# In[ ]:


valid_df,test_df = train_test_split(dt,train_size=0.5,shuffle=True,random_state=123)  # 按照1：1的比例划分验证集和测试集


# ## 创建图片数据生成器

# In[ ]:


batch_size=2#每次生成器提供2张图像的数据。
img_size = (224,224)#设置图像的尺寸为224x224像素
channels = 3 #设置为3，表示图像是RGB三通道的。
img_shape=(img_size[0],img_size[1],channels)  #224*224*3
tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()#创建ImageDataGenerator实例
train_gen = tr_gen.flow_from_dataframe( data, x_col= 'filepaths', y_col= 'labels',
                                         target_size= img_size, class_mode= 'categorical', color_mode= 'rgb', shuffle= True,batch_size= batch_size)
test_gen = tr_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels',
                                         target_size= img_size, class_mode= 'categorical', color_mode= 'rgb', shuffle= False ,batch_size= batch_size)
valid_gen = tr_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels',
                                         target_size= img_size, class_mode= 'categorical', color_mode= 'rgb', shuffle= True,batch_size= batch_size)


# In[ ]:


g_dict = train_gen.class_indices#获取类别索引
classes = list(g_dict.keys())
images,labels = next(train_gen)
plt.figure(figsize = (20,20))#创建一个大小为20x20英寸的图形
for i in range(2):
    plt.subplot(4,4,i+1)
    image = images[i]/255
    plt.imshow(image)
    index = np.argmax(labels[i])
    class_name= classes[index]
    plt.title(class_name,color='blue',fontsize=12)
    plt.axis('off')
plt.show()#显示图像


# ## 模型结构

# In[50]:


img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

class_count = len(list(train_gen.class_indices.keys()))#class_count` 通过 `train_gen.class_indices` 获取类别索引的字典，并计算类别总数。

model = Sequential([
    Conv2D(filters = 64 , kernel_size = (3,3) , padding = "same" , activation = 'relu' ,input_shape=img_shape),
    Conv2D(filters = 64 , kernel_size = (3,3) , padding = "same" , activation = 'relu'),
    MaxPooling2D(2,2),#最大池化层           #使用same填充，保持输出尺寸不变
    Conv2D(filters = 128 , kernel_size = (3,3) , padding = "same" , activation = 'relu'),
    Conv2D(filters = 128 , kernel_size = (3,3) , padding = "same" , activation = 'relu'),
    MaxPooling2D(2,2),
    Conv2D(filters = 256 , kernel_size = (3,3) , padding = "same" , activation = 'relu'),
    Conv2D(filters = 256 , kernel_size = (3,3) , padding = "same" , activation = 'relu'),
    Conv2D(filters = 256 , kernel_size = (3,3) , padding = "same" , activation = 'relu'),
    MaxPooling2D(2,2),
    Conv2D(filters = 512 , kernel_size = (3,3) , padding = "same" , activation = 'relu'),
    Conv2D(filters = 512 , kernel_size = (3,3) , padding = "same" , activation = 'relu'),
    Conv2D(filters = 512 , kernel_size = (3,3) , padding = "same" , activation = 'relu'),
    MaxPooling2D(2,2),
    Flatten(),#多维特征图展平为一维特征图
    

    Dense(256, activation='relu',input_dim=30),
    Dense(64,activation = 'relu'),
    Dense(class_count, activation = 'softmax')
])
# 在卷积层之后，模型添加了三个全连接层（`Dense`）。
   # 第一个全连接层有256个神经元，使用 'relu' 激活函数。
   # 第二个全连接层有64个神经元，也使用 'relu' 激活函数。
   # 最后一个全连接层的神经元数量等于类别数量 `class_count`，使用 'softmax' 激活函数进行多类分类。

model.compile(Adamax(learning_rate=0.001), loss = 'categorical_crossentropy',metrics = ['accuracy'])
#使用 `Adamax` 优化器编译模型，设置学习率为0.001
model.summary()#打印模型详细结构


# In[ ]:


epochs = 5

history = model.fit(x=train_gen,epochs= epochs, validation_data=valid_gen,shuffle= False)


# In[ ]:


train_acc = history.history['accuracy']#`train_acc` 和 `train_loss` 分别存储了训练过程中的准确率和损失值。
train_loss = history.history['loss']#`val_acc` 和 `val_loss` 分别存储了验证过程中的准确率和损失值。

val_acc = history.history['val_accuracy']#
val_loss = history.history['val_loss']#

index_loss = np.argmin(val_loss)#使用 `np.argmin` 找到 `val_loss` 中的最小值索引，表示验证损失最低的Epoch。
val_lowest = val_loss[index_loss]

index_acc = np.argmax(val_acc)# - 使用 `np.argmax` 找到 `val_acc` 中的最大值索引，表示验证准确率最高的Epoch。
val_highest = val_acc[index_acc]

Epochs = [i+1 for i in range(len(train_acc))]#`创建Epochs` 列表，包含了从1到训练周期总数的整数。

loss_label = f'Best epochs = {str(index_loss +1)}'#设置标签
acc_label = f'Best epochs = {str(index_acc + 1)}'

plt.figure(figsize= (20,8))#创建图形和子图
plt.style.use('fivethirtyeight')

plt.subplot(1,2,1)#绘制损失曲线，创建1行2列的子图布局，并在第一个子图中绘制
plt.plot(Epochs , train_loss , 'r' , label = 'Training Loss')
plt.plot(Epochs , val_loss , 'g' , label = 'Validation Loss')
plt.scatter(index_loss + 1 , val_lowest , s = 150 , c = 'blue',label = loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(Epochs , train_acc , 'r' , label = 'Training Accuracy')
plt.plot(Epochs , val_acc , 'g' , label = 'Validation Accuracy')
plt.scatter(index_acc + 1 , val_highest , s = 150 , c = 'blue',label = acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()#添加图例
plt.tight_layout#调整子图布局防止标签重叠
plt.show()#显示最终图形



# ## 评估模型

# In[ ]:


train_score = model.evaluate(train_gen ,  verbose = 1)#训练集数据生成器
valid_score = model.evaluate(valid_gen ,  verbose = 1)#验证集数据生成器
test_score = model.evaluate(test_gen ,  verbose = 1)#测试集数据生成器


# In[ ]:


preds = model.predict_generator(test_gen)

y_pred = np.argmax(preds , axis = 1)
y_pred#preds 是模型通过 predict_generator 方法对测试数据生成器 test_gen 进行预测的结果。
#predict_generator 是 Keras 框架中用于处理数据生成器的函数，它允许模型在内存不足时也能进行预测。y_pred 就是模型预测的结果


# In[ ]:


g_dict = test_gen.class_indices#生成混淆矩阵并获取类别名称
classes = list(g_dict.keys())
cm = confusion_matrix(test_gen.classes, y_pred)#使用 `confusion_matrix` 函数从 `sklearn.metrics` 模块创建混淆矩阵
cm #存储矩阵


# In[ ]:


plt.figure(figsize= (10, 10))#创建一个新的图形，并设置其大小为 10x10 英寸。
plt.imshow(cm, interpolation= 'nearest', cmap= plt.cm.Blues)#将混淆矩阵 `cm` 作为图像显示出来，使用 'nearest' 插值方法和 'Blues' 颜色图。
plt.title('Confusion Matrix')#设置图形的标题为 "Confusion Matrix"。
plt.colorbar()#在图形旁边添加一个颜色条，以表示不同数值的大小。

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation= 45)#设置 x 轴的刻度标签为类别名称并旋转 45 度。
plt.yticks(tick_marks, classes)#设置 y 轴的刻度标签为类别名称。


thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')
#通过循环 `itertools.product(range(cm.shape[0]), range(cm.shape[1]))` 遍历混淆矩阵的每个元素，
#并使用 `plt.text` 在每个位置上添加文本标签，文本颜色根据数值与阈值的比较结果设置为 'white' 或 'black'。
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[ ]:


print(classification_report(test_gen.classes,y_pred,target_names=classes))
#使用 `classification_report` 函数从 `sklearn.metrics` 模块打印出一个详细的分类报告，其中包含了主要的分类性能指标。


# ## 保存模型

# In[ ]:


model.save('Brain.h5')


# In[ ]:


#定义类别标签，将之前获取的类别名称列表赋值给 `class_labels`。
class_labels = classes

predictions = model.predict_generator(test_gen)#使用模型对测试数据生成器 `test_gen` 中的数据进行预测。

y_pred = np.argmax(preds , axis = 1)#从模型的预测结果中，获取每个样本预测概率最高的类别索引。


# predicted_labels = tf.argmax(predictions, axis=1).numpy()
predicted_labels = np.argmax(preds , axis = 1)

# 获取真实结果
true_labels = test_gen.classes#从测试数据生成器中获取实际的类别标签。


# 展示预测结果和真实结果
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, (image, true_label, predicted_label) in enumerate(zip(test_gen[0][0], true_labels, predicted_labels)):
    axes[i // 4, i % 4].imshow(image/255)
    axes[i // 4, i % 4].set_title(f'Predicted: {class_labels[predicted_label]}\nActual: {class_labels[true_label]}',fontsize = 8)
    axes[i // 4, i % 4].axis('off')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




