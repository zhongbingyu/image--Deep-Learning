#先引入必要的库
#import time #输出模型构造耗费的时间需要time库
import os   #对文件或文件夹操作需要os库
import numpy as np #数组运算使用numpy库
from PIL import Image  #常用图像处理函数库,现在它优化成了Pillow库 Pillow 和opencv选择一个即可
'''
v1版本中tensorflow中有很多函数，在v2版本集成到其他模块中去了。
在学习tensorflow时经常遇到因版本不对应而导致的找不到函数的问题
因此用以下语句来导入tensorflow，解决版本的兼容性问题
'''
import tensorflow.compat.v1 as tf #引入tensorflow库
tf.disable_v2_behavior()

#定义三个常量
# VGGNet自带的常量，写在VGGNet的code中，归一化做图像处理时需要的RGB通道的均值
VGG_MEAN = [103.939, 116.779, 123.68]

# 搭建VGGNet的网络结构,并载入已经训练好的参数
#首先定义一个类
class VGGNet:
    # 定义初始化函数，参数：data_dict——定义好VGGNet模型的变量名称
    def __init__(self, data_dict):
        self.data_dict = data_dict

    #写一个帮助我们把卷积参数从data_dict里读取出来的函数
    # 抽取卷积层的卷积核参数，在卷积层的第一个位置
    def get_conv_filter(self, name):#name可能是’conv1_1‘、'conv1_2'等等
        #返回参数为常量constant，因为模型是训练好的，解析的时候我们发现卷积层参数的第一个位置是卷积核，第二个位置是偏置
        #训练好的参数，定义为常量类型
        return tf.constant(self.data_dict[name][0], name='conv')

    # 抽取全连接层的权重参数，在全连接层的第一个位置
    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name='fc')

    # 抽取偏置参数，在卷积层/全连接层的第二个位置
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='bias')

    # 创建卷积层，它有2个参数：第一个参数x——输入（四维矩阵），name——网络层的名字
    def conv_layer(self, x, name):
        # name_scope：1.命名规范，防止命名冲突，2.使得结果图的显示更加规整
        with tf.name_scope(name):
            # 获得w和b的参数
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)
            # 获得参数之后，就可以进行卷积计算了
            # #将上面获取的参数传递，计算卷积层，进行卷积操作
            h = tf.nn.conv2d(x, conv_w, [1, 1, 1, 1], padding='SAME')
            h = tf.nn.bias_add(h, conv_b)
            h = tf.nn.relu(h)  # 激活函数
            return h

    # 创建池化层，参数：x——输入（四维矩阵），name——网络层的名字
    def pooling_layer(self, x, name):
        # 最大池化操作，不需要预设参数
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name)

    # 创建全连接层，参数：x——输入（四维矩阵），name——网络层的名字
    def fc_layer(self, x, name, activation=tf.nn.relu):
        with tf.name_scope(name):
            # 获得w和b的参数
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            # 将上面获取的参数传递，计算全连接层
            h = tf.matmul(x, fc_w)
            h = tf.nn.bias_add(h, fc_b)
            # 最后一个全连接层是不需要ReLu激活函数的，所以在此加一个参数选择
            if activation is None:
                return h
            else:
                return activation(h)

    # 卷积层的结果展平输入到全连接层
    def flatten_layer(self, x, name):
        with tf.name_scope(name):
            # 获得x在reshape后的size参数：4维向量[batch_size, image_width, image_height, channel]
            x_shape = x.get_shape().as_list()
            # 展平时，后三个参数相乘,因此定义连乘
            dim = 1
            for d in x_shape[1:]:
                dim *= d
            # 对x进行reshape，这里-1是batch_size的位置
            x = tf.reshape(x, [-1, dim])
            return x

    # 构建VGG16网络模型结构，输入参数是一张图像x_rgb： [1, 224, 224, 3]
    def build(self, x_rgb):
        # 记录开始时间
        #start_time = time.time()
        # 预告模型构建的开始
        #print('building model ...')

        # 归一化操作，对于图像的RGB三个通道，切分在三个通道上，分别减去其均值，之后在第四个通道——即维度上用concat函数将其再合并
        r, g, b = tf.split(x_rgb, [1, 1, 1], axis=3)
        # 由于VGGnet它的图像定义b,g,r的一个形式，因此我们这里需要转换一下
        x_bgr = tf.concat([b - VGG_MEAN[0],g - VGG_MEAN[1],r - VGG_MEAN[2]],axis=3)
        # 判断图像大小是224*224*3通道的，如果图像格式不正确，则此模型不能运行
        #assert x_bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # 将图像输入到网络中
        # 调用定义好的函数构建模型——这里模型的名字必须是跟之前一一对应的
        # 1——2个卷积层+一个池化层
        self.conv1_1 = self.conv_layer(x_bgr, 'conv1_1')  # conv1_1名字必须与VGG16的名字一一对应
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')  # 每一层的输入是前一层的输出
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')  # pool层名字命名可以随意，因为不需要从VGG16中获取参数

        # 2——2个卷积层+一个池化层
        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')

        # 3——3个卷积层+一个池化层
        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')

        # 4——3个卷积层+一个池化层
        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')

        # 5——3个卷积层+一个池化层
        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')

        '''
        #全连接层的参数载入时间较长，但是在做图像转换的算法的时候我们用到的是卷积神经网络的值，不会用到全连接层的值，
        #因此可以将这部分注释掉

        #输入全连接层之前做展平操作
        self.flatten5 = self.flatten_layer(self.pool5, 'flatten')
        #3个全连接层
        self.fc6 = self.fc_layer(self.flatten5, 'fc6')
        self.fc7 = self.fc_layer(self.fc6, 'fc7')
            #最后一个全连接层不需要ReLu激活函数
        self.fc8 = self.fc_layer(self.fc7, 'fc8', activation=None)
        #Softmax激活函数
        self.prob = tf.nn.softmax(self.fc8, name='prob')
        '''
        # 输出模型构建的时长
        #print('building model finished: %4ds' % (time.time() - start_time))

#VGGNet、风格图片、内容图片、结果的路径
vgg16_npy_path='D:/1/资料准备/vgg16.npy'
content_img_path='D:1/资料准备/content.jpg'
style_img_path='D:1/资料准备/style.jpg'
output_dir='D:1/资料准备/result'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 读取内容图片,并打印图片尺寸
content_val_in = Image.open(content_img_path)
#print(content_val_in.size)
# 对图像缩放到224*224
content_val = content_val_in.resize((224, 224))
#print(content_val.size)
#将图像转化为数组格式
content_val = np.array(content_val)
# 对图像扩展维数使输入变成四维
content_val = content_val.reshape(1, 224, 224, 3)

# 读取风格图片
style_val_in = Image.open(style_img_path)
# 对图像缩放到224*224
style_val = style_val_in.resize((224, 224))
#将图像转化为数组格式
style_val = np.array(style_val)
# 对图像扩展维数使输入变成四维
style_val = style_val.reshape(1, 224, 224, 3)

'''
Tensorflow的设计理念称之为计算流图
# 在编写程序时，首先构筑整个系统的框架，代码并不会直接生效，然后，在实际的运行时，启动一个session，程序才会真正的运行。
# 我们在构建网络的时候需要将风格图像和内容图像在框架中的位置先占好
# placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
'''
content = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
style = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])

# 定义随机化的初始图片，来作为初始的结果，在这张图片上不断地进行提取下降的操作，得到我们想要的效果
# 正态分布初始化，参数：shape——大小，mean——均值，stddev——方差
def initial_result(shape, mean, stddev):
    # truncated_normal——一个正态分布函数
    initial = tf.compat.v1.random.truncated_normal(shape, mean=mean, stddev=stddev)
    # 返回这个正态分布函数生成的变量
    return tf.Variable(initial)

# 调用它生成结果图(127.5是255的一半，即图像像素范围的中间值)
result = initial_result((1, 224, 224, 3), 127.5, 20)


# 接下来就开始调用我们之前搭建的VGGNET网络来进行训练
# 将风格图像、内容图像、结果图像分别输入到VGGNET中去提取特征和训练

# 1.定义模型的参数
# 加载预训练好的VGG16模型作为训练参数（括号里的参数是它报错提醒我应该设置这些默认参数）
data_dict = np.load(vgg16_npy_path, encoding="latin1", allow_pickle=True).item()
num_steps = 100  # 训练100步
learning_rate = 10  # 每一步的学习率，对参数的学习程度
# 损失函数的系数，内容和风格的损失函数要用两个系数来加权（Gramme函数的哪两个系数lambda）
lambda_c = 0.1  # content的损失函数前面的系数
lambda_s = 500  # style的损失函数前面的系数
# 2.对风格图像、内容图像、结果图像分别创建三个VGGNET模型（上面定义的三个class）
vgg_for_content = VGGNet(data_dict)
vgg_for_style = VGGNet(data_dict)
vgg_for_result = VGGNet(data_dict)
# 对三个class进行调用（build函数是调用的函数）
vgg_for_content.build(content)
vgg_for_style.build(style)
vgg_for_result.build(result)

# 3.特征提取：不同的层都可以用来做特征提取，我们可以自行选择其中一层
# feature_size, [1, width, height, channel]
# content的特征越低层越好
content_features = [
    vgg_for_content.conv1_2,
    # vgg_for_content.conv2_2,
    # vgg_for_content.conv3_3,
    # vgg_for_content.conv4_3,
    # vgg_for_content.conv5_3
]
# style的特征越高层越好
style_features = [
    # vgg_for_style.conv1_2,
    # vgg_for_style.conv2_2,
    # vgg_for_style.conv3_3,
    vgg_for_style.conv4_3,
    # vgg_for_style.conv5_3
]
# 结果图像要分别学习内容和风格的特征
# 结果图像的内容特征应该跟内容图像的特征提取保持一致
result_content_features = [
    vgg_for_result.conv1_2,
    # vgg_for_result.conv2_2,
    # vgg_for_result.conv3_3,
    # vgg_for_result.conv4_3,
    # vgg_for_result.conv5_3
]
# 结果图像的风格特征应该跟风格图像的特征提取保持一致
result_style_features = [
    # vgg_for_result.conv1_2,
    # vgg_for_result.conv2_2,
    # vgg_for_result.conv3_3,
    vgg_for_result.conv4_3,
    # vgg_for_result.conv5_3
]

# 4.计算损失函数
# 计算内容损失函数
content_loss = tf.zeros(1, tf.float32)  # 首先赋值为0，之后叠加每一层，迭代损失函数的值
# zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
# zip: [1, 2], [3, 4], zip([1,2], [3,4]) -> [(1, 3), (2, 4)]
# 也就是说将内容特征和结果特征一个一个对应在一起，然后求其平方差的平均，计算损失函数
# 每一层特征的shape: [1, width, height, channel]要对width, height, channel都分别求其平均，因此下面的括号里参数加上[1, 2, 3]
for c, c_ in zip(content_features, result_content_features):
    content_loss += tf.reduce_mean((c - c_) ** 2, [1, 2, 3])

# 定义gram矩阵——下面计算style_loss中需要调用
# 输入参数x: 从某一层VGGNET中提取出来的特征，shape: [1, width, height, ch]
def gram_matrix(x):
    # 获取x各维度的值
    b, w, h, ch = x.get_shape().as_list()
    # 将x中的width, height合成一个维度
    features = tf.reshape(x, [b, h * w, ch])  # [ch, ch] -> (i, j)
    # [h*w, ch] matrix 计算这个矩阵的相似度
    # 线性代数的知识->转置矩阵[ch, h*w] * 原矩阵[h*w, ch] -> 得到我们想要的相似度矩阵[ch, ch]
    # matmul()函数即进行矩阵乘法，参数adjoint_a=True即将第一个features矩阵进行转置
    # 为了防止最后的矩阵值过大，进行归一化操作，除以矩阵维度的乘积
    gram = tf.matmul(features, features, adjoint_a=True) / tf.constant(ch * w * h, tf.float32)
    return gram

# 给风格特征计算gram矩阵
style_gram = [gram_matrix(feature) for feature in style_features]
# 给结果图像的风格特征计算gram矩阵
result_style_gram = [gram_matrix(feature) for feature in result_style_features]

# 计算风格损失函数
# 风格特征是计算每个channel的相互关联性，每个channel都是width*height的矩阵——得到的是gram矩阵
style_loss = tf.zeros(1, tf.float32)
for s, s_ in zip(style_gram, result_style_gram):
    # 在gram矩阵的基础上计算损失函数
    style_loss += tf.reduce_mean((s - s_) ** 2, [1, 2])

# 最终的损失函数是内容损失和风格损失的加权
loss = content_loss * lambda_c + style_loss * lambda_s
# 给损失函数计算梯度
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 5.进行风格转换，计算每一步的损失函数，并保存结果
# 初始化梯度
init_op = tf.global_variables_initializer()
# 打开一个session
with tf.Session() as sess:
    # 调用run来初始化变量，对其进行梯度下降
    sess.run(init_op)
    # 循环进行训练（100次）
    for step in range(num_steps):
        # 每一次训练都需要调用损失函数的值，把参数传递进来
        loss_value, content_loss_value, style_loss_value, _ = sess.run([loss, content_loss, style_loss, train_op],
                                                                       feed_dict={
                                                                           content: content_val,
                                                                           style: style_val,
                                                                       })
        # 每一步输出一下
        print('step: %d' % (step + 1))
        '''
      
        # 打印中间每一步的损失函数
        print('step: %d, loss_value: %8.4f, content_loss: %8.4f, style_loss: %8.4f' % (step + 1,
                                                                                       loss_value[0],
                                                                                       content_loss_value[0],
                                                                                       style_loss_value[0]))
        '''

        # 结果图像存储在定义好的文件夹中
        # 定义结果图像的存储路径和名称
        result_img_path = os.path.join(output_dir, 'result-%05d.jpg' % (step + 1))

        # 取出结果变量的值
        result_val = result.eval(sess)[0]
        # 裁剪结果图像，将它的像素值设置在0-255之间
        result_val = np.clip(result_val, 0, 255)
        # 结果图像的数据类型转成uint8
        img_arr = np.asarray(result_val, np.uint8)
        # 用PIL库中的这个函数，将数组对象转换成图像对象
        img = Image.fromarray(img_arr)
        # 保存图片
        img.save(result_img_path)





