from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

# 显示每一层的结构,tensor输入


def print_activations(t):
    # t.op.name显示名称，t.get_shape()显示大小
    print(t.op.name, ' ', t.get_shape().as_list())

# 函数inference接受images作为输入，返回最后一层pool5(第五个池化层)及parameters,包括了很多卷积和池化层


def inference(images):
    parameters = []

    ###第一个卷积层###
    # 可将scope内生成的Variable自动命名为conv1/xxx,便于区分不同卷积层之间的组件
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],  # 使用tf.truncated_normal截断的正态分布函数（标准差为0.1）初始化卷积层的kernel
                                                 dtype=tf.float32, stddev=1e-1), name='weights')  # 卷积核尺寸为11*11，颜色通道为3，卷积核数量为64
        # 对输入images完成卷积操作，步长设为4*4，每次取样卷积核大小都为11*11
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(initial_value=tf.constant(
            0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')  # biases初始化为 0
        bias = tf.nn.bias_add(conv, biases)  # 将conv和biases加起来
        conv1 = tf.nn.relu(bias, name=scope)  # 激活函数RELU对结果进行非线性处理
        print_activations(conv1)  # 打印最后一层输出的tensor conv1的结构
        parameters += [kernel, biases]  # 将这一层可训练的参数kernel、biases添加到parameters中

    ###在第一个卷积层后添加LRN层和最大池化层###
    lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001 / 9,
                     beta=0.75, name='lrn1')  # （可选）“相邻神经元抑制”使用LRN会让前馈、反馈的速度大大下降
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],  # 池化尺寸3*3，即将3*3的大小像素降为1*1
                           padding='VALID', name='pool1')  # padding设定为VALID，取样时不能超过边框，设定为SAME可以填充边界外的点
    print_activations(pool1)

    ###第二个卷积层###
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(initial_value=tf.truncated_normal([5, 5, 64, 192],  # 卷积核尺寸5*5，输入通道数64（上层卷积核数量），卷积核数量192
                                                               dtype=tf.float32, stddev=1e-1), name='weights')
        # 卷积步长设置为1*1，即扫描全图像素。
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(initial_value=tf.constant(
            0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)

    ###在第二个卷积层后添加LRN层和最大池化层###
    lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9,
                     beta=0.75, name='lrn2')  # （可选）“相邻神经元抑制”使用LRN会让前馈、反馈的速度大大下降
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],  # 池化尺寸3*3，即将3*3的大小像素降为1*1
                           padding='VALID', name='pool2')  # padding设定为VALID，取样时不能超过边框，设定为SAME可以填充边界外的点
    print_activations(pool2)

    ###第三个卷积层###
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(initial_value=tf.truncated_normal([3, 3, 192, 384],  # 卷积核尺寸3*3，输入通道数192（上层卷积核数量），卷积核数量384
                                                               dtype=tf.float32, stddev=1e-1), name='weights')
        # 卷积步长设置为1*1，即扫描全图像素。
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(initial_value=tf.constant(
            0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    ###第四个卷积层###
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(initial_value=tf.truncated_normal([3, 3, 384, 256],  # 卷积核尺寸3*3，输入通道数384，卷积核数量下降到256
                                                               dtype=tf.float32, stddev=1e-1), name='weights')
        # 卷积步长设置为1*1，即扫描全图像素。
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(initial_value=tf.constant(
            0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    ###第五个卷积层###
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(initial_value=tf.truncated_normal([3, 3, 256, 256],  # 卷积核尺寸3*3，输入通道数256，卷积核数量也是256
                                                               dtype=tf.float32, stddev=1e-1), name='weights')
        # 卷积步长设置为1*1，即扫描全图像素。
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(initial_value=tf.constant(
            0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    ###五个卷积之后加一个最大池化层###
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[
                           1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool5)

    return pool5, parameters

###评估AlexNet每轮计算时间的函数###


# (TensorFlow的Session, 需要评测的运算算子, 测试的名称)
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10  # 给程序热身，头几轮迭代有显存加载、cache命中等问题因此可以跳过，只考量10轮之后的计算时间
    total_duration = 0.0
    total_duration_squared = 0.0  # 用以计算方差

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()  # 记录时间
        _ = session.run(target)  # 执行迭代
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now, i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    # 计算每轮迭代的平均耗时mn和标准差sd，最后将结果显示出来
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d step, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))

###主函数###


def run_benchmark():
    with tf.Graph().as_default():  # 定义默认Graph
        image_size = 224
        # 并不使用ImageNet数据集来训练，只使用随机图片数据测试前馈和反馈计算的耗时，tf.random_normal 函数构造正态分布的随机tensor
        images = tf.Variable(tf.random_normal([batch_size,  # 每轮迭代的样本数
                                               image_size,  # 图片尺寸
                                               image_size, 3],  # 颜色通道数
                                              dtype=tf.float32,
                                              stddev=1e-1))
        # # 使用inference函数构建整个AlexNet网络，得到最后一个池化层的输出pool5和网络中需要训练的参数集合parameters
        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()  # 初始化所有参数
        sess = tf.Session()  # 创建新的Session
        sess.run(init)

        # 使用time_tensorflow_run统计运算时间，传入的target就是pool5，即卷积网络最后一个池化层的输出
        time_tensorflow_run(sess, pool5, "Forward")

        # 与forward计算有些不同，需要给最后输出的pool5设置一个优化目标tf.nn.l2_loss计算pool5的loss
        objective = tf.nn.l2_loss(pool5)
        # 用tf.gradients求相对于loss的所有模型参数的梯度，这样就模拟了一个训练过程
        grad = tf.gradients(objective, parameters)
        # 使用time_tensorflow_run统计backward的运算时间，target就是求整个网络梯度gard的操作
        time_tensorflow_run(sess, grad, "Forward-backward")


###执行主函数###
run_benchmark()
