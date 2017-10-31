import tensorflow as tf


class Net1():
    def __init__(self):
        self.collection = ['net1', tf.GraphKeys.GLOBAL_VARIABLES]

        xavier_init_conv2d = tf.contrib.layers.xavier_initializer_conv2d()

        self.input_im = tf.placeholder(shape=[None, 300, 400, 3], dtype=tf.float32)#shape=(bs,h,w,d)

        ## conv1 layer ## (bs,300,400,3)->(bs,300,400,64)->(bs,100,100,64)
        self.W_conv1 = tf.Variable(xavier_init_conv2d([11, 11, 3, 64]), collections=self.collection)
        self.b_conv1 = tf.Variable(tf.constant(0., shape=[64]), collections=self.collection)
        self.conv1 = tf.nn.conv2d(self.input_im/255., self.W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        self.h_conv1 = tf.nn.relu(self.conv1 + self.b_conv1)
        self.h_pool1 = tf.nn.max_pool(self.h_conv1, ksize=[1, 4, 5, 1], strides=[1, 3, 4, 1], padding='SAME')

        ## conv2 layer  ## (bs,100,100,64)->(bs,100,100,32)->(bs,50,50,32)
        self.W_conv2 = tf.Variable(xavier_init_conv2d([5, 5, 64, 32]), collections=self.collection)
        self.b_conv2 = tf.Variable(tf.constant(0., shape=[32]), collections=self.collection)
        self.conv2 = tf.nn.conv2d(self.h_pool1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        self.h_conv2 = tf.nn.relu(self.conv2 + self.b_conv2)
        self.h_pool2 = tf.nn.max_pool(self.h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        ## conv3 layer ## (bs,50,50,32)->(bs,50,50,1)->(bs,25,25,1)
        self.W_conv3 = tf.Variable(xavier_init_conv2d([3, 3, 32, 1]),collections=self.collection)
        self.b_conv3 = tf.Variable(tf.constant(0., shape=[1]),collections=self.collection)
        self.conv3 = tf.nn.conv2d(self.h_pool2, self.W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        self.h_conv3 = tf.nn.relu(self.conv3 + self.b_conv3)
        self.h_pool3 = tf.nn.max_pool(self.h_conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.predict = tf.image.resize_images(self.h_pool3,[300,400]) #shape=(bs,25,25,1)->(bs,300,400,1)

        self.target = tf.placeholder(shape=[None, 300, 400, 1], dtype=tf.float32)#shape=(bs,w,h,d)

        self.error = tf.square(self.predict - self.target/255.)  #(bs,300,400,1)
        self.loss = tf.reduce_mean(tf.reduce_mean(self.error, (1, 2, 3)))  # (1,)
        tf.summary.scalar('net1_loss', self.loss)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0000001)
        self.optimize = self.trainer.minimize(self.loss)
        self.merged_summary = tf.summary.merge_all()


class Net2():
    def __init__(self):
        self.collection = ['net2', tf.GraphKeys.GLOBAL_VARIABLES]

        xavier_init_conv2d = tf.contrib.layers.xavier_initializer_conv2d()

        self.input_im = tf.placeholder(shape=[None, 300, 400, 3], dtype=tf.float32)#shape=(bs,h,w,d)

        ## conv1 layer ## (bs,300,400,3)->(bs,300,400,64)
        self.W_conv1 = tf.Variable(xavier_init_conv2d([11, 11, 3, 64]),collections=self.collection)
        self.b_conv1 = tf.Variable(tf.constant(0., shape=[64]),collections=self.collection)
        self.conv1 = tf.nn.conv2d(self.input_im/255., self.W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        self.h_conv1 = tf.nn.relu(self.conv1 + self.b_conv1)

        ## conv2 layer ## (bs,300,400,64)->(bs,300,400,32)->(bs,100,100,32)
        self.W_conv2 = tf.Variable(xavier_init_conv2d([5, 5, 64, 32]),collections=self.collection)
        self.b_conv2 = tf.Variable(tf.constant(0., shape=[32]),collections=self.collection)
        self.conv2 = tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        self.h_conv2 = tf.nn.relu(self.conv2 + self.b_conv2)
        self.h_pool2 = tf.nn.max_pool(self.h_conv2, ksize=[1, 4, 5, 1], strides=[1, 3, 4, 1], padding='SAME')

        ## conv3 layer ## (bs,100,100,32)->(bs,100,100,1)->(bs,50,50,1)
        self.W_conv3 = tf.Variable(xavier_init_conv2d([3, 3, 32, 1]),collections=self.collection)
        self.b_conv3 = tf.Variable(tf.constant(0., shape=[1]),collections=self.collection)
        self.conv3 = tf.nn.conv2d(self.h_pool2, self.W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        self.h_conv3 = tf.nn.relu(self.conv3 + self.b_conv3)
        self.h_pool3 = tf.nn.max_pool(self.h_conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.predict = tf.image.resize_images(self.h_pool3,[300,400]) #shape=(bs,50,50,1)->(bs,300,400,1)

        self.target = tf.placeholder(shape=[None, 300, 400, 1], dtype=tf.float32)#shape=(bs,w,h,d)

        self.error = tf.square(self.predict - self.target/255.)  #(bs,300,400,3)
        self.loss = tf.reduce_mean(tf.reduce_mean(self.error, (1, 2, 3)))  # (1,)
        tf.summary.scalar('loss', self.loss)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0000001)
        self.optimize = self.trainer.minimize(self.loss)
        self.merged_summary = tf.summary.merge_all()


class Net3():
    def __init__(self):
        self.collection = ['net3', tf.GraphKeys.GLOBAL_VARIABLES]

        xavier_init_conv2d = tf.contrib.layers.xavier_initializer_conv2d()

        self.input_im = tf.placeholder(shape=[None, 300, 400, 3], dtype=tf.float32)#shape=(bs,h,w,d)

        ## conv1 layer ## (bs,300,400,3)->(bs,300,400,64)
        self.W_conv1 = tf.Variable(xavier_init_conv2d([11, 11, 3, 64]),collections=self.collection)
        self.b_conv1 = tf.Variable(tf.constant(0., shape=[64]),collections=self.collection)
        self.conv1 = tf.nn.conv2d(self.input_im/255., self.W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        self.h_conv1 = tf.nn.relu(self.conv1 + self.b_conv1)

        ## conv2 layer ## (bs,300,400,64)->(bs,300,400,32)
        self.W_conv2 = tf.Variable(xavier_init_conv2d([5, 5, 64, 32]),collections=self.collection)
        self.b_conv2 = tf.Variable(tf.constant(0., shape=[32]),collections=self.collection)
        self.conv2 = tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        self.h_conv2 = tf.nn.relu(self.conv2 + self.b_conv2)

        ## conv3 layer ## (bs,300,400,32)->(bs,300,400,1)->(bs,25,25,1)
        self.W_conv3 = tf.Variable(xavier_init_conv2d([3, 3, 32, 1]),collections=self.collection)
        self.b_conv3 = tf.Variable(tf.constant(0., shape=[1]),collections=self.collection)
        self.conv3 = tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        self.h_conv3 = tf.nn.relu(self.conv3 + self.b_conv3)
        self.h_pool3 = tf.nn.max_pool(self.h_conv3, ksize=[1, 4, 5, 1], strides=[1, 3, 4, 1], padding='SAME')

        self.predict = tf.image.resize_images(self.h_pool3,[300,400]) #shape=(bs,25,25,1)->(bs,300,400,1)

        self.target = tf.placeholder(shape=[None, 300, 400, 1], dtype=tf.float32)#shape=(bs,w,h,d)

        self.error = tf.square(self.predict - self.target/255.)  #(bs,300,400,1)
        self.loss = tf.reduce_mean(tf.reduce_mean(self.error,(1,2,3)))  # (1,)
        tf.summary.scalar('net3_loss', self.loss)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0000001)
        self.optimize = self.trainer.minimize(self.loss)
        self.merged_summary = tf.summary.merge_all()