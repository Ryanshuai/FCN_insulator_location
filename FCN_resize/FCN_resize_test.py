import tensorflow as tf
import numpy as np
from PIL import Image

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = tf_config)

BATCH_SIZE = 1

img_dir = 'E:/projecting/img/'
premodel_dir = 'H:/desktop/tfModel_linear_0/'

alexnet_npy_path = 'bvlc_alexnet.npy'
alexnet_data_dict = np.load(alexnet_npy_path, encoding="latin1").item()

def computLoss(im_predic,annot_im):
    cross_loss_1 = annot_im * tf.log(tf.clip_by_value(im_predic, 1e-10 ,1.0))
    loss_1 = tf.reduce_mean( tf.reduce_sum( cross_loss_1, [1,2]) )
    cross_loss_0 = (1.0 - annot_im) * tf.log(tf.clip_by_value(1.0-im_predic, 1e-10, 1.0))
    loss_0 = tf.reduce_mean( tf.reduce_sum( cross_loss_0, [1,2]) )
    loss = -(loss_0 + loss_1 ) / 2
    return loss

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel,group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

im_origin = tf.placeholder(tf.float32, shape=[None, 300,400,3])
annot_im = tf.placeholder(tf.float32, shape=[None, 300,400,1])
im_norm = ( im_origin - 128.0 ) / 255.0
annot_im_norm = annot_im / 255.0

#conv1:
#300*400*3->75*100*96->37*49*96
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(alexnet_data_dict['conv1'][0])
conv1b = tf.Variable(alexnet_data_dict['conv1'][1])
conv1_in = conv(im_norm, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)
#max_pool1(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv2
#37*49*96->37*49*256->18*24*256
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(alexnet_data_dict["conv2"][0])
conv2b = tf.Variable(alexnet_data_dict["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)
#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(conv2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#18*24*256->18*24*384
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(alexnet_data_dict["conv3"][0])
conv3b = tf.Variable(alexnet_data_dict["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#18*24*384->18*24*384
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(alexnet_data_dict["conv4"][0])
conv4b = tf.Variable(alexnet_data_dict["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

#conv5
#18*24*384->18*24*256->8*11->256
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(alexnet_data_dict["conv5"][0])
conv5b = tf.Variable(alexnet_data_dict["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)
#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

## c6 layer ##
#8*11*256->8*11*4096
W_c6 = tf.Variable(tf.truncated_normal([1,1,256,4096], stddev=0.1))
b_c6 = tf.Variable(tf.constant(0.1, shape=[4096]))
c6 = tf.nn.conv2d(maxpool5, W_c6, strides=[1, 1, 1, 1], padding='SAME')
c6_relu = tf.nn.relu(c6 + b_c6)

## c7 layer ##
#8*11*4096->8*11*4096
W_c7 = tf.Variable(tf.truncated_normal([1,1,4096,4096], stddev=0.1))
b_c7 = tf.Variable(tf.constant(0.1, shape=[4096]))
c7 = tf.nn.conv2d(c6_relu, W_c7, strides=[1, 1, 1, 1], padding='SAME')
c7_relu = tf.nn.relu(c7 + b_c7)

## c8 layer ##
#8*11*4096->8*11*1
W_c8 = tf.Variable(tf.truncated_normal([1,1,4096,1], stddev=0.1))
b_c8 = tf.Variable(tf.constant(0.1, shape=[1]))
c8 = tf.nn.conv2d(c7_relu, W_c8, strides=[1, 1, 1, 1], padding='SAME')
c8_relu = tf.nn.relu(c8 + b_c8)

######deconvolution1 : ... * 2 = x2############
#deconv1:c8 10*13*1->19*25*1
c8_2x = tf.image.resize_images(c8_relu,[18, 24])

#reduce dim: conv5 19*25*256->19*25*1
W_5 = tf.Variable(tf.truncated_normal([1,1,256,1], stddev=0.1))
h_5 = tf.nn.conv2d(conv5,W_5,strides=[1, 1, 1, 1], padding='SAME')
x2 = c8_2x + h_5

######deconvolution2 : x2 * 2 = x4############
#deconv1:x2 19*25*1->38*50*1
x2_2 = tf.image.resize_images(x2,[37, 49])
#reduce dim: conv2 38*50*256->38*50*1
W_2 = tf.Variable(tf.truncated_normal([1,1,256,1], stddev=0.1))
h_2 = tf.nn.conv2d(conv2,W_2, strides=[1, 1, 1, 1], padding='SAME')
x4 = x2_2 + h_2

######deconvolution3 : x4 * 8 = x32############
#deconv1:x2 38*50*1->300*400*1
x32 = tf.image.resize_images(x4,[300, 400])

x32_norm = tf.nn.sigmoid(x32)
loss_L2 = computLoss(x32_norm,annot_im_norm)

#train_step = tf.train.AdadeltaOptimizer(learning_rate=0.2, rho=0.95, epsilon=1e-08).minimize(loss_L2)
train_step = tf.train.AdadeltaOptimizer(0.5).minimize(loss_L2)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

ckpt = tf.train.get_checkpoint_state(premodel_dir)
if ckpt and ckpt.model_checkpoint_path:
    print('loading_model')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

img_path = img_dir + str(6089) + '.jpg'
img = Image.open(img_path,'r')
img = img.resize((400, 300))
img_array = np.array(img, np.uint8)
img_array = img_array.reshape(1,300,400,3)

ternel = sess.run(x32_norm,feed_dict={im_origin: img_array})
print(type(ternel))
ternel = np.array(ternel, np.float32) * 255.0
ternel_u8 = np.array(ternel, np.uint8)
a = np.zeros(shape=[300,400,3], dtype=np.uint8)
a[:,:,0] = ternel_u8[0][:,:,0]
a[:,:,1] = ternel_u8[0][:,:,0]
a[:,:,2] = ternel_u8[0][:,:,0]

img = Image.fromarray(a)
img.show()
sess.close()
