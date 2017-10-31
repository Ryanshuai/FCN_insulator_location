import tensorflow as tf
from tfRecords import TFRecords_Reader
import res_net
from PIL import Image
import numpy as np

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = tf_config)

NUM_EXAMPLE = 30000 #number of samples
NUM_EPOCH = 100
BATCH_SIZE = 32
SAVE_STEP = 10000 #10k

tfmodel_load_dir = 'tfModel_resize_0/' #load model from here
tfmodel_save_name = 'tfModel_resize_1/model.ckpt' #save model to here

tfreader = TFRecords_Reader(NUM_EXAMPLE)
index, reader_image, reader_annot_image = tfreader.readbatch_by_queue('train_imAndanot.tfrecords',batch_size=BATCH_SIZE,num_epoch=NUM_EPOCH)

net1 = res_net.Net1()
net2 = res_net.Net2()
net3 = res_net.Net3()

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(tfmodel_load_dir)

if ckpt and ckpt.model_checkpoint_path:
    print('loading_model')
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print('no_pre_model')


sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

train_step = 0
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


while True:
    image, annot_image = sess.run([reader_image, reader_annot_image])
    net1_predict = sess.run(net1.predict,feed_dict={net1.input_im: image})

    ternel = np.array(image[0], np.float32)
    ternel_u8 = np.array(ternel, np.uint8)
    ori = Image.fromarray(ternel_u8)
    plt.subplot(1, 3, 1)
    plt.title('origin image')
    plt.imshow(ori)

    ternel = np.array(annot_image, np.float32)
    ternel_u8 = np.array(ternel, np.uint8)
    a = np.zeros(shape=[300, 400, 3], dtype=np.uint8)
    a[:, :, 0] = ternel_u8[0][:, :, 0]
    a[:, :, 1] = ternel_u8[0][:, :, 0]
    a[:, :, 2] = ternel_u8[0][:, :, 0]
    img = Image.fromarray(a)
    plt.subplot(1, 3, 2)
    plt.title('annot')
    plt.imshow(img)

    ternel = np.array(net1_predict, np.float32) * 255.0
    ternel_u8 = np.array(ternel, np.uint8)
    a = np.zeros(shape=[300, 400, 3], dtype=np.uint8)
    a[:, :, 0] = ternel_u8[0][:, :, 0]
    a[:, :, 1] = ternel_u8[0][:, :, 0]
    a[:, :, 2] = ternel_u8[0][:, :, 0]
    img = Image.fromarray(a)
    plt.subplot(1, 3, 2)
    plt.title('predict')
    plt.imshow(img)

    a = input()