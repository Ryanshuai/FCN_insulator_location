import tensorflow as tf
import tfRecords as TFRecords_Reader

img_dir = '../img_rotation/'
annot_image_dir = '../edge_rotation/'
tfrecords_name = 'train_imAndanot.tfrecords'

NUM_EXAMPLE = 9600 #number of samples
tfreader = TFRecords_Reader(NUM_EXAMPLE)
tfreader.write_records(img_dir,annot_image_dir,tfrecords_name)