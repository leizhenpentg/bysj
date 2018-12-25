import numpy as np
import tensorflow as tf
import random
import logging
import numpy as np
import argparse
import os
from PIL import Image
from datetime import datetime
import math
import time
from resnet_aaa import *
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pickle
slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', 21097, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 224, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 30000001, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 1000, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 30000, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './train_tfrecord3755/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './test_tfrecord3755/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'validation', 'Running mode. One of {"train", "validation"}')
FLAGS = tf.app.flags.FLAGS

print("-----------------------------main.py start--------------------------")

class DataIterator:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        print(truncate_path)
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names)
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)
        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        min_after_dequeue=1000
        capacity = min_after_dequeue+3*batch_size
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        return image_batch, label_batch

def get_batch(dirpath = 'H:/why_workspace/char_classify/resnet_810/tfrecord/',is_training = True):
    if is_training:
        file_path = dirpath+"train_data.tfrecord-*"
    else:
        file_path = dirpath+"test_data.tfrecord-*"
    files = tf.train.match_filenames_once(file_path)
    filea_queue = tf.train.string_input_producer(files, shuffle=False)
    reader = tf.TFRecordReader()
    _, serailized_example = reader.read(filea_queue)
    feautures = tf.parse_single_example(
        serailized_example,
        features={
            'image':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64)
        }
    )
    images, labels = feautures['image'], feautures['label']
    decoded_image = tf.decode_raw(images, tf.uint8)
    decoded_image = tf.reshape(decoded_image, [64, 64, 1])
    new_decoded_image = tf.image.resize_images(decoded_image, (224, 224))
    min_after_dequeue = 10000
    batch_size = FLAGS.batch_size
    capacity = min_after_dequeue + 3*batch_size
    image_batch, label_batch = tf.train.shuffle_batch([new_decoded_image, labels], min_after_dequeue=min_after_dequeue,
                                                     capacity=capacity, batch_size=batch_size)
    return image_batch, label_batch

def file_name(file_dir, is_train=True):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if is_train == True:
                if os.path.splitext(file)[0] == 'train_data':
                    L.append(os.path.join(root, file))
            else:
                if os.path.splitext(file)[0] == 'test_data':
                    L.append(os.path.join(root, file))
    return L

def get_batch2(dirpath =  'H:/why_workspace/char_classify/resnet_810/tfrecord/',is_train=True):
    filenames = file_name(dirpath, is_train)
    #print(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    def parser(record):
        features = {
                'image': tf.FixedLenFeature([], tf.string, default_value=""),
                'label': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            }
        parsed = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.reshape(image, [64, 64, 1])
        new_image = tf.image.resize_images(image, (224, 224))
        label = tf.cast(parsed["label"], tf.int32)
        return new_image, label
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=5000).batch(FLAGS.batch_size)
    dataset = dataset.repeat(FLAGS.epoch)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch

def build_graph(num_classes=FLAGS.charset_size, top_k=3, is_training=True):
    images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    net, end_points = resnet_v2_50(images, num_classes=num_classes, is_training=is_training)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), labels), tf.float32))
    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(0.001, global_step, decay_steps=8000, decay_rate=0.997, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
    probabilities = tf.nn.softmax(net)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
    return {'images': images,
            'labels': labels,
            'logits': net,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}

def train():
    print('Begin training')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=tf_config)
    with sess:
        train_images, train_labels = get_batch2(dirpath=FLAGS.train_data_dir, is_train=True)
        test_images, test_labels = get_batch2(dirpath=FLAGS.test_data_dir, is_train=False)
        graph = build_graph(num_classes=FLAGS.charset_size, top_k=1)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info(':::Training Start:::')
        #try:
            #while not coord.should_stop():
        while True:
            try:
                #start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])

                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             }
                _, loss_val, train_summary, step, logit, _labels = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step'], graph['logits'],
                     graph['labels']],
                    feed_dict=feed_dict)
                #end_time = time.time()
                train_writer.add_summary(train_summary, step)
                '''//打印图片
                print(train_images_batch.shape)
                for i in range(FLAGS.batch_size):
                    images = train_images_batch[i]
                    #h, w, c = images.shape
                    #assert c == 1
                    images = images.reshape(224, 224)
                    print(images)
                    plt.imshow(images)
                    plt.show()
                '''
                #logger.info("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                if step > FLAGS.max_steps:
                    break
                if step % FLAGS.eval_steps == 1:
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 }
                    accuracy_test, test_summary = sess.run(
                        [graph['accuracy'], graph['merged_summary_op']],
                        feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, step)
                    #logger.info('===============Eval a batch=======================')
                    logger.info('the step {0} test accuracy: {1}'
                                .format(step, accuracy_test))
                    #logger.info('===============Eval a batch=======================')
                if step % FLAGS.save_steps == 1:
                    logger.info('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                               global_step=graph['global_step'])
            except tf.errors.OutOfRangeError:
                logger.info('==================Train Finished================')
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step'])
                break
        # finally:
        #     coord.request_stop()
        # coord.join(threads)

def validation():
    print('validation')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=tf_config)
    with sess:
        test_images, test_labels = get_batch2(dirpath='H:/why_workspace/char_classify/resnet_practice/test_tfrecord/', is_train=False)
        graph = build_graph(num_classes=FLAGS.charset_size, top_k=3)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        saver = tf.train.Saver()
        #test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val_all')
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
        logger.info(':::Start validation:::')
        while True:
            try:
                i = 0
                acc_top_1, acc_top_k = 0.0, 0.0
                while not coord.should_stop():
                    i += 1
                    #start_time = time.time()
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 }
                    acc_1, acc_k, step, test_summary = sess.run([graph['accuracy'],
                                                                 graph['accuracy_top_k'],
                                                                 graph['global_step'],
                                                                 graph['merged_summary_op']], feed_dict=feed_dict)
                    #test_writer.add_summary(test_summary, step)
                    acc_top_1 += acc_1
                    acc_top_k += acc_k
                    #end_time = time.time()
                    if(i % 100 == 0):
                        logger.info("the batch {0} takes x seconds, accuracy = {1}(top_1) {2}(top_k)"
                                    .format(i, acc_1, acc_k))
            except tf.errors.OutOfRangeError:
                logger.info('==================Validation Finished================')
                #acc_top_1 = acc_top_1 * FLAGS.batch_size / 2200.0
                #acc_top_k = acc_top_k * FLAGS.batch_size / 2200.0
                acc_top_1 = acc_top_1 / (i-1)
                acc_top_k = acc_top_k / (i-1)
                logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))
                print(i)
                break


def inference(image):
    print('inference')
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 224, 224, 1])
    with tf.Session() as sess:
        logger.info('========start inference============')
        # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        # Pass a shadow label 0. This label will not affect the computation graph.
        graph = build_graph(top_k=3)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image})
    return predict_val, predict_index

def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "validation":
        validation()
    elif FLAGS.mode == 'inference':
        image_path = 'H:/why_workspace/char_data/test2/00009/00009_0815.png'
        final_predict_val, final_predict_index = inference(image_path)
        logger.info('the result info label {0} predict index {1} predict_val {2}'.format(190, final_predict_index,
                                                                                         final_predict_val))

if __name__ == "__main__":
    tf.app.run()
