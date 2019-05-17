from __future__ import absolute_import, division, print_function

# pylint: disable=unused-import
import gzip
import os
import tempfile
import sys

import numpy as np
import sklearn as sk
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_input = mnist.train.images.shape[1]
n_hidden_1 = 64
n_hidden_2 = 64
# n_hidden_3 = 250
n_classes = 10


def mlp_config(n_input, n_hidden_1, n_hidden_2, n_classes):
    x = tf.placeholder("float", [None, n_input], name='x')
    y = tf.placeholder("float", [None, n_classes], name='y')
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out_h': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    return x, y, weights


def mlp_model(x, y, weights):

    l1 = tf.add(tf.matmul(tf.cast(x,tf.uint8), tf.cast(weights['h1'],tf.uint8)), tf.cast(weights['b1'], tf.uint8))
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, tf.cast(weights['h2'],tf.uint8)), tf.cast(weights['b2'], tf.uint8))
    l2 = tf.nn.relu(l2)
    #l3 = tf.add(tf.matmul(l2, tf.cast(weights['h3'],tf.float16)), tf.cast(weights['b3'], tf.float16))
    #l3 = tf.nn.relu(l3)
    logits = tf.add(tf.matmul(l2,tf.cast(weights['out_h'],tf.uint8)), tf.cast(weights['out'],tf.uint8))
    # hidden_1 = tf.nn.relu(tf.matmul(x, weights['h1']))
    # hidden_2 = tf.nn.relu(tf.matmul(hidden_1, weights['h2']))
    # hidden_3 = tf.nn.relu(tf.matmul(hidden_2, weights['h3']))
    # logits = tf.matmul(hidden_3, weights['out'])
    pred = tf.one_hot(tf.cast(tf.argmax(logits, 1), tf.int32), depth=10)
    return pred, logits


def get_loss(logits, y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
    return loss


def get_accuracy(pred, y):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.uint8))
    return accuracy


def main():
    x, y, weights = mlp_config(n_input, n_hidden_1, n_hidden_2, n_classes)
    pred, logits  = mlp_model(x, y, weights)
    loss          = get_loss(logits, y)
    accuracy      = get_accuracy(pred, y)

    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    batch_size = 50
    
    epoch = 100
    final_acc = []
    final_los = []
    for i in range(1,epoch+1):
        print ('i am in epoch')
        #print (type(mnist.train.num_examples))
        #print (mnist.train.num_examples)
        # print (int(mnist.train.num_examples / batch_size))
        avg_acc = []
        avg_los = []
        for batch_no in range(int(mnist.train.num_examples / batch_size)):
            # print ('batch number',batch_no)
            batch = mnist.train.next_batch(batch_size)
            # print ('i am in batch')
            if (batch_no % 100) == 0:
                acc = sess.run(accuracy, feed_dict={x:batch[0], y:batch[1]})
                print('test accuracy at step %s: %s' %(i,acc))
                los = sess.run(loss, feed_dict={x:batch[0], y:batch[1]})
                print('loss at step %s: %s' %(i,los))
                #print (accuracy)
                #print (los)
                avg_acc.append(acc)
                avg_los.append(los)
            else:
                sess.run(train_step, feed_dict={x:batch[0], y:batch[1]})
        final_acc.append(sum(avg_acc)/len(avg_acc))
        final_los.append(sum(avg_los)/len(avg_los))
        #final_acc.append
        print("##########")
        print("Accuracy")
        # print(acc)
        # print(sess.run(accuracy, feed_dict={x:batch[0], y:batch[1]}))
        # print(pred.eval())
    # print(sess.run(pred, feed_dict={x:batch[0], y:batch[1], weights:weights}))
    # assert False
    # print ("Prediction values: ",pred)    
    # indexes = tf.arg_max(pred, axis=1) 
    # print("predicted values", indexes)    
    y_p = tf.argmax(pred, 1)
    # print ("Argmax Values: ",y_p)
    
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:mnist.test.images, y:mnist.test.labels})

#     print("Predicted Values: ", y_pred)
    print("size of predicted values", y_pred.shape)

    print("Accuracy using tensorflow is: ")
    print(val_accuracy)
    
    print("shape of test data ", mnist.test.labels.shape)
    print("some of samples", mnist.test.labels[5,:])
#  changing the one hot 
    y_true = mnist.test.labels[:,:]

    y_true = tf.argmax(y_true,1)
    print("y true", sess.run(y_true[1:]))
    true_value = sess.run(y_true[:]).reshape(10000,1)
#     confusion matrix
    cm = tf.confusion_matrix( y_true, y_pred, num_classes=10,dtype=tf.int32,name=None,weights=None)
    test_confusion = sess.run(cm)
    print(test_confusion)
    
#     print("diag", np.diag(test_confusion))
    
#     TP = sess.run(tf.diag_part(test_confusion))\
    TP = np.diag(test_confusion)
    print("TP", TP)

    
    FP = np.sum(test_confusion, axis=0) - TP
    print("FP", FP)
    FN = np.sum(test_confusion, axis=1) - TP
    print("FN", FN)
#     TN = []
#     num_classes =10
#     for i in range(num_classes):
#         temp = np.delete(test_confusion, i, 0)    # delete ith row
#         temp = np.delete(temp, i, 1)  # delete ith column
#         TN.append(sum(sum(temp)))
    

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print("prescision", precision)
    print("recall", recall)
    plt.plot(precision, recall)
    plt.step(recall, precision, color='b', alpha=0.2)
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision vs recall')
    plt.show()
    
    return final_acc, final_los

final_acc, final_los = main()

print (final_acc)
print (final_los)

x = final_acc
y = final_los
epoch_list = []
epoch = 100

for i  in range (1,epoch+1):
    epoch_list.append(i)

print (epoch_list)

plt.plot(epoch_list, x)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.title('Accuracy per epoch')


plt.show()

plt.plot(epoch_list, y)

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.title('Loss per epoch')
#fig.savefig("loss.png")

plt.show()