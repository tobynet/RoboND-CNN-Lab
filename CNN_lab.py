
# coding: utf-8

# # Image Classification
# In this lab, you'll classify images from the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist#get-the-data).  The dataset consists of different types of clothing items such as shirts, trousers, sneakers etc. You'll preprocess the images, then train a convolutional neural network on all the samples. The images need to be normalized and the labels need to be one-hot encoded.  You'll get to apply what you learned and build a model with convolutional, max pooling, dropout, and fully connected layers.  At the end, you'll get to see your neural network's predictions on the sample images.
# ## Get the Data
# We have provided you with a pickle file for the dataset available in the GitHub repo. We have provided with a script - helper.py, which extracts the dataset for you when the corresponding functions are called.

# ## Explore the Data
# The Fashion-MNIST dataset consists of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from the following 10 classes:
# 
# * T-shirt/top
# * Trouser
# * Pullover
# * Dress
# * Coat
# * Sandal
# * Shirt
# * Sneaker
# * Bag
# * Ankle boot
# 
# Understanding a dataset is part of making predictions on the data.  Play around with the code cell below by changing the `sample_id`. The `sample_id` is the id for a image and label pair in the dataset.
# 
# Ask yourself "What are all possible labels?", "What is the range of values for the image data?", "Are the labels in order or random?".  Answers to questions like these will help you preprocess the data and end up with better predictions.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import helper
import numpy as np

import pickle

filename = "fashion-mnist.p"

# Explore the dataset
sample_id = 1232
helper.display_stats(filename, sample_id)


# ## Implement Preprocess Functions
# ### Normalize
# In the cell below, implement the `normalize` function to take in image data, `x`, and return it as a normalized Numpy array. The values should be in the range of 0 to 1, inclusive.  The return object should be the same shape as `x`.

# In[3]:


import problem_unittests as tests
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (28, 28, 1)
    : return: Numpy array of normalize data
    """
    # DONE: Implement Function
    a = 0.0; b = 1.0
    min_of_x = np.min(x)
    max_of_x = np.max(x)
    return a + (x-min_of_x) * (b-a) / (max_of_x-min_of_x)        

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_normalize(normalize)


# ### One-hot encode
# Just like the previous code cell, you'll be implementing a function for preprocessing.  This time, you'll implement the `one_hot_encode` function. The input, `x`, are a list of labels.  Implement the function to return the list of labels as One-Hot encoded Numpy array.  The possible values for labels are 0 to 9. The one-hot encoding function should return the same encoding for each value between each call to `one_hot_encode`.  Make sure to save the map of encodings outside the function.
# 
# Hint: Don't reinvent the wheel. You have multiple ways to attempt this: Numpy, TF, or even sklearn's preprocessing package.

# In[4]:


from sklearn.preprocessing import LabelBinarizer

one_hot_encoder = LabelBinarizer()
one_hot_encoder.fit(range(len(helper._load_label_names())))
    
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # DONE: Implement Function
    lables = one_hot_encoder.transform(x).astype(np.float32)
    return lables


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)


# ### Randomize Data
# As you saw from exploring the data above, the order of the samples are randomized.  It doesn't hurt to randomize it again, but you don't need to for this dataset.

# ## Preprocess all the data and save it
# Running the code cell below will preprocess all the Fashion-MNIST data and save it to file. The code below also uses 10% of the training data for validation.

# In[6]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(filename, normalize, one_hot_encode)


# # Check Point
# This is your first checkpoint.  If you ever decide to come back to this notebook or have to restart the notebook, you can start from here.  The preprocessed data has been saved to disk.

# In[3]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle
import gzip
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
with gzip.open('preprocess_validation.p.gz', mode='rb') as file:
    valid_features, valid_labels = pickle.load(file)


# ## Build the network
# For the neural network, you'll build each layer into a function.  Most of the code you've seen has been outside of functions. To test your code more thoroughly, we require that you put each layer in a function.  This allows us to give you better feedback and test for simple mistakes using our unittests.
# 
# Let's begin!
# 
# ### Input
# The neural network needs to read the image data, one-hot encoded labels, and dropout keep probability. Implement the following functions
# * Implement `neural_net_image_input`
#  * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
#  * Set the shape using `image_shape` with batch size set to `None`.
#  * Name the TensorFlow placeholder "x" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
# * Implement `neural_net_label_input`
#  * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
#  * Set the shape using `n_classes` with batch size set to `None`.
#  * Name the TensorFlow placeholder "y" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
# * Implement `neural_net_keep_prob_input`
#  * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) for dropout keep probability.
#  * Name the TensorFlow placeholder "keep_prob" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
# 
# These names will be used at the end of the lab to load your saved model.
# 
# Note: `None` for shapes in TensorFlow allow for a dynamic size.

# In[4]:


import tensorflow as tf

# 入力画像用placeholder
def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # DONE: Implement Function
    return tf.placeholder(tf.float32, (None, *image_shape), name='x')

# 出力ラベル用placeholder
def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # DONE: Implement Function
    return tf.placeholder(tf.float32, (None, n_classes), name='y')

# dropoutパラメータ用placeholder
def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # DONE: Implement Function
    return tf.placeholder(tf.float32, name='keep_prob')


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)


# ### Convolution and Max Pooling Layer
# Convolution layers have a lot of success with images. For this code cell, you should implement the function `conv2d_maxpool` to apply convolution then max pooling:
# * Create the weight and bias using `conv_ksize`, `conv_num_outputs` and the shape of `x_tensor`.
# * Apply a convolution to `x_tensor` using weight and `conv_strides`.
#  * We recommend you use same padding, but you're welcome to use any padding.
# * Add bias
# * Add a nonlinear activation to the convolution.
# * Apply Max Pooling using `pool_ksize` and `pool_strides`.
#  * We recommend you use same padding, but you're welcome to use any padding.

# In[5]:


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides, name='conv'):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    with tf.name_scope(name):
        # Create weights and bias
        input_depth = int(x_tensor.shape[3])
        weights = tf.Variable(
            tf.truncated_normal((*conv_ksize, input_depth, conv_num_outputs), stddev=0.1), name="W")
        biases = tf.Variable(tf.constant(0.1, shape=[conv_num_outputs]), name="B")

        # Conv2D layer + bias
        layer = tf.nn.conv2d(x_tensor, weights, strides=(1, *conv_strides, 1), padding='SAME') + biases

        # Nonlinear Activation Layer
        layer = tf.nn.relu(layer)
        
        # Logging
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", layer)
    
        # Max pooling layer
        layer = tf.nn.max_pool(layer, ksize=(1, *pool_ksize, 1),  strides=(1, *pool_strides, 1), padding='SAME')
        return layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_con_pool(conv2d_maxpool)


# ### Flatten Layer
# Implement the `flatten` function to change the dimension of `x_tensor` from a 4-D tensor to a 2-D tensor.  The output should be the shape (*Batch Size*, *Flattened Image Size*). 
# 
# Shortcut Option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer which help with some high-level features. For more of a challenge, only use other TensorFlow packages.

# In[6]:


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # DONE: Implement Function
    return tf.contrib.layers.flatten(x_tensor)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_flatten(flatten)


# ### Fully-Connected Layer
# Implement the `fully_conn` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). 
# 
# Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.

# In[7]:


def fully_conn(x_tensor, num_outputs, name='fully'):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # DONE: Implement Function
    #return tf.contrib.layers.fully_connected(x_tensor, num_outputs)
    
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal([int(x_tensor.shape[-1]), num_outputs], stddev=0.1), name="W")
        biases = tf.Variable(tf.constant(0.1, shape=[num_outputs]), name="B")
        
        layer = x_tensor @ weights + biases
        
        # Logging
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", layer)
        
        return layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_fully_conn(fully_conn)


# ### Output Layer
# Implement the `output` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). 
# 
# Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.
# 
# **Note:** Activation, softmax, or cross entropy should **not** be applied to this.

# In[9]:



def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # DONE: Implement Function
    #return tf.contrib.layers.fully_connected(x_tensor, num_outputs)
    return fully_conn(x_tensor, num_outputs, 'output')


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_output(output)


# ### Create Convolutional Model
# Implement the function `conv_net` to create a convolutional neural network model. The function takes in a batch of images, `x`, and outputs logits.  Use the layers you created above to create this model:
# 
# * Apply 1, 2, or 3 Convolution and Max Pool layers
# * Apply a Flatten Layer
# * Apply 1, 2, or 3 Fully Connected Layers
# * Apply an Output Layer
# * Return the output
# * Apply [TensorFlow's Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) to one or more layers in the model using `keep_prob`. 

# In[10]:


def conv_net(x, keep_prob, use_two_conv=False, use_two_fully=True):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # DONE: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    #conv_k = 3; conv_stride = 2; pool_k = 2
    conv_k = 5; conv_stride = 1; pool_k = 2
    if use_two_conv:
        layer = conv2d_maxpool(x, conv_num_outputs=32,
                           conv_ksize=(conv_k, conv_k), conv_strides=(conv_stride, conv_stride),
                           pool_ksize=(pool_k, pool_k), pool_strides=(pool_k, pool_k),
                           name='conv1')
        layer = conv2d_maxpool(layer, conv_num_outputs=64,
                           conv_ksize=(conv_k, conv_k), conv_strides=(conv_stride, conv_stride),
                           pool_ksize=(pool_k, pool_k), pool_strides=(pool_k, pool_k),
                           name='conv2')
    else:
        layer = conv2d_maxpool(x, conv_num_outputs=16,
                               conv_ksize=(conv_k, conv_k), conv_strides=(conv_stride, conv_stride),
                               pool_ksize=(pool_k, pool_k), pool_strides=(pool_k, pool_k),
                               name='conv')    
    # DONE: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    layer = flatten(layer)
    
    # DONE: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    if use_two_fully:
        layer = fully_conn(layer, num_outputs=1024, name='fully1')
        layer = tf.nn.relu(layer)
        tf.summary.histogram("fully1/relu", layer)
        layer = tf.nn.dropout(layer, keep_prob)
        
    # DONE: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    layer = output(layer, num_outputs=10)
    
    # DONE: return output
    return layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

def build_network(learning_rate, use_two_conv, use_two_fully):
    global x, y, keep_prob, logits, cost, optimizer, accuracy, summ
    
    # Remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # Inputs
    x = neural_net_image_input((28, 28, 1))
    tf.summary.image('input', x, 3)
    y = neural_net_label_input(10)
    keep_prob = neural_net_keep_prob_input()

    # Model
    logits = conv_net(x, keep_prob, use_two_conv, use_two_fully)

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    with tf.name_scope('cost-func'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name='cost')
        tf.summary.scalar('cost', cost)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Accuracy
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

build_network(0.001,False,True)
tests.test_conv_net(conv_net)


# ## Train the Neural Network
# ### Single Optimization
# Implement the function `train_neural_network` to do a single optimization.  The optimization should use `optimizer` to optimize in `session` with a `feed_dict` of the following:
# * `x` for image input
# * `y` for labels
# * `keep_prob` for keep probability for dropout
# 
# This function will be called for each batch, so `tf.global_variables_initializer()` has already been called.
# 
# Hint: You can refer to the "Convolutional Network in TensorFlow" section in the lesson.
# 
# Note: Nothing needs to be returned. This function is only optimizing the neural network.

# In[11]:



def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # DONE: Implement Function
    session.run(optimizer,
               feed_dict={
                   x: feature_batch,
                   y: label_batch,
                   keep_prob: keep_probability
               })
    
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_train_nn(train_neural_network)


# ### Show Stats
# Implement the function `print_stats` to print loss and validation accuracy.  Use the global variables `valid_features` and `valid_labels` to calculate validation accuracy.  Use a keep probability of `1.0` to calculate the loss and validation accuracy.
# 
# Hint: You can refer to the "Convolutional Network in TensorFlow" section in the lesson.

# In[12]:


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # DONE: Calculate loss and validation accuracy
    loss = session.run(cost, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: 1.0
    })
    validation_accuracy = session.run(accuracy, feed_dict={
        x: valid_features,
        y: valid_labels,
        keep_prob: 1.0
    })
    
    # DONE: Print loss and validation accuracy
    print(' Loss: {:>10.4f}    Validation Accuracy: {:.6f}'.format(loss, validation_accuracy))


# ### Hyperparameters
# Tune the following parameters:
# * Set `epochs` to the number of iterations until the network stops learning or start overfitting
# * Set `batch_size` to the highest number that your machine has memory for.  Most people set them to common sizes of memory:
#  * 64
#  * 128
#  * 256
#  * ...
# * Set `keep_probability` to the probability of keeping a node using dropout

# 📝 My Memo:
# 
# 1. First, simple implementation
#   * Use a **single** convolutional layer with max pooling.
#   * Use a **single** connected layer.
#   * Observe the loss, validation accuracy.
# 2. Slowly refine a model
#   * Add more layers, dropouts
#   * Tune Hyperparameters

# In[13]:


# DONE: Tune Parameters
#epochs = 100; batch_size = 64; keep_probability = 1.0 #  0.798500
epochs = 100; batch_size = 100; keep_probability = 1.0 


LOGDIR = 'log/'
log_filename = None
SAVE_MODEL = 'save_model/'

def param_string(learning_rate, use_two_conv, use_two_fully):
    conv_num = {False:1, True:2}[use_two_conv]
    fully_num = {False:1, True:2}[use_two_fully]
    return 'lrate={}_conv={}_fully={}'.format(learning_rate, conv_num, fully_num)


# ### Train the Model
# Now that you have your model built and your hyperparameters defined, let's train it!

# In[14]:


get_ipython().run_cell_magic('time', '', 'import os.path\n"""\nDON\'T MODIFY ANYTHING IN THIS CELL\n"""\ndef train(learning_rate, use_two_conv, use_two_fully, log_filename):\n    build_network(learning_rate, use_two_conv, use_two_fully)\n    \n    #with tf.Session(config=tf.ConfigProto(device_count = {\'GPU\': 0})) as sess:\n    with tf.Session() as sess:\n        # Initializing the variables\n        sess.run(tf.global_variables_initializer())\n\n        save_model_path = os.path.join(SAVE_MODEL, \'image_classification_\' + log_filename)\n\n        writer = tf.summary.FileWriter(os.path.join(LOGDIR, log_filename))\n        writer.add_graph(sess.graph)\n\n        saver = tf.train.Saver()\n\n        # Training cycle\n        for epoch in range(epochs):\n            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_size):\n                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)\n            print(\'  Epoch {:>2}:  \'.format(epoch + 1), end=\'\')\n            print_stats(sess, batch_features, batch_labels, cost, accuracy)\n\n            if epoch % 5 == 0:\n                [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x:batch_features, y: batch_labels, keep_prob: 1.0})\n                writer.add_summary(s, epoch)\n\n#             if epoch % 100 == 0:\n#                 # Save Model\n#                 save_path = saver.save(sess, save_model_path)\n\n        # Save Model\n        save_path = saver.save(sess, save_model_path)\n    print("  done")\n\n\n# 全部実行\nlearning_rates = [0.001, 0.0001]\nfor learning_rate in learning_rates:\n    for use_two_fully in [False, True]:\n        for use_two_conv in [False, True]:\n            params_s = param_string(learning_rate, use_two_conv, use_two_fully)\n            print(\'Starting train: {}\'.format(params_s))\n            %time train(learning_rate, use_two_conv, use_two_fully, params_s)\n\nprint("All done")\nprint(\'run `tensorboard --logdir={}` to see the results.\'.format(LOGDIR))')


# # Checkpoint
# The model has been saved to disk.
# ## Test Model
# Test your model against the test dataset.  This will be your final accuracy. You should have an accuracy greater than 50%. If you don't, keep tweaking the model architecture and parameters.

# In[18]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import tensorflow as tf
import pickle
import gzip
import helper
import random
import os.path


n_samples = 4
top_n_predictions = 3


def test_model(learning_rate, use_two_conv, use_two_fully, log_filename):
    """
    Test the saved model against the test dataset
    """
    with gzip.open('preprocess_test.p.gz', mode='rb') as file:
        test_features, test_labels = pickle.load(file)
    
    build_network(learning_rate, use_two_conv, use_two_fully)
    
    save_model_path = os.path.join(SAVE_MODEL, 'image_classification_' + log_filename)
    #save_model_path = os.path.join(SAVE_MODEL, 'image_classification')

    loaded_graph = tf.Graph()
    
    config = tf.ConfigProto(device_count = {'GPU': 0})

    with tf.Session(config=config, graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy/accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for test_feature_batch, test_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)

learning_rates = [0.001, 0.0001]
for learning_rate in learning_rates:
    for use_two_fully in [False, True]:
        for use_two_conv in [False, True]:
            params_s = param_string(learning_rate, use_two_conv, use_two_fully)
            print('Starting train: {}'.format(params_s))
            
            get_ipython().run_line_magic('time', 'test_model(learning_rate, use_two_conv, use_two_fully, params_s)')

