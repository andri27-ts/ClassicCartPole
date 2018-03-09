import tensorflow as tf
import numpy as np
import gym

class ModelNeuralNetwork:
    def __init__(self, n_inputs, n_outputs, n_hidden):
        '''
        n_inputs: 	Integer. 	number of inputs of the network
        n_outputs: 	Integer.	number of outputs of the network
        n_hidden:	Integer.	number of hidden neurons (same for every hidden layer)

        '''

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
    
    def _build_nn(self, learning_rate=0.001):
        self.X = tf.placeholder(tf.float32, shape=(None, self.n_inputs), name='X')
        self.y = tf.placeholder(tf.float32, shape=(None, self.n_outputs), name='y')
        
        ## Neural network's architecture
        with tf.name_scope("model_nn"):
            hidden = tf.layers.dense(self.X, self.n_hidden, activation=tf.nn.relu, kernel_initializer=self.initializer)
            hidden2 = tf.layers.dense(hidden, self.n_hidden, activation=tf.nn.relu, kernel_initializer=self.initializer)
            tf.summary.histogram("hidden2",hidden2)
            self.obs_predicted = tf.layers.dense(hidden2, 4, activation=None)
            self.done_predicted = tf.layers.dense(hidden2, 1, activation=tf.nn.sigmoid)
            self.reward_predicted = tf.layers.dense(hidden2, 1, activation=tf.nn.sigmoid)
            tf.summary.histogram("obs_predicted", self.obs_predicted)
            tf.summary.histogram("done_predicted", self.done_predicted)
            tf.summary.histogram("reward_predicted", self.reward_predicted)
            
        ## mean square error calulate for observations, reward and done predictions. Then will be summed together
        with tf.name_scope("losses"):
            self.obs_loss = tf.reduce_mean(tf.square(self.y[:, 0:4] - self.obs_predicted))
            self.reward_loss = tf.square(self.y[:, 4:5] - self.reward_predicted)
            self.done_loss = tf.square(self.y[:, 5:6] - self.done_predicted)

            self.loss = tf.reduce_mean(self.obs_loss*4 + self.reward_loss + self.done_loss)
            
        ## Calculate the gradient and update the gradient using Adam optimizer
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.training_op = optimizer.minimize(self.loss)

    ## Some scalar to keep track during training
    def _summary(self):
        loss_scalar = tf.summary.scalar('loss', tf.squeeze(self.loss))
        obs_loss = tf.summary.scalar('obs_loss', tf.reduce_mean(self.obs_loss))
        reward_loss = tf.summary.scalar('reward_loss', tf.squeeze(self.reward_loss))
        done_loss = tf.summary.scalar('done_loss', tf.squeeze(self.done_loss))
        
        return tf.summary.merge([loss_scalar, obs_loss, reward_loss, done_loss])
