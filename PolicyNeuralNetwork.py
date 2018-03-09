import tensorflow as tf
import numpy as np
import gym


## POLICY GRADIENT NEURAL NETWORK
## given a set of observation(inputs of the network), learn which action take. 
class PolicyNeuralNetwork:
    def __init__(self, n_inputs, n_outputs, n_hidden, optimizer_version=1):
        '''
        n_inputs: 	Integer. 	number of inputs of the network
        n_outputs: 	Integer.	number of outputs of the network
        n_hidden:	Integer.	number of hidden neurons (same for every hidden layer)
        optimizer_version: 1 or 2.	type of policy optimization chosed
        '''

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.optimizer_version = optimizer_version

    def _build_nn(self, learning_rate=0.01):
        self.X = tf.placeholder(tf.float32, shape=(None, self.n_inputs), name='X')
        
        ## Neural network's architecture
        with tf.name_scope("policy_nn"):
            hidden = tf.layers.dense(self.X, self.n_hidden, activation=tf.nn.relu, kernel_initializer=self.initializer)
            hidden2 = tf.layers.dense(hidden, self.n_hidden, activation=tf.nn.relu, kernel_initializer=self.initializer)
            logits = tf.layers.dense(hidden2, 1, activation=None)

        with tf.name_scope("policy_actions"):
            self.output = tf.nn.sigmoid(logits)
            self.p_action = tf.concat([self.output, 1-self.output], axis=1) # Tensor with probabilities of every action
            self.action = tf.multinomial(tf.log(self.p_action), num_samples=1) # Sampled following the multinomial distribution

        ## loss calculate using the cross_entropy
        with tf.name_scope("policy_loss"):
            y = 1 - tf.to_float(self.action) 
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits) # Assume that the action took is correct (the gradient will be correct later)

        ## Two similar version for update the gradient.
        ## Both use Adam optimizer
        if self.optimizer_version == 1:
            with tf.name_scope("policy_optimizer1"):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                gradients = optimizer.compute_gradients(self.loss)

                self.grads = [g for g, v in gradients]	# List of gradients compute by the optimizer

                ## create a placeholder for every gradient. This will be updated later, accordingly to the future rewards
                self.gradients_to_minimize = []
                grads_and_vars = []
                for gradient, variable in gradients:
                    place_h = tf.placeholder(tf.float32, shape=gradient.get_shape())
                    self.gradients_to_minimize.append(place_h)
                    grads_and_vars.append((place_h, variable))

                self.training_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
                
        elif self.optimizer_version == 2:
            with tf.name_scope("policy_optimizer2"):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                gradients = tf.gradients(self.loss, tf.trainable_variables())
                self.grads = gradients

                ## create a placeholder for every gradient. This will be updated later, accordingly to the future rewards
                self.gradients_to_minimize = []
                grads_and_vars = []
                for gradient, variable in zip(gradients, tf.trainable_variables()):
                    place_h = tf.placeholder(tf.float32, shape=gradient.get_shape()) ## IS THIS CORRECT??
                    self.gradients_to_minimize.append(place_h)
                    grads_and_vars.append((place_h, variable))

                self.training_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        
    def _summary(self):
        loss_scalar = tf.summary.scalar('policy_loss', tf.squeeze(self.loss))
        return loss_scalar