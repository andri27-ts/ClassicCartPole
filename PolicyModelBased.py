import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import random

import auxiliar as aux
from PolicyNeuralNetwork import PolicyNeuralNetwork
from ModelNeuralNetwork import ModelNeuralNetwork




tf.reset_default_graph()

env = gym.make('CartPole-v0')


## Create and initialize the policy neural network
with tf.name_scope('policy_neural_network'):
    p_nn = PolicyNeuralNetwork(n_inputs=4, n_outputs=1, n_hidden=5)
    p_nn._build_nn()
    
## Create and initialize the model neural network
with tf.name_scope('model_neural_network'):
    m_nn = ModelNeuralNetwork(n_inputs=5, n_outputs=6, n_hidden=10)
    m_nn._build_nn(learning_rate=0.002)
    
policy_merged_s = p_nn._summary()
model_merged_s = m_nn._summary()


## Summary variables
root_logdir = "tf_logs"
now = datetime.utcnow().strftime("%Y_%m_%d_%H.%M.%S")
logdir = "{}/run-{}/".format(root_logdir, now)
writer = tf.summary.FileWriter(logdir)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

max_actions = 100
games_for_it = 5
iterations = 100
discount_factor = 0.95
save_iterations=50
initial_model_iterations = 1000
model_iterations = 0
count_m_it = 0
count_p_it = 0


policy_rewards = [1]
model_rewards = []
model_losses = []
losses = []
tests_rewards = []


'''
The main body of the programs.
Because it's a policy gradient model based, instead of learn the policy by training the agent directly into the environment,
it'll learn to play on the model of the cart pole game (previously learnt).

Let's see how this works. 
The algorithm on every iteration will try to improve the representation of the model so that the policy will improve its performance by playing on it.

To keep track of the improvements, the mean reward of the policy and the mean loss of the model will be printed for every iteration.
To be sure of the correct merge of the policy and the model, on every iteration, the policy will be executed on the real environment. If the reward of the latter will keep improving, the algorithm will be correct
'''

with tf.Session() as sess:
    sess.run(init)
    writer.add_graph(sess.graph)

    for it in range(iterations):
        games_rew = []
        games_grads = []
        pol_rews = []
        model_lss = []
        prev_env_state = env.reset()
        hist_dataset = []
        
        if it == 0:
            model_iterations = initial_model_iterations
        else:
            model_iterations = 500

        ## Learn the MODEL of the cartpole game
        for it_m in range(model_iterations):
            prev_env_state = env.reset()
            ## V2
            X_model = []
            y_model = []
            batch_size=8

            for i in range(max_actions):
                policy_act = sess.run(p_nn.action, feed_dict={p_nn.X:[prev_env_state]}) 

                ## TO CHECKUP TRY THIS DUMP POLICY
                #if policy_act == 0: policy_act = 1
                #else: policy_act = 0

                try:
                    env_state, env_reward, env_done, _ = env.step(np.squeeze(policy_act))
                except AssertionError:
                    print(prev_env_state, sess.run(p_nn.p_action, feed_dict={p_nn.X:[prev_env_state]}))
                    print('\n',np.squeeze(policy_act),'\n')

                model_y = np.concatenate([env_state, [env_reward], [int(env_done)]], axis=0)
                model_inp = np.concatenate([prev_env_state, [np.squeeze(policy_act)]], axis=0)

                if count_m_it % 45 == 0:
                    s = sess.run(model_merged_s, feed_dict={m_nn.X:[model_inp], m_nn.y:[model_y]})
                    writer.add_summary(s, count_m_it)

                #o_p, r_p, d_p = sess.run([m_nn.obs_predicted, m_nn.reward_predicted, m_nn.done_predicted], feed_dict={m_nn.X:[model_inp]})


                ## V3
                hist_dataset.append([model_inp.tolist(), model_y.tolist()])
                minim = min(batch_size, len(hist_dataset))

                '''if count_m_it % (batch_size/2) == 0 and minim > 1:
	                batch = aux.get_batch(hist_dataset, minim)
	                X_model = batch[:, 0].tolist()
	                y_model = batch[:, 1].tolist()
	                
	                model_loss, _ = sess.run([m_nn.loss, m_nn.training_op], feed_dict={m_nn.X:X_model, m_nn.y:y_model})
	                model_losses.append(model_loss)
	                model_lss.append(model_loss)'''

                # V2
                X_model.append(model_inp)
                y_model.append(model_y)

                if i % batch_size == 0 or env_done:
                	model_loss, _ = sess.run([m_nn.loss, m_nn.training_op], feed_dict={m_nn.X:X_model, m_nn.y:y_model})
                	model_losses.append(model_loss)
                	model_lss.append(model_loss)

                	X_model = []
                	y_model = []
                	

                ## V1
                '''model_loss, _ = sess.run([m_nn.loss, m_nn.training_op], feed_dict={m_nn.X:[model_inp], m_nn.y:[model_y]})
               	model_losses.append(model_loss)
                model_lss.append(model_loss)'''


                prev_env_state = env_state
               

                count_m_it += 1

                if env_done:
                	break
                
            
        ## Learn the policy of the cartpole game
        for g in range(games_for_it):
            prev_env_state = env.reset()
            game_rew = []
            game_grads = []
            
            
            for i in range(max_actions):
                
                ## take an action given the current model state
                policy_loss, policy_act, policy_grads = sess.run([p_nn.loss, p_nn.action, p_nn.grads], feed_dict={p_nn.X:[prev_env_state]}) 
                
                model_inp = np.concatenate((prev_env_state, [np.squeeze(policy_act)]), axis=0)		# input of the model (state concatenate with an action)

                ## state, reward and done returned by the model (NB: the actor play on the model, not on the real environment) (if correct they should be as close as possible as the real environment)
                model_state, model_reward, model_done = sess.run([m_nn.obs_predicted, m_nn.reward_predicted, m_nn.done_predicted], feed_dict={m_nn.X:[model_inp]}) 
                model_state, model_reward, model_done = np.squeeze(model_state), np.squeeze(model_reward), np.squeeze(model_done)

                game_grads.append(policy_grads)
                game_rew.append(model_reward)
                
                
                if count_p_it % 45 == 0:
                    s = sess.run(policy_merged_s, feed_dict={p_nn.X:[prev_env_state]})
                    writer.add_summary(s, count_p_it)
                
                
                prev_env_state = model_state
                
                count_p_it += 1

                if model_done > 0.5:
                    break
                
            pol_rews.append(np.sum(game_rew))
           
            
            games_grads.append(game_grads)
            games_rew.append(game_rew)
            
        
        ## Update the gradients following the games played
        ## Calculate the discounted normalized reward for every action. Based on this, the gradients will be updated
        dis_games_rew = aux.discount_and_normalize_rewards(games_rew, discount_factor)
        
        update_gradients = [np.array(games_grads[g][step])*np.array(dis_games_rew[g][step]) for g, game_g in enumerate(games_grads) for step, _ in enumerate(game_g)]
        update_mean_gradient = np.mean(update_gradients, axis=0)

        feed_dict = dict([(gtm, umg) for gtm, umg in zip(p_nn.gradients_to_minimize, update_mean_gradient)])
        sess.run(p_nn.training_op, feed_dict=feed_dict)
        
        

        prev_env_state = env.reset()   
        test_reward = 0

        ## Test the policy. If the model of the game and the policy trained on it are correct, we'll see the reward increase throughout the iterations
        done = False
        while not done:
            action = sess.run([p_nn.action], feed_dict={p_nn.X:[prev_env_state]}) 
            state, reward, done, info = env.step(np.squeeze(action))
            prev_env_state = state
            test_reward += reward
            
        
        ## Print the values used to keep track of the progress
        print("Iteration: {}, rewards:{:.2f}, model_loss:{:.5f}, test_reward:{}".format(it, np.mean(pol_rews), np.mean(model_lss), test_reward)) 
        policy_rewards.append(np.mean(pol_rews))
        tests_rewards.append(test_reward)
        
        #if it % save_iterations == 0:
        #    saver.save(sess, "/tmp/policy_cartpole/1")
    
print('\n',np.mean(policy_rewards))
plt.plot(tests_rewards)
plt.plot(policy_rewards)
plt.show()
plt.plot(model_losses, alpha=0.7)
plt.show()



