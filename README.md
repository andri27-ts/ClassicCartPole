# ClassicCartPole

CartPole solved using policy gradient neural network trained on the model of the game.

### Motivation

This project is inspired by the amazing article about simple RL algorithms posted by Arthur Juliani https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-3-model-based-rl-9a6fe0cce99
The basic ideas are the same as the post but implemented in a different way.

### How it works

In short, ClassicCartPole is constituted by two major parts: the model-based part and the policy-based part. The former try to learn the dynamics of the real environment, the latter the policy to take in every state. 
The basic principle is that the policy is trained on the model instead of the real environment so that we actually never run the policy on the real environment.
In real tasks it's a huge opportunity to learn the model of the physical world, because you become capable to train the agent on the model of the environment, saving time and energy.

### Code structure
- [ModelNeuralNetwork.py](./ModelNeuralNetwork.py) : Contains the model neural network class
- [PolicyNeuralNetwork.py](./PolicyNeuralNetwork.py) : Contained the policy neural network class
- [PolicyModelBased.py](./PolicyModelBased.py) : Contains the main runner for the code. Here the model and the policy are trained and evaluated.
- [auxiliar.py](./auxiliar.py) : Contains some useful functions