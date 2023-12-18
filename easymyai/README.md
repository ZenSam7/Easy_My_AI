[GitHub](https://github.com/ZenSam7/My_AI)
# DOCUMENTATION
SimpleMyAI is a small library for creating your own simple AI. I wrote it completely from scratch, using only numpy, and I took ONLY the basic principles from the internet. I also put a lot of comments to the code, so you can understand how everything works and dig into the code yourself (maybe even learn something new).

# DOWNLOAD
```
pip install easymyai
```

# How to use my library: 👉
# Super short:

```python
from simplemyai import AI_ensemble, AI

# Create AI
ai = AI(architecture=[10, 50, 50, 1],
        add_bias=True,
        name="First_AI")
""" Or you can create an ensemble

ai = AI_ensemble(amount_ais=10, architecture=[10, 50, 50, 1],
                      add_bias_neuron=True,
                      name="First_AI")
"""

# Set the coefficients
ai.alpha = 1e-3 # Learning rate
ai.batch_size = 10 # Batch size
# For Adam optimizer
ai.impulse1 = 0.9 # Typically 0.8 to 0.999
ai.impulse2 = 0.999 # Slightly different from beta1
# Regularization
ai.l1 = 0.0 # L1 regularization
ai.l2 = 0.0 # L2 regularization

ai.what_act_func = ai.kit_act_func.tanh
ai.end_act_func = ai.kit_act_func.tanh
ai.save_dir = "Saves AIs" # To which folder we save AIs.

# Train (e.g. to recognize images)
for image in dataset:
    data = image.tolist()
    answer = [image.what_num]
    ai.learning(data, answer, squared_error=True)

""" There is also Q-learning (see below for table update functions)

actions = ("left", "right", "up", "down")
ai.make_all_for_q_learning(actions, ai.kit_upd_q_table.standart,
                           gamma=0.5, epsilon=0.01, q_alpha=0.1)

state, reward = game.get_state_and_reward()
ai.q_learning(state, reward,
              learning_method=2.5, squared_error=False)
"""
```




# Details:

#### You can copy the "simplemyai" package to your project (everything else is just sample usage), and imporitize the AI or AI_ensemble classes from there

### How to initialize AI:
```python
from simplemyai import AI_ensemble, AI

ai = AI()
# Or
ensemble = AI_ensemble(5) # 5 — the number of AIs in the ensemble
```
> An ensemble is several AIs in one box that make a decision together (an ensemble is suitable for Q-learning (reinforcement learning; when there is no "right" and "wrong" answer, but only a reward for some chosen action)).

> P.s. As an example, look at the AI for snake (in the file "AI for snake.py")


#### • To use the AI you need to create the architecture in one of the ways:
```python
ai = AI(architecture=[3, 4, 4, 4, 3],
        add_bias_neuron=True,
        name="First_AI")

""" If using ensemble

ensemble = AI_ensemble(5, architecture=[3, 4, 4, 4, 3],
                       add_bias_neuron=True,
                       name="First_AI")
"""
```
or

```python
ai.create_weights([3, 4, 4, 4, 3], add_bias=True)
ai.name = "First_AI"
# The name does not have to be a name, but then when you save it.
# A random number will be used instead of the name
```

The following architecture will be created in this way:
<div id="header" align="left">
  <img src="https://i.ibb.co/nbbTLZS/Usage-example.png" width="600"/>
</div>


####  
### • Hyperparameters:
```python
""" It is optional to specify or modify all haperparameters."""""

ai.alpha = 1e-2 # Скорость обучения

ai.number_disabled_weights = 0.0 # Какую долю весов отключаем
# (Это надо чтобы не возникало переобучение)

ai.batch_size = 10 # Сколько ответов усредняем, чтобы на них учиться
# (Ускоряет обучение и (иногда, далеко не всегда) улучшает качество обучения)

# Neuron activation function (I highly recommend leaving tanh if it is
# is available, as the AI works much faster with it)
ai.what_act_func = ai.kit_act_func.tanh

# Activation function for the last layer (similarly, I recommend leaving tanh)
# P.s. end_act_func may not exist (i.e. you can set it to None)
ai.end_act_func = ai.kit_act_func.tanh

# Impulse coefficients for the Adam optimizer
ai.impulse1 = 0.9  
ai.impulse2 = 0.999
# If you don't know what an optimizer is google it, it's very interesting))))

# Coefficients for regularizing the weights 
# (Regularization is keeping the weights around 0 (or [-1; +1]) ) )
ai.l1 = 0.001  # How much we reduce the weights (takes the weights straight to 0)
ai.l2 = 0.001   # How many times we reduce the weights (keeps the weights near 0)
```
####  
### • Training:

```python
# The input and output of the neural network should simply be a list of numbers
data = [0, 1, 2] # Input data
answer = [0, 1, 0] # Output data

ai.learning(data, answer,
            squared_error=True)
"""
Quadratic error allows you to learn faster on large joints
and turn a blind eye to small errors (but sometimes it is better to turn it off).
"""
```

# Q-learning:
> Q-learning (aka reinforcement learning) is when there is no "right" and "wrong" answer, but only a reward for some chosen action, i.e., how good it is 

> (<0 = bad choice, 0 = neutral choice, >0 = good choice)

```python
# The AI can only select a specific action from the possible ones.
all_possible_actions = ["left", "right", "up", "down"]

gamma = 0.4 # Experience confidence factor (to "smooth" the Q-table)
epsilon = 0.15 # Percentage on random actions (for the AI to learn the environment)
q_alpha = 0.1 # Q-table update rate (actually it has almost no effect on anything) 

ai.make_all_for_q_learning(all_possible_actions,
                           ai.kit_upd_q_table.standart,
                           gamma=gamma, epsilon=epsilon, q_alpha=q_alpha)
# Table update functions have a pretty big impact on learning

# As with normal learning, the input is simply a list of numbers (state)
ai_state = [0, 1] # For example, the coordinates of the neural network

ai.q_learning(ai_state, reward_for_state, learning_method=1,
              recce_mode=False, squared_error=True)

# What decision the AI made given certain data
predict = ai.q_predict(ai_state)
```

> recce_mode: "environmental exploration" mode (choose a random action all the time)

> Learning methods (learning_method value determines) :
> - 1 : The "correct" answer is chosen as the one that is maximally rewarded, and the
the place of the action (which leads to the best answer) is set to the maximum value of the
of the activation function, and on the other places the minimum of the activation function.
P.s. This is not very good, because it ignores other options that bring either the same amount of reward,
or slightly less reward (and only one "right" one is chosen). BUT IT'S A GOOD FIT,
WHEN YOU HAVE ONLY 1 CORRECT ANSWER IN A PROBLEM, AND THERE CAN BE NO "MORE" OR "LESS" CORRECT ANSWERS.
> - 2 : Making the answers that are more rewarded more "correct". The fractional part of the number means the degree to which we will raise "striving for better results" (EXAMPLE: 2.3 means that we use learning method 2 and raise "striving for better results" to degree 3, and 2.345 means that the degree will be 3.45).


### • And of course saving and loading the neural network
```python
ai.save() # Save under the current name
ai.save("First_AI") # Save under the name First_AI ()
```
> You can also choose your own folder for saving AIs (why? I don't know).
```python
# Everything will be saved in the SAVES
ai.save_dir = "SAVES"
```

####  
### • I also did genetic learning for fun.
> This is when the weights of the AIs are mixed up.

```python
# Here we mix the weights of ai_0 with ai_1 and leave ai_1 untouched.
better_ai_0.genetic_crossing_with(better_ai_1)
```
### • In addition to the genetic algorithm, I created the ability to create mutations
> Some fraction of the weights are replaced with random numbers from -1 to 1 

```python
ai.make_mutations(0.05) # 5% of the weights turn out to be random numbers
```

####  
#### • By the way, it is highly recommended to translate the input numbers to between -1 and 1 (or 0 and 1) 
> It's just easier for the AI to work with numbers from -1 to 1 (or 0 to 1) than with incomprehensible huge values.

```python
# It's easier to use normalize, but you can also use tanh (or sigmoid).
ai.kit_act_funcs.normalize(data, 0, 1)
```


#  
#  
#  
###### _Good luck catching bugs_ (づ｡◕‿‿◕｡)づ