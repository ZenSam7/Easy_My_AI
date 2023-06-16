# DOCUMENTATION
My_library is for learning how AI works (see source code: "Code_My_AI.py") and for creating your own AI in the easiest possible way (see: "Example use My_AI.py", "AI for snake.py" and "Example use Q-learning")


####  
# How to use my library: 👉
### • Import the main file "Code_My_AI.py" and Create an instance of the class AI
```python
import Code_My_AI

ai = Code_My_AI.AI()
```


####  
### • Create the AI architecture
```python
ai.create_weights( [number_of_inputs,
                    number_of_neurons_in_layer_1,
                    number_of_neurons_in_layer_2, 
                    number_of_neurons_in_layer_3, 
                    ...,
                    number_of_outputs ],
                    add_bias_neuron = True)
```
<div id="header" align="left">
  <img src="https://i.ibb.co/nbbTLZS/Usage-example.png" width="600"/>
</div>


####  
### • Customize your settings:
> You can change nothing, or change only a some of parameters

```python
ai.alpha = 1e-2  # Alpha coefficient (learning rate coefficient)

ai.number_disabled_weights = 0.0  # What proportion of neurons we "turn off" during training
# (This is necessary so that there is no overlearning (memorizing responses instead of finding correlations))

ai.batch_size = 1  # Batch size in batch gradient descent
# How many errors we will average in order to change the weights based on this average error
# The larger the packet_size, the lower the "quality of training", but the speed of training iterations is greater

ai.act_func.value_range(0, 1)  # What is the range of values in the output

# Which activation function we use for the output values
ai.end_act_func = ai.act_func.Tanh  # (May be None)
```


####  
### • Train AI based on input and correct answer

```python
data = [0, 1, 2]  # Required as a list of numbers (required length: number of inputs)
# (Input must be at least sometimes completely different from zero! (otherwise it will not learn))
answer = [2, 1, 0]  # Required as a list of numbers (required length: number of outputs)

ai.learning(data, answer, type_error=1, type_regularization=1, regularization_value=100)
"""
Errors can be:
1: (regular:) |ai_answer - answer| / len(answer) 
2: (quadratic:) (ai_answer - answer)^2 / len(answer)
3: (logarithmic:) ln^2( (ai_answer - answer) +1 ) / len(answer)

Regularization can be:
1: delta += SameSign * sqrt( sum((weights/10) ^2) )
2: delta += SameSign * sum( abs( weight * (abs(weight) >= 10) -10 ) )

regularization_value: in what interval (±) do we keep weights
"""
```

####  
### • If there is no correct answer to your task, then use Q-learning, reward AI for good (reward_for_state > 0) and punish for bad (reward_for_state < 0)

```python
# But first write commands ->

# Check that the number of output neurons is equal to the number of actions
all_possible_actions = ["left", "right", "up", "down"]

gamma = 0.4  # Coefficient of "confidence in experience"
epsilon = 0.15  # 15% chance that the Agent (AI) will give a more random answer (Needed to "study" the environment)
q_alpha = 0.2  # Q-table update rate

ai.make_all_for_q_learning(all_possible_actions, gamma, epsilon, q_alpha)

# After ->

# Also make sure the number of input neurons is equal to the size of the state list
ai_state = [0, 1]  # For example, coordinates (Input must be at least sometimes completely different from zero! (otherwise it will not learn))

ai.q_learning(ai_state, reward_for_state, num_update_function=1, learning_method=2.1, type_error=1, recce_mode=False, type_regularization=1, regularization_value=100)
"""
recce_mode - if set to True, enable "reconnaissance mode", i.e. in this mode, the AI does not learn, but only the Q-table is replenished (and random actions are performed)
P.s. I recommend turning it on before training

Errors can be:
1: (regular:) |ai_answer - answer| / len(answer)
(Standard)
2: (quadratic:) (ai_answer - answer)^2 / len(answer)
(For big mistakes we punish a lot, and score on small mistakes)
3: (logarithmic:) ln^2( (ai_answer - answer) +1 ) / len(answer)
(We punish for medium and large errors of the same, quite a bit by reducing small errors)

Regularization (Regularize the weights so that they are not too large):
1: delta += SameSign * sqrt( sum( (weights/regular_val) ^2) )
2: delta += SameSign * sum( abs( weights * (abs(weights) >= regular_val) ) -regular_val )

regularization_value: In what interval (±) do we keep weights


You can choose the most suitable q-table update function for you
(Instead of 1, supply any other number that is in the description for this function)
P.s. The difference between the functions is negligible

Learning methods (the value of learning_method determines) :
1 : As the "correct" answer, the one that is most rewarded is selected, and the place of action (which leads to the best answer) is set to the maximum value of the activation function, and to the other places the minimum of the activation function
P.s. This is not very good, because. other options that bring either the same or a little less reward are ignored (and only one "correct" one is selected). BUT IT IS WELL SUITABLE WHEN YOU HAVE EXCLUSIVELY ONE CORRECT ANSWER IN THE PROBLEM AND THERE CANNOT BE "MORE" AND "LESS" CORRECT

2 : Making answers that are more rewarding more "correct" that we are using learning method 2 and raising to the power of 2 "striving for better results", and 2.345 means that the power will be 3.45 )


BTW, if your AI learns very poorly (or does not learn at all), then look at the Q-table, if there are mostly (> 50%) negative numbers, then in this case reward more and punish less (so that there are more positive numbers )
"""
```


> And if you want to see for yourself what answer the AI gave out, then just pass the input data to this method:
```python
# If you are using backpropagation learning, then:
ai.start_work(data)

# And if you use q_learning, then:
ai.q_start_work(data)        # Gives the selected AI action (in our example it is "left", "right", "up" or "down")
```
> And if you want, you can write down the value of the error during training
> (This cannot be done using q_learning)

```python
errors.append(ai.learning(data, answer))
```


####  
### • Or then you can choose (yourself, according to your parameters) the AI that coped with it better than others, and cross it with another (also good) AI
```python
better_ai_0.genetic_crossing_with(better_ai_1)
```


### • Or you can change (mutate) the AI so that it doesn't stand still or hope that some of mutations turn out to be good
> If you want to eliminate overlearning in this way (memorizing responses instead of finding correlations), then you better use number_disabled_neurons
```python
ai.get_mutations(0.05)  # Replacing 5% of all weights with random numbers
```

####  
### • If your input data is analog (i.e. it can take any value and/or vary by a large range), then normalize it with a suitable range of values for your data.
```python
ai.act_func.normalize(data, min_value, max_value)
```


####  
### • You can save, delete and load your own (or ready for examples) settings AI
> Including weights and Q-table of course
```python
ai.save_data("Name_AI")
ai.delete_data("Name_AI")
ai.load_data("Name_AI")
```


####  
####  
####  
####  
Good luck
(づ｡◕‿‿◕｡)づ
