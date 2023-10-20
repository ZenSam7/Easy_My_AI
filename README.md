# DOCUMENTATION
My_AI is for learning how AI works (see source code: "Code_My_AI.py") as well as building your own AI in the easiest way (see: "My_AI.py use case", "AI for snake.py" and "Example of using Q-learning"). I also wrote it only with numpy, and I implemented it completely myself (did not view / did not copy anyone's code)

# How to use my library: 👉
### • Import the main file "Code_My_AI.py" and Create an instance of the class AI

```python

from My_AI import Code_My_AI

ai = Code_My_AI.AI()
```

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
ai._alpha = 1e-2  # Alpha coefficient (learning rate)

ai._number_disabled_weights = 0.0  # What proportion of weights we "turn off" during training
# (This is necessary so that there is no overlearning (memorizing responses instead of finding correlations))

ai._batch_size = 1  # Batch size in batch gradient descent

ai.kit_act_funcs.value_range(0, 1)  # What is the range of activation functions

# Which activation function we use for the output values (May be None)
ai.end_act_func = ai.kit_act_funcs.tanh
```


####  
### • Train AI based on input and correct answer

```python
data = [0, 1, 2]   # Required as a list of numbers (required length: number of inputs)
answer = [2, 1, 0] # Required as a list of numbers (required length: number of outputs)

ai.learning(
    data, answer, get_error = False, type_error="regular",
    type_regularization=1, regularization_value=10,
    regularization_coefficient=0.1, impulse_coefficient=0.9
            )
"""
Errors can be: regular, quadratic, logarithmic

Regularization can be: quadratic (the more weight, the more punish),
                       penalty   (if weights exceed regularization_value, then we punish)

get_error: returns a list of errors

impulse_coefficient: Momentum factor in Adam optimizer (usually around 0.7 ~ 0.99)
"""
```
> And if you want to see for yourself what answer the AI gave out, then just pass the input data to this method:  ai.start_work(data)


####  
### • If there is no correct answer to your task, then use Q-learning, reward AI for good (reward_for_state > 0) and punish for bad (reward_for_state < 0)

```python
# But first write commands ->

# Check that the number of output neurons is equal to the number of actions
all_possible_actions = ["left", "right", "up", "down"]

gamma = 0.4     # Coefficient of "confidence in experience"
epsilon = 0.15  # 15% chance that the Agent (AI) will give a more random answer (Needed to "study" the environment)
q_alpha = 0.01  # Q-table update rate

ai.make_all_for_q_learning(all_possible_actions, gamma, epsilon, q_alpha)

# After ->

# Also make sure the number of input neurons is equal to the size of the state list
ai_state = [0, 1]  # For example, coordinates

ai.q_learning(ai_state, reward_for_state,
              num_update_function=1, learning_method=2.2,
              type_error="regular", recce_mode=False,
              type_regularization="quadratic", regularization_value=2, regularization_coefficient=0.1,
              impulse_coefficient=0.9)
"""
recce_mode - if set to True, enable "reconnaissance mode",
i.e. in this mode, the AI does not learn, but only the Q-table is replenished
(and random actions are performed)
P.s. I recommend turning it on before training

Errors can be: regular, quadratic, logarithmic

Regularization can be: quadratic (the more weight, the more punish),
                       penalty   (if weights exceed regularization_value, then we punish)

impulse_coefficient: Momentum factor in Adam optimizer (usually around 0.7 ~ 0.99)


You can choose the most suitable q_table-table update function for you
P.s. The difference between the functions is negligible

learning_method :
1 : As the "correct" answer, the one that is most rewarded is selected
P.s. This is not very good, because other options that bring either the
     same or a little less reward are ignored (and only one "correct" one is selected).
     BUT IT IS WELL SUITABLE WHEN YOU HAVE EXCLUSIVELY ONE CORRECT ANSWER
     IN THE PROBLEM AND THERE CANNOT BE "MORE" AND "LESS" CORRECT

2: Making more useful answers more “correct”
(in the fractional part after 2, indicate the degree by which we notice the discrepancy
between the “more” and “less” correct answers 
(for example: 2.345 means a degree of difference of 3.45))

BTW, if your AI learns very badly (or does not learn at all), then look at the Q-table, if there are mostly (> 50%) negative numbers, then in this case you need to reward more and punish less (so that there are more positive numbers)
"""
```
> And if you want to see for yourself what answer the AI gave out, then just pass the input data to this method:  ai.q_start_work(data) 


####  
#### • Or then you can create several AIs, and then cross the best of them, using the method:
```python
better_ai_0.genetic_crossing_with(better_ai_1)
```


#### • Or you can change (mutate) the AI so that it doesn't stand still or hope that some of mutations turn out to be good

```python
ai.make_mutations(0.05)  # Replacing 5% of all weights with random numbers
```

####  
### • If your input data can take any value and/or vary over a large range, then normalize it with :

```python
# (Better to normalize from 0 to 1 OR -1 to 1)
ai.kit_act_funcs.normalize(data, min_value, max_value)
```


####  
####  
####  
####  
Good luck
(づ｡◕‿‿◕｡)づ
