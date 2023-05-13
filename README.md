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
ai.alpha = 1e-7   # Alpha coefficient (learning rate coefficient)
# ATTENTION !! IF YOU HAVE ERRORS IN THE CALCULATION (or have too large numbers), THEN DECREASE alpha

ai.number_disabled_neurons = 0.0    # What proportion of neurons we "turn off" during training

ai.packet_size = 1     # Batch size in batch gradient descent
# How many errors we will average in order to change the weights based on this average error
# The larger the packet_size, the lower the "quality of training", but the speed of training iterations is greater

ai.activation_function.value_range(0, 1)    # What is the range of values in the output

# Which activation function we use for the output values
ai.end_activation_function = ai.activation_function.Sigmoid 
```


####  
### • Train AI based on input and correct answer
```python
data  =  [0, 1, 2]   # Required as a list of numbers (required length: number of inputs)
# (The input must be completely non-zero! (otherwise won't learn))
answer = [2, 1, 0]   # Required as a list of numbers (required length: number of outputs)

ai.learning(data, answer)
```

> And if you want, you can write down the value of the error during training
```python
errors.append( ai.learning(data, answer, get_error=True) )
```
> And if you want to see for yourself what answer the AI gave out, then just pass the input data to this method:
```python
ai.start_work(data)
```
####  
### • If there is no correct answer to your task, then use Q-learning, reward AI for good (reward_for_state >= 0) and punish for bad (reward_for_state < 0)
```python
# But first run command ->
all_possible_actions = ["left", "right", "up", "down"]
gamma = 1.1       # Coefficient of "confidence in experience"
epsilon = 0.15    # 15% chance that the Agent (AI) will give a random answer (Needed to "learn" the environment)
ai.make_all_for_q_learning(all_possible_actions, gamma, epsilon)

# After ->
ai_state = [0,1]   # For example coordinates (The input must be completely non-zero! (otherwise won't learn))
future_state = [1,1]
ai.q_learning(ai_state, "right", reward_for_state, future_state)
```


####  
### • Or then you can choose (yourself, according to your parameters) the AI that coped with it better than others, and cross it with another (also good) AI
```python
better_ai_1.genetic_crossing_with(better_ai_2)
```

### • Or you can change (mutate) the AI so that it doesn't stand still or hope that some of mutations turn out to be good
> If you want to eliminate overlearning in this way (memorizing responses instead of finding correlations), then you better use number_disabled_neurons
```python
ai.get_mutations(0.01)  # Replacing 1% of all weights with random numbers
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
Good luck to fixing bugs
(づ｡◕‿‿◕｡)づ
