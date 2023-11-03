# ДОКУМЕНТАЦИЯ
My_AI это небольшая библиотека (скорее даже микро библиотека) для создания собственного простенького ИИ. Я её написал полностью с нуля, используя только numpy, а из интернета я брал ТОЛЬКО основные принципы работы. Также я иставил кучу комментариев к коду, чтобы вы могли сами разобраться как здесь всё работает и покапаться в коде (может даже что-то новое узнаете)

# Как использовать мою библиотеку: 👉
# Супер кратко:
```python
from My_AI import AI_ensemble, AI

# Создаём ИИ
ai = AI(architecture=[10, 50, 50, 1],
        add_bias_neuron=True,
        name="First_AI")
""" Или можно создать ансамбль
ai = AI_ensemble(amount_ais=10, architecture=[10, 50, 50, 1],
                      add_bias_neuron=True,
                      name="First_AI")
"""


# Устанавливаем коэффициенты
ai.alpha = 1e-3
ai.batch_size = 10

ai.what_act_func = ai.kit_act_func.tanh
ai.end_act_func = ai.kit_act_func.tanh
ai.save_dir = "Saves AIs"

# Обучаем (Например распозновать картинки)
for image in dataset:
    data = image.tolist()
    answer = [image.what_num]
    ai.learning(data, answer, squared_error=True)

""" Есть также и Q-бучение (функции обновления таблицы см. ниже)
actions = ("left", "right", "up", "down")
ai.make_all_for_q_learning(actions, ai.kit_upd_q_table.standart,
                           gamma=0.5, epsilon=0.01, q_alpha=0.1)

state, reward = game.get_state_and_reward()
ai.q_learning(state, reward,
              learning_method=2.5, squared_error=False)
"""
```




# Подробности:

#### Можете скопировать пакет "My_AI" к себе в проект (всё остальное просто примеры использования), и импоритровать от туда классы AI или AI_ensemble

### Как инициализировать ИИшку:
```python
from My_AI import AI_ensemble, AI

ai = AI()
# Или
ensemble = AI_ensemble(5) # 5 —— количество ИИшек в ансамбле
```
> Ансамбль — это несколько ИИ в одной коробке, которые вместе принимают решение (ансамбль подходит для Q-обучения (обучение с подкреплением; когда нету "правильного" и "неправильного" ответа, а только вознаграждение за какое-то выбранное действие))

> P.s. В качестве примера посмотрите на ИИ для змейки (в файле "AI for snake.py")


### • Чтобы использовать ИИшку надо создать архитектуру одним из способов:
```python
ai = AI(architecture=[3, 4, 4, 4, 3],
        add_bias_neuron=True,
        name="First_AI")

""" Если используете ансамбль
ensemble = AI_ensemble(5, architecture=[3, 4, 4, 4, 3],
                       add_bias_neuron=True,
                       name="First_AI")
"""
```
или
```python
ai.create_weights([3, 4, 4, 4, 3],
                  add_bias_neuron=True)
ai.name = "First_AI"
# Имя не обязательно указывать имя, но тогда при сохранении
# будет использовано случайное число вместо имени
```
Таким образом будет создана следущая архитектура:
<div id="header" align="left">
  <img src="https://i.ibb.co/nbbTLZS/Usage-example.png" width="600"/>
</div>


####  
### • Гаперпараметры:
```python
"""Прописывать или изменять все гаперпараметры необязательно"""

ai.alpha = 1e-2  # Альфа (скорость обучения)

ai.number_disabled_weights = 0.0  # Какую долю весов отключаем
# (Это надо чтобы не возникало переобучение)

ai.batch_size = 10  # Сколько ответов усредняем, чтобы на них учиться
# (Ускоряет обучение и (иногда, далеко не всегда) улучшает качество обучения)

# Функция активации нейронов (крайне рекомендую оставить tanh,
# т.к. с ним ИИ работает в разы быстрее)
ai.what_act_func = ai.kit_act_func.tanh

# Функция активации для последнего слоя (аналогично, рекомендую оставить tanh)
# P.s. end_act_func может и отстутсвовать (т.е. можно установить None)
ai.end_act_func = ai.kit_act_func.tanh
```


####  
### • Обучение:

```python
# На вход и на выход нейросети надо просто подавать список чисел
data = [0, 1, 2]   # Входные данные
answer = [0, 1, 0] # Выходные данные

ai.learning(data, answer,
            squared_error=True)
"""
Квадратичная ошибка позволяет быстрее обучаться на больших косяках
и закрывать глаза на мелкие недочёты (но иногда лучше её отключать)
"""
```

### Q-бучение:
> Q-обучение (оно же обучение с подкреплением) — это когда нету "правильного" и "неправильного" ответа, а только вознаграждение за какое-то выбранное действие, т.е. на сколько оно хорошее 

> (0 = нейтральный выбор, <0 = плохой выбор, >0 = хороший выбор)

```python
# ИИшка может только выбрать какое-то конкретное действие из возмажных
all_possible_actions = ["left", "right", "up", "down"]

gamma = 0.4     # Коэффициент "доверия опыту"
epsilon = 0.15  # Доля на случайных действий (чтобы ИИшка изучала окружающую среду)
q_alpha = 0.1   # Скорость обновления Q-таблицы (на самом деле оно почти ни на что не влияет) 

ai.make_all_for_q_learning(all_possible_actions,
                           ai.kit_upd_q_table.standart,
                           gamma=gamma, epsilon=epsilon, q_alpha=q_alpha)


# Also make sure the number of input neurons is equal to the size of the state list
ai_state = [0, 1]  # For example, coordinates

ai.q_learning(ai_state, reward_for_state, learning_method=2.2,
              recce_mode=False, squared_error=True)
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
