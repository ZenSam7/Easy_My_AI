# DOCS ON ENG: [Here](https://pypi.org/project/easymyai/)


# КАК СКАЧАТЬ
```
pip install easymyai 
```

## Кстати, можно посмотреть на змейку, запустив EXEшку "Запусти меня!.exe" (надо скачать к нему папку "_internal")

# ДОКУМЕНТАЦИЯ
EasyMyAI это небольшая библиотека для создания собственного простенького ИИ. Я её написал _полностью_ с нуля, используя только numpy. Также я иставил кучу комментариев к коду, чтобы вы могли сами разобраться как здесь всё работает и покапаться в коде (может даже что-то новое узнаете)

# Как использовать мою библиотеку: 👉
# Супер кратко:

```python
from My_AI import AI_ensemble, AI

# Создаём ИИ
ai = AI(architecture=[10, 50, 50, 1],
        add_bias=True,
        name="First_AI")

# Или можно создать ансамбль:
ai = AI_ensemble(amount_ais=10, architecture=[10, 50, 50, 1],
                 add_bias_neuron=True,
                 name="First_AI")

# Остаточное обучение (переносим градиент с 3 слоя весов а 2)
ai.make_short_ways((2, 3))

# Устанавливаем коэффициенты
ai.alpha = 1e-3  # Скорость обучения
ai.batch_size = 10  # Размер батча

# Для оптимизатора Adam
ai.impulse1 = 0.9  # Обычно от 0.8 до 0.999
ai.impulse2 = 0.999  # Немного отличется от beta1

# Регуляризаторы
ai.l1 = 0.0  # L1 регуляризация
ai.l2 = 0.0  # L2 регуляризация

ai.main_act_func = ai.kit_act_func.tanh  # Просто функция активации
ai.end_act_func = ai.kit_act_func.tanh  # Функция активации для последнего слоя

ai.save_dir = "Saves AIs"  # В какую папку сохраняем ИИшки

# Обучаем (Например распозновать картинки)
for image in dataset:
    data = image.tolist()
    answer = [image.what_num]
    ai.learning(data, answer, squared_error=True)

# Есть также и Q-бучение (функции обновления таблицы в ai.kit_upd_q_table)
actions = ("left", "right", "up", "down")
ai.make_all_for_q_learning(actions, ai.kit_upd_q_table.standart,
                           gamma=0.5, epsilon=0.01, q_alpha=0.1)

state, reward = game.get_state_and_reward()
ai.q_learning(state, reward,
              learning_method=2.5,
              squared_error=False,
              use_Adam=True,
              recce_mode=False)

ai.make_mutations(0.05)  # 5% весов оказываются случайными числами
# Тут мы у ai_0 перемешиваем веса с ai_1, но ai_1 не трогаем
ai_0.genetic_crossing_with(ai_1)
```




# Подробности:

### Как инициализировать ИИшку:
```python
from My_AI import AI_ensemble, AI

ai = AI()
# Или
ensemble = AI_ensemble(5) # 5 —— количество ИИшек в ансамбле
```
> Ансамбль — это несколько ИИ в одной коробке, которые вместе принимают решение и за счёт этого вероятность случайной ошибки сильно понижается (ансамбль хорошо подходит для Q-обучения (это обучение с подкреплением; когда нету "правильного" и "неправильного" ответа, а только вознаграждение за какое-то выбранное действие))

> P.s. В качестве примера посмотрите на ИИ для змейки (в файле "learn snake.py")


### • Чтобы создать архитектуру:
```python
ai = AI(architecture=[3, 4, 4, 4, 3],
        add_bias_neuron=True,
        name="First_AI")

# Если используете ансамбль
ensemble = AI_ensemble(5, architecture=[3, 4, 4, 4, 3],
                       add_bias_neuron=True,
                       name="First_AI")
```
или

```python
ai = AI()
ai.create_weights([3, 4, 4, 4, 3], add_bias=True)
ai.name = "First_AI"
# Имя не обязательно указывать имя, но тогда при сохранении
# будет использовано случайное число вместо имени
```
Таким образом будет создана следущая архитектура:

![](Saves%20AIs/Пример%20архитектуры.png)

####  
### • Гаперпараметры:

```python
"""Прописывать или изменять ВСЕ гаперпараметры необязательно"""

ai.alpha = 1e-3  # Скорость обучения

ai.disabled_neurons = 0.0  # Какую долю нейронов отключаем
# (Это надо чтобы не возникало переобучение)

ai.batch_size = 1  # Сколько ответов усредняем, чтобы на них учиться
# (Ускоряет обучение, но в некоторых задачах лучше не использовать)

# Функция активации нейронов (крайне рекомендую оставить tanh если
# есть такая возможность, т.к. с ним ИИ работает в разы быстрее)
ai.main_act_func = ai.kit_act_func.tanh

# Функция активации для последнего слоя (аналогично, рекомендую оставить tanh)
# P.s. end_act_func может и отстутсвовать (т.е. можно установить None)
ai.end_act_func = ai.kit_act_func.tanh

# Коэффициенты имапульса для оптимизатора Adam
ai.impulse1 = 0.9
ai.impulse2 = 0.999
# Если не знаете что такое оптимизатор погуглите, очень интересно))))

# Коэффициенты для регуляризации весов 
# (Регуляризация — удержание весов около 0 (или же в диапазоне [-1; +1]) )
ai.l1 = 0.001  # НА сколько уменьшаем веса (устремляет веса прямо к 0)
ai.l2 = 0.01  # ВО сколько уменьшаем веса (удерживает веса около 0)
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
# ИИшка может только выбрать какое-то конкретное действие из возможных
all_possible_actions = ["left", "right", "up", "down"]

gamma = 0.4     # Коэффициент "доверия опыту" (для "сглаживания" Q-таблицы)
epsilon = 0.05  # Доля на случайных действий (чтобы ИИшка изучала окружающую среду)
q_alpha = 0.1   # Скорость обновления Q-таблицы (на самом деле оно почти ни на что не влияет) 

ai.make_all_for_q_learning(all_possible_actions,
                           ai.kit_upd_q_table.standart,
                           gamma=gamma, epsilon=epsilon, q_alpha=q_alpha)
# Функции обновления таблицы довольно сильно влияют на обучение

# Как и с обычным обучением, на вход подаём просто список чисел (состояние)
ai_state = [0, 1]  # Например координаты Агента

ai.q_learning(ai_state, reward_for_state, learning_method=1,
              recce_mode=False, squared_error=True)

# Какое решение приняла ИИшка при определённых данных (возвращает название действия)
predict = ai.q_predict(ai_state)
```
> recce_mode: Режим "исследования окружающей среды" (постоянно выбирать случайное действие)

> rounding: На сколько округляем состояние, для Q-таблицы (это надо чтобы классифицировать (сгруппировать) какой-то промежуток данных и на этой греппе данных обучать ИИ делать конкрентный выбор, и на дробных данных можно было обучаться)
> 
> rounding=0.1: 0.333333333 -> '0.3'; rounding=10: 123,456 -> 120

> Методы обучения (значение learning_method определяет) :
> - 1 : В качестве "правильного" ответа выбирается то, которое максимально вознаграждается, и
на место действия (которое приводит к лучшему ответу) ставиться максимальное значение функции
активации, а на остальные места минимум функции активации
P.s. Это неочень хорошо, т.к. игнорируются другие варианты, которые приносят либо столько же,
либо немного меньше вознаграждения (а вибирается только один "правильный"). _Но он хорошо подходит,
когда у вас в задаче имеется исключительно 1 правильный ответ, а "более" и "менее" правильных ответов быть не может_
> - 2 : Делаем ответы которые больше вознаграждаются, более "правильным". Дробная часть числа означает, в какую степень будем возводить "стремление у лучшим результатам" (НАПРИМЕР: 2.3 означает, что мы используем метод обучения 2 и возводим в степень 3 "стремление у лучшим результатам", а 2.345 означает, что степень будет равна 3.45 )

####  
### • Ну и конечно же сохранения и загрузки нейросети
```python
ai.save()  # Сохраниться под текущим именем
ai.save("First_AI")  # Сохранится под названием First_AI

ai.load("123")  # Загрузиться нейронка по названием "123" из папки сохранений
```
> Ещё можно выбрать свою папку для сохранений ИИшек
```python
# Всё будет сохраняться в папку SAVES рядышком с пакетом My_AI
ai.save_dir = "SAVES"
```

####  
### • Также ради прикола сделал и генетическое обучение
> Это когда перемешиваются веса у ИИшек

```python
# Тут мы у ai_0 перемешиваем веса с ai_1, а ai_1 не трогаем
ai_0.genetic_crossing_with(ai_1)
```
### • В дополнение к генетическому алгоритму создал возможность создание мутаций
> Какую-то долю весов заменяем на случайные числа от -1 до 1 

```python
ai.make_mutations(0.05)  # 5% весов оказываются случайными числами
```


####  
### Ещё можно подать данные на выходные нейроны и получить из входных (Зачам? я сам не знаю, но это может быть полезным)
```python
temp_data = ai.predict(data, reverse=True)
new_data = ai.predict(temp_data)
# new_data == data
```

####  
#### • Кстати, очень советую переводить входные числа в промежуток от -1 до 1 (или от 0 до 1) 
> Просто для ИИшки проще работать с числами от -1 до 1 (или от 0 до 1) чем с непонятными огромными значениями

```python
# Проще использовать normalize, но можно и tanh (или sigmoid)
ai.kit_act_funcs.normalize(data, 0, 1)
```

####  
## Что делает каждый файл:
- Запусти меня!.exe — экзешка для запуска предварительно обученной Змейки (залипательная)
- _internal — программа для экзешки
- AI for snake — скрипт для отображения игры обученной змейки
- AI for snake TF — точно такая же змейка, как и в "Запусти меня!.exe", но написана не на моей библиотеке, а на tensorflow с ручным Q-обучением (работает крайне медленно)
- AI for Tic-Tac-toe — нейронка для игры в крестики-нолики 
- Example use Q-learning — в реальном времени маленькая нейронка, только за счёт подаваемых координат, пытается дойти до правого нижнего угла не попадая на стены
- Games:
- - Code_Snake — Змейка
- - Game_for_Q_learning — среда для обучения "Example use Q-learning"
- - Tic_Tac_toe — крестики-нолики
- - Swarm_Game — заброшенная идея для реализации роевого интеллекта
- Learning_Snakes — при помощи optuna в нескольких параллельных процессах подбираются гиперпарамтры для Змейки
- MNIST_AI — нейронка для распознавания цифр mnist 
- learn snake — прога для обучения Змейки 
- easymyai:
- - Ai_funcs — все функции активации и функции обновления Q-таблицы
- - Ai_property — свойства для коэффициентов нейронки (типа "только от 0 до 1" или "только функция", ...)
- - Code_My_AI — сам код нейронки
- - Ensemble_AI — дополнение к Code_My_AI, но работает с несколькими нейронками сразу (сложный код, но работает)
- - README — документация на английском
- setup — настройки для публикации библиотеки

_+_ Каждый файл работает без багов и можно что угодно запустить и посмотреть на приколы

_fun fact_ — всего тут 2550 строк отлаженного и работающего кода

#  
#  
#  
###### _Удачи в ловле жучков_ (づ｡◕‿‿◕｡)づ
