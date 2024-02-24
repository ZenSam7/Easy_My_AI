from typing import Callable


class MyProperties(object):
    @classmethod
    def get_property(
            cls, property_func: Callable, attr_name: str, doc: str
    ) -> property:
        """Создаём объект property со свойством property_func (которое мы выбираем из этого же класса)"""

        return property(
            fget=cls.property_getter(attr_name),
            fset=cls.property_setter(property_func, attr_name),
            fdel=cls.property_deleter,
            doc=doc,
        )

    @classmethod
    def property_getter(cls, attr_name: str) -> float:
        """Возвращаем атрибут из класса AI (замыкание нужно чтобы мы знали какой атрибут
        мы хотим получить, при этом не вызывая функцию)"""
        name = attr_name

        def getter(cls) -> float:
            nonlocal name

            # Если к нам попал AI_with_ensemble, то достаем из его атрибута атрибут
            if "ais" in cls.__dict__:
                return cls.__dict__["ais"][0].__dict__["_AI__" + name]

            if ("_AI__" + name) in cls.__dict__:
                return cls.__dict__["_AI__" + name]
            else:
                return cls.__dict__[name]

        return getter

    def property_setter(proiperty_func: Callable, attr_name: str) -> Callable:
        """По аналогии с property_getter, но при этом мы не достаём, а записываем в класс AI наш атрибут"""

        def property(cls, value: [float or Callable]):
            # Проверяем, подходит ли под выбранное свойство
            proiperty_func(value)

            # Если к нам попал AI_with_ensemble, то для него у каждой ИИшки
            # устанавливаем значение коэффициента или значение функции
            if "ais" in cls.__dict__:
                for ai in cls.__dict__["ais"]:
                    if isinstance(value, float) or isinstance(value, int):
                        # Если это коэффициент, то добавляеи "_AI__"
                        ai.__dict__["_AI__" + attr_name] = value

                    else:
                        ai.__dict__[attr_name] = value

            # Если просто работаем с AI
            else:
                # Если это коэффициент, то добавляеи "_AI__"
                if isinstance(value, float) or isinstance(value, int):
                    # Если ошибки (в proiperty_func(value)) не произошло, то перезаписываем атрибут
                    cls.__dict__["_AI__" + attr_name] = value

                else:
                    cls.__dict__[attr_name] = value

        return property

    def property_deleter(*args):
        print(f"Слышь, пёс, {args} нельзя удалять! Они вообще-то используются!!!")

    @staticmethod
    def from_1_to_0(value: float) -> Exception:
        """Свойство"""
        if not 0 <= value < 1:
            raise ImpossibleContinue(f"Число {value} не в диапазоне [0; 1)")

    @staticmethod
    def non_negative(value: float) -> Exception:
        """Свойство"""
        if value < 0:
            raise ImpossibleContinue(f"Число {value} меньше 0")

    @staticmethod
    def only_uint(value: float) -> Exception:
        """Свойство"""
        if not isinstance(value, int) or value < 0:
            raise ImpossibleContinue(f"Число {value} не целое и/или отрицательное")

    @staticmethod
    def just_pass(*args, **kwargs) -> Exception:
        """Свойство"""
        pass

    @staticmethod
    def is_bool(value: Callable) -> Exception:
        """Свойство"""
        if not isinstance(value, bool):
            raise ImpossibleContinue(f"{value} не boolean")
