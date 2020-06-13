class Animal(object):
    def __init__(self, voice="######"):
        self.voice = voice

    def say(self):
        return self.voice


class Dog(Animal):
    def __init__(self, voice="wangwang"):
        self.voice = voice


class Cat(Animal):
    def __init__(self, voice="nyanya"):
        self.voice = voice


class Cow(Animal):
    def __init__(self, voice="mumu"):
        self.voice = voice


class AnimalFactory():
    def create_button(self, typ):
        # 将首字母大写
        targetclass = typ.capitalize()
        # 根据类型找到对应的类
        return globals()[targetclass]()


animal_factory = AnimalFactory()
animals = ['dog', 'cat', 'cow']
for animal in animals:
    print(animal_factory.create_button(animal).say())
