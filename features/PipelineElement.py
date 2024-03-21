from re import error
import sys

class PipelineElement():

    def __init__(self) -> None:
        self.PARENT = False
        self.children = []

    # def __getattr__(self, name, *args):
    #     # print(*args)
    #     # try:
    #     #     module = __import__('features')
    #     #     class_ = getattr(module, name)
    #     #     return class_(*args)
    #     # except NameError:
    #     #     print('error')
    #     def function(*args):
    #         try:
    #             module = __import__('features')
    #             class_ = getattr(module, name)
    #             return class_(*args)
    #         except NameError:
    #             print(error)
    #     return function

    def add_child(self, object):
        # print('add : ', self, object)
        object.PARENT = self
        for child in self.children:
            # print('compare : ', child, object)
            if object == child:
                return child
        self.children.append(object)
        # print('added :', self.children)
        return object

    def __eq__(self, other) :
        if self.__class__.__name__ != other.__class__.__name__:
            return False

        for key, value in self.__dict__.items():
            if key.isupper():
                if key in other.__dict__:
                    if value != other.__dict__[key]:
                        return False
                else:
                    return False

        for key, value in other.__dict__.items():
            if key.isupper():
                if key in self.__dict__:
                    if value != self.__dict__[key]:
                        return False
                else:
                    return False
        print('eq : ', self, other)
        return True