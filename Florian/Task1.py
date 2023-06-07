# this file contains a collection of functions that are used to solve the first task of the project
# it consists of 5 taks for Steepest Descent and 5 tasks for Newton's Method, they should have an order of at least 3
# the functions are:
import math

import numpy as np

from NewtonsMethod import NewtonsMethod
from SteepestDescent import SteepestDescent


def f1(x):
    return ((1 / 4) * x ** 4) - ((17 / 3) * x ** 3) + (40 * x ** 2) - (100 * x)


def f1_str():
    return "((1 / 4) * x ** 4) - ((17 / 3) * x ** 3) + (40 * x ** 2) - (100 * x)"


f1.__str__ = f1_str


def df1(x):
    return ((x - 10) * (x - 5) * (x - 2))


def df1_str():
    return "((x - 10) * (x - 5) * (x - 2))"

def ddf1(x):
    return (x-10)*(x-5) + (x-10)*(x-2) + (x-5)*(x-2)

df1.__str__ = df1_str


# ((1)/(4)) x^(4) - 2x^(3) + ((11)/(2)) x^(2) - 6x
def f2(x):
    return ((1 / 4) * x ** 4) - (2 * x ** 3) + ((11 / 2) * x ** 2) - (6 * x)


def f2_str():
    return "((1 / 4) * x ** 4) - (2 * x ** 3) + ((11 / 2) * x ** 2) - (6 * x)"



f2.__str__ = f2_str

def df2(x):
    return (x-1) * (x-2) * (x-3)


def df2_str():
    return "(x-1) * (x-2) * (x-3)"

def ddf2(x):
    return (x-1)*(x-2) + (x-1)*(x-3) + (x-2)*(x-3)

df2.__str__ = df2_str

# ((1)/(6)) x^(6) - 4x^(5) + ((155)/(4)) x^(4) - ((580)/(3)) x^(3) + 522x^(2) - 720x
def f3(x):
    return ((1 / 6) * x ** 6) - (4 * x ** 5) + ((155 / 4) * x ** 4) - ((580 / 3) * x ** 3) + (522 * x ** 2) - (720 * x)


def f3_str():
    return "((1 / 6) * x ** 6) - (4 * x ** 5) + ((155 / 4) * x ** 4) - ((580 / 3) * x ** 3) + (522 * x ** 2) - (720 * x)"


f3.__str__ = f3_str


def df3(x):
    return (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)


def df3_str():
    return "(x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)"

def ddf3(x):
    return (x-2)*(x-3)*(x-4)*(x-5) + (x-2)*(x-3)*(x-4)*(x-6) + (x-2)*(x-3)*(x-5)*(x-6) + (x-2)*(x-4)*(x-5)*(x-6) + (x-3)*(x-4)*(x-5)*(x-6)

df3.__str__ = df3_str


#  sin(x+0.5)
def f4(x):
    return -math.sin(x+1)


def f4_str():
    return "sin(x + 0.5)"


f4.__str__ = f4_str


def df4(x):
    return -math.cos(x+1)


def df4_str():
    return "cos(x + 0.5)"

def ddf4(x):
    return math.sin(x+1)

df4.__str__ = df4_str


# ((1)/(5)) x^(5) - 1.08x^(3) + 2.25x
def f5(x):
    return ((1 / 5) * x ** 5) - (1.08 * x ** 3) + (2.25 * x)


def f5_str():
    return "((1 / 5) * x ** 5) - (1.08 * x ** 3) + (2.25 * x)"


f5.__str__ = f5_str


def df5(x):
    return (x+1) * (x-1) * (x+1.5) * (x-1.5)


def df5_str():
    return "(x+1) * (x-1) * (x+1.5) * (x-1.5) "

def ddf5(x):
    return (x+1)*(x-1)*(x+1.5) + (x+1)*(x-1)*(x-1.5) + (x+1)*(x+1.5)*(x-1.5) + (x-1)*(x+1.5)*(x-1.5)

df5.__str__ = df5_str


# -(4 x^(4)-4 x^(2) + x- 2)
def f6(x):
    return -(4 * x ** 4 - 4 * x ** 2 + x - 2)


def f6_str():
    return "-(4 * x ** 4 - 4 * x ** 2 + x - 2)"


f6.__str__ = f6_str


def df6(x):
    return -16 * x ** 3 + 8 * x - 1


def df6_str():
    return "-16 * x ** 3 + 8 * x - 1"

def ddf6(x):
    return -48*x**2 + 8

def ddf6_str():
    return "-48*x**2 + 8"




df6.__str__ = df6_str
ddf6.__str__ = ddf6_str


#  (x-2)^(6)-3 (x-2)^(4)+2 (x-2)^(2)+2
def f7(x):
    return (x - 2) ** 6 - 3 * (x - 2) ** 4 + 2 * (x - 2) ** 2 + 2


def f7_str():
    return "(x - 2) ** 6 - 3 * (x - 2) ** 4 + 2 * (x - 2) ** 2 + 2"


f7.__str__ = f7_str


def df7(x):
    return 6 * (x - 2) ** 5 - 12 * (x - 2) ** 3 + 4 * (x - 2)


def df7_str():
    return "6 * (x - 2) ** 5 - 12 * (x - 2) ** 3 + 4 * (x - 2) "

def ddf7(x):
    return 30*(x-2)**4 - 36*(x-2)**2 + 4

def ddf7_str():
    return "30*(x-2)**4 - 36*(x-2)**2 + 4 "

df7.__str__ = df7_str
ddf7.__str__ = ddf7_str


# x**4 - 4*x**3 + 5*x**2 - 2*x + 3
def f8(x):
    return x ** 4 - 4 * x ** 3 + 5 * x ** 2 - 2 * x + 3


def f8_str():
    return "x ** 4 - 4 * x ** 3 + 5 * x ** 2 - 2 * x + 3"


f8.__str__ = f8_str


def df8(x):
    return 4 * x ** 3 - 12 * x ** 2 + 10 * x - 2


def df8_str():
    return "4 * x ** 3 - 12 * x ** 2 + 10 * x - 2"

def ddf8(x):
    return 12*x**2 - 24*x + 10

def ddf8_str():
    return "12*x**2 - 24*x + 10"


df8.__str__ = df8_str
ddf8.__str__ = ddf8_str


# x^4 + 3x^3 + 2x^2 - x + 1
def f9(x):
    return (x ** 4) + (3 * x ** 3) + (2 * x ** 2) - (x) + 1


def f9_str():
    return "(x ** 4) + (3 * x ** 3) + (2 * x ** 2) - (x) + 1"


f9.__str__ = f9_str


def df9(x):
    return (4 * x ** 3) + (9 * x ** 2) + (4 * x) - 1


def df9_str():
    return "(4 * x ** 3) + (9 * x ** 2) + (4 * x) - 1"

def ddf9(x):
    return (12 * x ** 2) + (18 * x) + 4

def ddf9_str():
    return "(12 * x ** 2) + (18 * x) + 4"


df9.__str__ = df9_str
ddf9.__str__ = ddf9_str


# -((x+0.5)^(5)-10 (x+0.5)^(4)+35 (x+0.5)^(3)-50 (x+0.5)^(2)+25 (x+0.5)-2)
def f10(x):
    return -((x + 0.5) ** 5 - 10 * (x + 0.5) ** 4 + 35 * (x + 0.5) ** 3 - 50 * (x + 0.5) ** 2 + 25 * (x + 0.5) - 2)


def f10_str():
    return "-((x + 0.5) ** 5 - 10 * (x + 0.5) ** 4 + 35 * (x + 0.5) ** 3 - 50 * (x + 0.5) ** 2 + 25 * (x + 0.5) - 2)"


f10.__str__ = f10_str

# -5x^(4) + 30x^(3) - 52.5x^(2) + 22.5x + 3.44
def df10(x):
    return -5 * x ** 4 + 30 * x ** 3 - 52.5 * x ** 2 + 22.5 * x + 3.44


def df10_str():
    return "-5 * x ** 4 + 30 * x ** 3 - 52.5 * x ** 2 + 22.5 * x + 3.44"

def ddf10(x):
    return -20*x**3 + 90*x**2 - 105*x + 22.5

def ddf10_str():
    return "-20*x**3 + 90*x**2 - 105*x + 22.5 "


df10.__str__ = df10_str
ddf10.__str__ = ddf10_str


class Task1:
    def __init__(self):
        self.functions = [f1, f2, f4, f8, f9]
        self.gradients = [df1, df2, df4, df8, df9]
        self.solutions = [[2, 10], [1, 3], [0.57079632], [0.2928932188134, 1.707106781187], [0.1753905296791, -1.425] ]
        self.hessians = [ddf1, ddf2, ddf4, ddf8, ddf9]
    def getFunction(self, i):
        return self.functions[i]

    def getAllFunctions(self):
        return self.functions

    def getGradient(self, i):
        return self.gradients[i]

    def getAllGradients(self):
        return self.gradients

    def getSolution(self, i):
        return self.solutions[i]

    def getAllSolutions(self):
        return self.solutions

    def getHessian(self, i):
        return self.hessians[i]



if __name__ == '__main__':
    task1 = Task1()

    for i in range(0, 5):
        f = task1.getFunction(i)
        df = task1.getGradient(i)
        ddf = task1.getHessian(i)

        print("Function {} : The function is defined as {}".format(i + 1, f.__str__()))
        print("Gradient {} : The gradient is defined as {}".format(i + 1, df.__str__()))
        print("Solution {} : The solutions are {}".format(i + 1, task1.getSolution(i)))
        sd = SteepestDescent(f, df, np.array([0]), 0.5, 1e-6, 100000)
        result, results, iterations, _ = sd.run()
        print("Solution from Steepest Descent: {}".format(results[-1]))
        print("Gradient norm Value: {}".format(np.linalg.norm(df(results[-1]))))
        print("Number of Iterates: {}".format(iterations))
        closest_solution = task1.getSolution(i)[np.argmin([np.linalg.norm(results[-1] - sol) for sol in task1.getSolution(i)])]
        print("Distance from solution: {}".format(np.linalg.norm(closest_solution - results[-1])))

        nm = NewtonsMethod(1, f, df, ddf, np.array([0]), 1e-6, 10000)
        results, iterations = nm.run()
        print("Solution from Newton's Method: {}".format(results[-1]))
        print("Gradient norm Value: {}".format(np.linalg.norm(df(results[-1]))))
        print("Number of Iterates: {}".format(iterations))
        closest_solution = task1.getSolution(i)[np.argmin([np.linalg.norm(results[-1] - sol) for sol in task1.getSolution(i)])]
        print("Distance from solution: {}".format(np.linalg.norm(closest_solution - results[-1])))
        print("#########################################################################################")

