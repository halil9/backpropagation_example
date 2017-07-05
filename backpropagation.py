# -*- coding: utf-8 -*-
from numpy import exp, array, random, dot

class backprop():
    def __init__(self):
        random.seed(1)
        self.agirlik=2*random.random((3,1))-1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_turev(self, x):
        return x * (1 - x)

    def egit(self, gercek_inputs, gercek_outputs,_iterations):
        for iteration in range(10000):
            output = self.beklenen(gercek_inputs)
            error = gercek_outputs - output
            ayar = dot(gercek_inputs.T, error * self.__sigmoid_turev(output))
            self.agirlik += ayar

    def beklenen(self, inputs):
     return self.__sigmoid(dot(inputs, self.agirlik))
if __name__ == "__main__":
    back_propagation = backprop()
    print "Rastgele agirliklar: "
    print back_propagation.agirlik

    gercek_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    gercek_outputs = array([[0,1,1,0]]).T
    back_propagation.egit(gercek_inputs, gercek_outputs, 10000)

    print "eğitimden sonraki ağırklar: "

    print back_propagation.agirlik

    print "beklenen çözüm [1, 0, 0] -> ?: "
    print back_propagation.beklenen(array([1, 0, 0]))
