#This module will house all functions one of which we would need to call periodically

#Create a class and dynamically call with specified string defined in parameter dict

#The x and y inputs will be class attributes
#The solve for method takes a string and can call other functions

import numpy as np

class OptFunc:

    def __init__(self, x, y):
        self.x = np.double(x)
        self.y = np.double(y)

    def calc_rosenbrock(self):
        x = self.x
        y = self.y
        xsqr = np.power(x,2.0)
        funcval = np.power(1.0-x, 2.0) + 100.0*( np.power(y-xsqr, 2.0) )
        return funcval

    def calc_rastrigin(self):
        x = self.x
        y = self.y
        xsqr = np.power(x,2.0)
        ysqr = np.power(y,2.0)
        funcval = 20 + (xsqr - 20*np.cos(2*np.pi*x)) + (ysqr - 20*np.cos(2*np.pi*x))
        return funcval

    def calc_ackley(self):
        x = self.x
        y = self.y
        xsqr = np.power(x,2.0)
        ysqr = np.power(y,2.0)
        funcval = -20*np.exp(-np.double(0.2)*np.sqrt(np.double(0.5)*(xsqr+ysqr)))\
            -np.exp(np.double(0.5)*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.e + 20
        return funcval

    def calc_beale(self):
        x = self.x
        y = self.y
        xsqr = np.power(x,2.0)
        ysqr = np.power(y,2.0)
        ycub = np.power(y,3.0)
        funcval = np.power(np.double(1.5)-x+(x*y), 2.0) + np.power(np.double(2.25)-x+(x*ysqr), 2.0)\
            + np.power(np.double(2.625)-x+(x*ycub), 2.0)
        return funcval

    def calc_booth(self):
        x = self.x
        y = self.y
        funcval = np.power(x+(2*y)-7, 2.0) + np.power((2*x)+y-5, 2.0)
        return funcval

    def calc_matyas(self):
        x = self.x
        y = self.y
        xsqr = np.power(x,2.0)
        ysqr = np.power(y,2.0)
        funcval = np.double(0.26)*(xsqr+ysqr) - np.double(0.48)*x*y
        return funcval

    def calc_himmelblau(self):
        x = self.x
        y = self.y
        xsqr = np.power(x,2.0)
        ysqr = np.power(y,2.0)
        funcval = np.power(xsqr+y-11,2.0) + np.power(x+ysqr-7,2.0)
        return funcval

    def calc_easom(self):
        x = self.x
        y = self.y
        funcval = -np.cos(x)*np.cos(y)*np.exp(-(np.power(x-np.pi,2.0)+np.power(y-np.pi,2.0)))
        return funcval


    #This is what we will call to get the output from all other functions
    def solve_for(self, name: str):
        do = f"calc_{name}"
        if hasattr(self,do) and callable(getattr(self,do)):
            func = getattr(self, do)
            function_out = func()
        return function_out
