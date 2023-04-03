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
        xsqr = np.power(self.x,2.0)
        funcval = np.power(1.0-self.x,2.0) + 100.0*(np.power(self.y-xsqr,2.0))
        return funcval

    #This is what we will call to get the output from all other functions
    def solve_for(self, name: str):
        do = f"calc_{name}"
        if hasattr(self,do) and callable(getattr(self,do)):
            func = getattr(self, do)
            function_out = func()
        return function_out
