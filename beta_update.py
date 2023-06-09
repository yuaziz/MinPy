#This module will house all functions we could use to update the beta parameter

#Similarly to the optmisation functions module, we create a class and dynamically call dependent on
# a string which is user defined and has been checked by the parameters module.

import numpy as np

class BetaUpd:

    def __init__(self, res_x, res_y, res_x_old, res_y_old, px_step_old, py_step_old):
        self.res_x       = np.double(res_x)
        self.res_y       = np.double(res_y)
        self.res_x_old   = np.double(res_x_old)
        self.res_y_old   = np.double(res_y_old)
        self.px_step_old = np.double(px_step_old)
        self.py_step_old = np.double(py_step_old)


    def beta_update_fletcher_reeves(self):

        beta_upper = np.power(self.res_x,2.0) + np.power(self.res_y,2.0)
        beta_lower = np.power(self.res_x_old,2.0) + np.power(self.res_y_old,2.0)
        beta = beta_upper / beta_lower

        return beta


    def beta_update_polak_ribiere(self):

        beta_upper = self.res_x*(self.res_x - self.res_x_old) + self.res_y*(self.res_y - self.res_y_old)
        beta_lower = np.power(self.res_x_old,2.0) + np.power(self.res_y_old,2.0)
        beta = beta_upper / beta_lower

        return beta


    def beta_update_hestenes_stiefel(self):

        beta_upper = self.res_x*(self.res_x - self.res_x_old) + self.res_y*(self.res_y - self.res_y_old)
        beta_lower = -self.px_step_old*(self.res_x - self.res_x_old) - self.py_step_old*(self.res_y - self.res_y_old)
        beta = beta_upper / beta_lower

        return beta


    def beta_update_dai_yuan(self):

        beta_upper = np.power(self.res_x,2.0) + np.power(self.res_y,2.0)
        beta_lower = -self.px_step_old*(self.res_x - self.res_x_old) - self.py_step_old*(self.res_y - self.res_y_old)
        beta = beta_upper / beta_lower

        return beta


    def calc_beta_update(self, name: str):
        do = f"beta_update_{name}"
        if hasattr(self,do) and callable(getattr(self,do)):
            func = getattr(self, do)
            function_out = func()
        return function_out



