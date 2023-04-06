#!/usr/bin/env python
""" 

MinPy

Apply nonlinear conjugate gradient or BFGS to various optimisation
functions with the aim of finding a global minimum.


Developed by Yusuf Aziz


This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

"""

import sys
from param import init_params
from nlcg import nlcg_secant, nlcg_newton_raphson
#from bfgs import bfgs_solver
from output import write_output

def main():

    #Abort if user specifies multiple/no parameter files
    #Not a robust check as the user could specify other runtime options which can fool this
    if len(sys.argv) != 2:
        sys.exit('Require a single parameter file')

    file_name = sys.argv[1]

    #Read in parameters and intialise
    parameters = init_params(file_name)

    #Call nlcg solver based on parameter input, only support certain combinations...
    beta_update = parameters.get('beta_update')
    line_search = parameters.get('line_search')
    solver      = parameters.get('solver')

    if(solver=='nlcg' and line_search=='secant' and beta_update=='polak_ribiere' ):
        success, x, y, history_array, num_iter = nlcg_secant(parameters)
    elif(solver=='nlcg' and line_search=='newton_raphson' and beta_update=='fletcher_reeves'):
        success, x, y, history_array, num_iter = nlcg_newton_raphson(parameters)
    elif(solver=='bfgs'):
        success, x, y, history_array, num_iter = bfgs_solver(parameters)
    else:
        sys.exit('Apologies, this combination of line search and beta_update is not supported yet')

    #Write output
    write_output(success, x, y, history_array, num_iter)

    #Plot if required


if __name__ == '__main__':
    main()

