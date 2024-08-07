import numpy as np
import sys

def init_params(filename):
    """
    Given a supplied parameter file, this module intialises all variables associated with the upcoming computation.
    It also does some sanity checks as to whether the intial starting points lie within the search domain with respect
    to the function of interest.

    Parameters:
    ------------
    solver      : nlcg or bfgs REQUIRED
    function    : the function in question the user must specify in parameter file REQUIRED
    x_initial   : starting point for x REQUIRED
    y_initial   : starting point for y REQUIRED
    beta_update : choice of four formulas for beta OPTIONAL (Default : fletcher_reeves)
    line_search : choice of either secant or newton_raphson OPTIONAL (Default : newton_raphson)
    max_iter    : maximum number of iterations for the solver to find global minimum OPTIONAL (Default : 10000 )
    tolerance   : tolerance factor to which x and y are determined OPTIONAL (Default : 1.0e-9)
    """

    #Empty Dictionary
    myparams = {}

    with open(filename) as f:
        file = f.readlines()

    for line in file:
        #ignore comments in line
        if not line.startswith('#'):
            #parameters should be supplied with equals sign
            key_value = line.split('=')
            if len(key_value) == 2:
                myparams[key_value[0].strip()] = key_value[1].strip()

    #convert everything to lowercase
    myparams= dict((k.lower() if isinstance(k,str) else k, v.lower() if isinstance(v,str) else v) for k, v in myparams.items())
    
    #Now we need to check these params and intialise some defaults in case the user does not specify
    parameters = check_params(myparams)

    return parameters


def check_params(parameters):
    #Check if user has specified a function and if it makes sense
    if 'function' in parameters:
        myfunc = parameters.get('function')
    else:
        sys.exit('Function not specified in parameter file')

    allowed_funcs = ['rosenbrock', 'rastrigin', 'ackley', 'beale', 'booth', 'matyas',\
            'himmelblau', 'easom', 'three_hump_camel']

    if myfunc not in allowed_funcs:
        sys.exit('Function is not supported or check spelling of function parameter')

    #Call an error if user does not specify a solver
    if 'solver' in parameters:
        my_solver = parameters.get('solver')
    else:
        sys.exit('You must specify a solver in parameter file')

    allowed_solvers = ['nlcg', 'bfgs']

    if my_solver not in allowed_solvers:
        sys.exit('Solver is not supported or check spelling of solver parameter')

    #Check if user has specified a beta_update and that it is valid, otherwise set a default
    if 'beta_update' in parameters:
        my_beta_update = parameters.get('beta_update')
    else:
        parameters.update({'beta_update' : 'fletcher_reeves'})

    allowed_beta_updates = ['fletcher_reeves', 'polak_ribiere', 'hestenes_stiefel', 'dai_yuan']

    if my_beta_update not in allowed_beta_updates:
        sys.exit('Beta_update is not supported or check spelling of beta_update parameter ')

    #Check if user has specified a valid line search method, otherwise set a default
    if 'line_search' in parameters:
        my_line_search = parameters.get('line_search')
    else:
        parameters.update({'line_search' : 'newton_raphson'})

    allowed_line_searches = ['secant', 'newton_raphson']

    if my_line_search not in allowed_line_searches:
        sys.exit('Line_search is not supported or check spelling of line_search parameter')

    #Check if user has specified x_int and y_int
    if 'x_initial' in parameters:
        myx_init = np.float64(parameters.get('x_initial'))
        parameters.update({'x_initial' : myx_init })
    else:
        sys.exit('x_initial not specified in parameter file')
    if 'y_initial' in parameters:
        myy_init = np.float64(parameters.get('y_initial'))
        parameters.update({'y_initial' : myy_init })
    else:
        sys.exit('y_initial not specified in parameter file')

    #Now do a check on x_int and y_int to see if it is within the search space of the function in question
    if myfunc == 'rosenbrock':
        if(check_in_range(np.float64(-100.0),np.float64(100.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_inital and/or y_initial do not lie in range[-100,100] for Rosenbrock function search')

    if myfunc == 'rastrigin':
        if(check_in_range(np.float64(-5.12),np.float64(5.12),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-5.12,5.12] for Rastrigin function search')

    if myfunc == 'ackley':
        if(check_in_range(np.float64(-5.0),np.float64(5.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-5,5] for Ackley function search')

    if myfunc == 'beale':
        if(check_in_range(np.float64(-4.5),np.float64(4.5),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-4.5,4.5] for Beale function search')

    if myfunc == 'booth':
        if(check_in_range(np.float64(-10.0),np.float64(10.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-10,10] for Booth function search')

    if myfunc == 'matyas':
        if(check_in_range(np.float64(-10.0),np.float64(10.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-10,10] for Matyas function search')

    if myfunc == 'himmelblau':
        if(check_in_range(np.float64(-5.0),np.float64(5.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-5,5] for Himmelblau function search')

    if myfunc == 'easom':
        if(check_in_range(np.float64(-100.0),np.float64(100.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-100,100] for Easom function search')

    if myfunc == 'three_hump_camel':
        if(check_in_range(np.float64(-5.0),np.float64(5.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_inital and/or y_initial do not lie in range[-5,5] for Three Hump Camel function search')

    #Set some default values if this has not been specified by user, these are all optional parameters
    if 'max_iter' in parameters:
         parameters.update({'max_iter' : np.int32(parameters.get('max_iter'))})
    else:
        parameters.update({'max_iter' : np.int32(10000)})

    if 'tolerance' in parameters:
        parameters.update({'tolerance' : np.float64(parameters.get('tolerance'))})
    else:
        parameters.update({'tolerance' : np.float64(1.0e-9)})


    return parameters






def check_in_range(lower,upper,x,y):
    #Checks if x and y are in range of lower and upper
    if(np.greater_equal(x,lower) and np.greater_equal(y,lower) and np.less_equal(x,upper) and np.less_equal(y,upper)):
        return True
    else:
        return False




    


