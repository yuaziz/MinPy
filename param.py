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
    beta_update : choice of either polak_ribiere or fletcher_reeves OPTIONAL (Default : polar_ribiere)
    line_search : choice of either secant or newton_raphson OPTIONAL (Default : secant)
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

    allowed_funcs = ['rosenbrock', 'rastrigin', 'ackley', 'beale', 'booth', 'matyas', 'himmelblau', 'easom']

    if myfunc not in allowed_funcs:
        sys.exit('Function is not supported or check spelling of function parameter')

    #Call an error if user does not specify a solver
    if 'solver' in parameters:
        pass
    else:
        sys.exit('You must specify a solver in parameter file')

    #Check if user has specified x_int and y_int
    if 'x_initial' in parameters:
        myx_init = np.double(parameters.get('x_initial'))
        parameters.update({'x_initial' : myx_init })
    else:
        sys.exit('x_initial not specified in parameter file')
    if 'y_initial' in parameters:
        myy_init = np.double(parameters.get('y_initial'))
        parameters.update({'y_initial' : myy_init })
    else:
        sys.exit('y_initial not specified in parameter file')

    #Now do a check on x_int and y_int to see if it is within the search space of the function in question
    if myfunc == 'rosenbrock':
        if(check_in_range(np.double(-100.0),np.double(100.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_inital and/or y_initial do not lie in range[-100,100] for Rosenbrock function search')

    if myfunc == 'rastrigin':
        if(check_in_range(np.double(-5.12),np.double(5.12),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-5.12,5.12] for Rastrigin function search')

    if myfunc == 'ackley':
        if(check_in_range(np.double(-5.0),np.double(5.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-5,5] for Ackley function search')

    if myfunc == 'beale':
        if(check_in_range(np.double(-4.5),np.double(4.5),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-4.5,4.5] for Beale function search')

    if myfunc == 'booth':
        if(check_in_range(np.double(-10.0),np.double(10.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-10,10] for Booth function search')

    if myfunc == 'matyas':
        if(check_in_range(np.double(-10.0),np.double(10.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-10,10] for Matyas function search')

    if myfunc == 'himmelblau':
        if(check_in_range(np.double(-5.0),np.double(5.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-5,5] for Himmelblau function search')

    if myfunc == 'easom':
        if(check_in_range(np.double(-100.0),np.double(100.0),myx_init,myy_init)):
            pass
        else:
            sys.exit('x_initial and/or y_initial do not lie in range[-100,100] for Easom function search')

    
    #Set some default values if this has not been specified by user, these are all optional parameters
    if 'max_iter' in parameters:
         parameters.update({'max_iter' : np.int(parameters.get('max_iter'))})
    else:
        parameters.update({'max_iter' : np.int(10000)})

    if 'beta_update' in parameters:
        pass
    else:
        parameters.update({'beta_update' : 'fletcher_reeves'})

    if 'line_search' in parameters:
        pass
    else:
        parameters.update({'line_search' : 'newton_raphson'})

    if 'tolerance' in parameters:
        parameters.update({'tolerance' : np.double(parameters.get('tolerance'))})
    else:
        parameters.update({'tolerance' : np.double(1.0e-9)})


    return parameters





def check_in_range(lower,upper,x,y):
    #Checks if x and y are in range of lower and upper
    if(np.greater_equal(x,lower) and np.greater_equal(y,lower) and np.less_equal(x,upper) and np.less_equal(y,upper)):
        return True
    else:
        return False




    


