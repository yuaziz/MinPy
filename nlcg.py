#House the nonlinear conjugate gradient solver here

import numpy as np
from optimisation_functions import OptFunc

def nlcg_secant(parameters):
    """Perform nonlinear conjugate gradient with Secant, polak-ribiere utilised exclusively (for now) for the beta update

    Keyword Arguments:
    parameters -- a dicitonary which is checked by init_params before used here.

    Output:
    success        -- True if x and y are converged solutions and have satisfied tolerance
    x              -- global minimum of the x component
    y              -- global minimum of the y component
    output_history -- a numpy array containing the history of all components saved at every iteration
    k              -- Number of iterations required for converged solution, otherwise equal to max_iter if not converged
    """
    #Begin by setting the variables from a checked parameter dict
    x = parameters.get('x_initial')
    y = parameters.get('y_initial')
    line_search = parameters.get('line_search')
    function = parameters.get('function')
    max_iter = parameters.get('max_iter')
    mytolerance = parameters.get('tolerance')
    xconv = False
    yconv = False
    success = False
    SIGMA_INIT = 0.01 #set as a constant SECANT method step parameter

    #Intialise array which would hold the computed solutions, saved at every iteration
    solution_history = np.empty((max_iter,2), dtype=np.float64)

    for k in range(0,max_iter):

        if (k==0):
            #Inital search step is equal to intial x and y
            px_step = x
            py_step = y

            #Compute derivatives
            fx_prime = differentiate_x(x,y,function)
            fy_prime = differentiate_y(x,y,function)

            #Compute the residuals
            res_x = -fx_prime
            res_y = -fy_prime

            #Compute beta (scalar for search direction)
            beta_upper = res_x*(res_x - px_step) + res_y*(res_y-py_step)
            beta_lower = np.power(res_x,2.0) + np.power(res_y,2.0)
            beta = beta_upper / beta_lower

            #Reset Search Direction if beta is negative for Polak-Ribiere
            if np.less(beta, np.double(0.0)):
                px_step = res_x
                py_step = res_y
            else:
                px_step = res_x + beta*px_step
                py_step = res_y + beta*py_step

            #Require derivatives for secant line search
            fpsec_x = differentiate_x(x+(SIGMA_INIT*px_step), y, function)
            fpsec_y = differentiate_y(x, y+(SIGMA_INIT*py_step), function)

            #Carry out line-search (compute alpha)
            eta_prev = fpsec_x*px_step + fpsec_y*py_step

            eta = fx_prime*px_step + fy_prime*py_step

            #Initialise alpha and update
            alpha = -SIGMA_INIT

            alpha *= (eta/(eta_prev-eta))

            #Update solutions
            x += alpha*px_step
            y += alpha*py_step

            #Save to history array
            solution_history[k,0] = x
            solution_history[k,1] = y

            #To be used for k>0 iterations
            sigma = -alpha

        else:

            #Set search direction to previous residual
            px_step = res_x
            py_step = res_y

            #Compute derivatives
            fx_prime = differentiate_x(x,y,function)
            fy_prime = differentiate_y(x,y,function)

            #Compute the residuals for SD
            res_x = -fx_prime
            res_y = -fy_prime

            #Compute beta (scalar for search direction)
            beta_upper = res_x*(res_x - px_step) + res_y*(res_y-py_step)
            beta_lower = np.power(res_x,2.0) + np.power(res_y,2.0)
            beta = beta_upper / beta_lower

            #Reset Search Direction if beta is negative for Polak-Ribiere
            if np.less(beta, np.double(0.0)):
                px_step = res_x
                py_step = res_y
            else:
                px_step = res_x + beta*px_step
                py_step = res_y + beta*py_step

            #Now do line search with sigma
            #Require derivatives for secant line search
            fpsec_x = differentiate_x(x+(sigma*px_step), y, function)
            fpsec_y = differentiate_y(x, y+(sigma*py_step), function)

            #Carry out line-search (compute alpha)
            eta_prev = fpsec_x*px_step + fpsec_y*py_step

            eta = fx_prime*px_step + fy_prime*py_step

            alpha *= (eta/(eta_prev-eta))

            #Update sigma
            sigma = -alpha

            #Store old solutions for comparison later
            x_old = x
            y_old = y

            #Update x and y
            x += alpha*px_step
            y += alpha*py_step

            #Save to history array
            solution_history[k,0] = x
            solution_history[k,1] = y

            #tolerance check both x and y must be sufficiently converged
            if(np.isclose(x_old, x, rtol=mytolerance)):
                xconv = True

            if(np.isclose(y_old, y, rtol=mytolerance)):
                yconv = True

            if (xconv and yconv):
                success = True
                output_history = solution_history[0:k]
                break

    return success, x, y, output_history, k


#Uses O(H^6) accurate central difference to approximate the derivative wrt x
def differentiate_x(x,y,myfunc):
    
    H=np.double(0.01) #set as constant for now

    #initialise coeff array for O(H^6)
    coeff = np.array([-45, -9, 1], dtype=np.float64)

    #intialise result to zero
    partial_x = np.double(0.0)

    for i in range(0,3):
        #Need func for both +- h
        func_plus = OptFunc(x+(i+1.0)*H, y)
        func_minus = OptFunc(x-(i+1.0)*H, y)
        #Evaluate function at the point
        f_plus = func_plus.solve_for(myfunc)
        f_minus = func_minus.solve_for(myfunc)
        #Add to partial_x
        partial_x += coeff[i]*(f_plus - f_minus)

    #Divide by 60 (from central difference formula) and H
    partial_x /= (60.0*H)

    return partial_x



#Uses O(H^6) accurate central difference to approximate the derivative wrt y
def differentiate_y(x,y,myfunc):
    
    H=np.double(0.01) #set as constant for now

    #initialise coeff array for O(H^6)
    coeff = np.array([-45, -9, 1], dtype=np.float64)

    #intialise result to zero
    partial_y = np.double(0.0)

    for i in range(0,3):
        #Need func for both +- h
        func_plus = OptFunc(x, y+(i+1.0)*H)
        func_minus = OptFunc(x, y-(i+1.0)*H)
        #Evaluate function at the point
        f_plus = func_plus.solve_for(myfunc)
        f_minus = func_minus.solve_for(myfunc)
        #Add to partial_y
        partial_y += coeff[i]*(f_plus - f_minus)

    #Divide by 60 (from central difference formula) and H
    partial_y /= (60.0*H)

    return partial_y

#Uses (O^(H^2*H^2)) accurate mixed derivative formula to approximate the mixed derivative,
#this is symmetric, as in dxdy = dydx
def differentiate_xy(x,y,myfunc):

    H = np.double(0.01) #use same step size to treat both variables

    #Need func for all combinatoric factors of +- h
    func_plus_plus   = OptFunc(x+H, y+H)
    func_plus_minus  = OptFunc(x+H, y-H)
    func_minus_plus  = OptFunc(x-H, y+H)
    func_minus_minus = OptFunc(x-H, y-H)

    #Evaluate functions at these points
    f_plus_plus   = func_plus_plus.solve_for(myfunc)
    f_plus_minus  = func_plus_minus.solve_for(myfunc)
    f_minus_plus  = func_minus_plus.solve_for(myfunc)
    f_minus_minus = func_minus_minus.solve_for(myfunc)

    partial_xy = np.double(0.0)

    #Evaluate mixed derivative
    partial_xy = f_plus_plus + f_plus_minus + f_minus_plus + f_minus_minus

    partial_xy /= (4*H*H)

    return partial_xy






















 


