#House the nonlinear conjugate gradient solver here

import numpy as np
from optimisation_functions import OptFunc
from beta_update import BetaUpd

def nlcg_secant(parameters):
    """Perform nonlinear conjugate gradient with Secant, polak-ribiere utilised exclusively (for now) for the beta update

    Keyword Arguments:
    parameters -- a dicitonary which is checked by init_params before used here.

    Returns:
    success        -- True if x and y are converged solutions and have satisfied tolerance
    x              -- global minimum of the x component
    y              -- global minimum of the y component
    output_history -- a numpy array containing the history of all components saved at every iteration
    k              -- Number of iterations required for converged solution, otherwise equal to max_iter-1 if not converged
    """
    #Begin by setting the variables from a checked parameter dict
    x = parameters.get('x_initial')
    y = parameters.get('y_initial')
    line_search = parameters.get('line_search')
    function = parameters.get('function')
    max_iter = parameters.get('max_iter')
    mytolerance = parameters.get('tolerance')
    beta_update = parameters.get('beta_update')
    xconv = False
    yconv = False
    success = False
    SIGMA_INIT = 0.01 #set as a constant SECANT method step parameter

    #Step size for numerical derivative evaluation
    STEP_SIZE=np.double(0.01) #set as constant for now, could be user defined

    # sd_count = 0 #Can use to check how many times a SD reset is needed

    #Dont know how big our solution_history will be so use a list
    #Intialise list with starting x and y, save computed solutions at every iteration
    solution_history = [[x,y]]

    for k in range(0,max_iter):


        #The first iteration will correspond to a Steepest Descent step
        if (k==0):
            #Compute derivatives
            fx_prime = differentiate_x(x,y,function,STEP_SIZE)
            fy_prime = differentiate_y(x,y,function,STEP_SIZE)

            #Compute the residuals
            res_x = -fx_prime
            res_y = -fy_prime

            #line search using SD step direction,
            px_step = res_x
            py_step = res_y
            #Require derivatives for secant line search
            fpsec_x = differentiate_x(x+(SIGMA_INIT*px_step), y, function,STEP_SIZE)
            fpsec_y = differentiate_y(x, y+(SIGMA_INIT*py_step), function,STEP_SIZE)

            #Carry out line-search (compute alpha)
            eta_prev = fpsec_x*res_x + fpsec_y*res_y

            eta = fx_prime*res_x + fy_prime*res_y

            #Initialise alpha and update
            alpha = -SIGMA_INIT

            alpha *= (eta/(eta_prev-eta))

            #Update solutions using SD
            x += alpha*res_x
            y += alpha*res_y

            #Save to solution history
            solution_history.append([x,y])
            
            #To be used for k>0 iterations
            sigma = -alpha

        else:

            #Save old_residuals
            res_x_old = res_x
            res_y_old = res_y

            #Save old search directions
            px_step_old = px_step
            py_step_old = py_step

            #Compute derivatives
            fx_prime = differentiate_x(x,y,function,STEP_SIZE)
            fy_prime = differentiate_y(x,y,function,STEP_SIZE)

            #Compute the residuals
            res_x = -fx_prime
            res_y = -fy_prime

            #Compute beta (scalar for search direction)
            beta_init = BetaUpd(res_x, res_y, res_x_old, res_y_old, px_step_old, py_step_old)
            beta = beta_init.calc_beta_update(beta_update)

            #Reset Search Direction if beta is negative for Polak-Ribiere
            if np.less_equal(beta, np.double(0.0)):
                px_step = res_x
                py_step = res_y
                # sd_count +=1 #for interest
            else:
                px_step = res_x + beta*px_step
                py_step = res_y + beta*py_step

            #Now do line search with sigma
            #Require derivatives for secant line search
            fpsec_x = differentiate_x(x+(sigma*px_step), y, function,STEP_SIZE)
            fpsec_y = differentiate_y(x, y+(sigma*py_step), function,STEP_SIZE)

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

            solution_history.append([x,y])

            #tolerance check both x and y must be sufficiently converged
            if(np.isclose(x_old, x, rtol=mytolerance)):
                xconv = True

            if(np.isclose(y_old, y, rtol=mytolerance)):
                yconv = True

            if (xconv and yconv):
                # print('Number of SD resets: ', sd_count) #for interest
                success = True
                output_history = solution_history
                break

    #After the for loop, if success is false we still need to write a history array
    if(xconv==False or yconv==False):
        success = False
        output_history = solution_history

    return success, x, y, output_history, k+1



def nlcg_newton_raphson(parameters):
    """Perform nonlinear conjugate gradient with Newton-Raphson, Fletcher-Reeves utilised exclusively (for now) for the beta update

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
    beta_update = parameters.get('beta_update')
    xconv = False
    yconv = False
    success = False

    #Step size for numerical derivative evaluation
    STEP_SIZE=np.double(0.01) #set as constant for now, could be user defined

    #Dont know how big our solution_history will be so use a list
    #Intialise list with starting x and y, save computed solutions at every iteration
    solution_history = [[x,y]]

    #Intialise x_old and y_old here to something very high so that algorithm doesnt converge immediately
    x_old = np.float64(1e30)
    y_old = np.float64(1e30)

    for k in range(0,max_iter):
        
        #First iteration is a SD step
        if(k==0):
            #Compute derivatives
            fx_prime = differentiate_x(x,y,function,STEP_SIZE)
            fy_prime = differentiate_y(x,y,function,STEP_SIZE)

            #Compute the residuals
            res_x = -fx_prime
            res_y = -fy_prime

            #Set search direction to residual
            px_step = res_x 
            py_step = res_y


        #We can compute the numerator of alpha now
        alpha_upper = -(fx_prime*px_step + fy_prime*py_step)

        #-------------------------------------------------------------------------#
        # Store hessian matrix as a 2X2 array which is symmetric (dxdy=dydx):     #
        # hessian = ([dxdx, dxdy],                                                #
        #            [dxdy, dydy])                                                #
        #                                                                         #
        # After obtaining alpha, will switch back to updating x and y             #
        #-------------------------------------------------------------------------#

        #Initialise array
        #p is a column vector
        p_arr = np.zeros((2, 1))
        p_arr[0] = px_step
        p_arr[1] = py_step

        #Define its transpose
        p_arr_t = np.transpose(p_arr)

        #Now define the hessian
        hess = np.zeros((2,2))
        #Now need to populate elements by computing all necessary derivatives, note dxdy=dydx so 
        #we only need to compute three terms
        dxdx = differentiate_xx(x,y,function,STEP_SIZE)
        dydy = differentiate_yy(x,y,function,STEP_SIZE)
        dxdy = differentiate_xy(x,y,function,STEP_SIZE)
        dydx = dxdy
        #Populate hessian
        hess[0,0] = dxdx
        hess[0,1] = dxdy
        hess[1,0] = dydx
        hess[1,1] = dydy

        #Need to compute pT hess p, so do in stages
        hess_p = np.matmul(hess,p_arr)

        pt_hess_p = np.matmul(p_arr_t,hess_p)

        #Now we can compute the denominator for alpha
        alpha_lower = np.double(pt_hess_p[0])

        #Compute alpha
        alpha = alpha_upper / alpha_lower

        #Store old solutions for comparison later
        if not(k==0):
            x_old = x
            y_old = y
        
        #Update solutions
        x += alpha*px_step
        y += alpha*py_step

        #Save to solution history
        solution_history.append([x,y])

        #Compute derivatives
        fx_prime = differentiate_x(x,y,function,STEP_SIZE)
        fy_prime = differentiate_y(x,y,function,STEP_SIZE)

        #Save old_residuals
        res_x_old = res_x
        res_y_old = res_y

        #Save old step directions
        px_step_old = px_step
        py_step_old = py_step

        #Set the new residuals
        res_x = -fx_prime
        res_y = -fy_prime
        
        #Compute beta (scalar for search direction)
        beta_init = BetaUpd(res_x, res_y, res_x_old, res_y_old, px_step_old, py_step_old)
        beta = beta_init.calc_beta_update(beta_update)

        #Update search direction
        px_step = res_x + beta*px_step
        py_step = res_y + beta*py_step

        #Reset if search direction is not in a descent direction
        reset_factor = res_x*px_step + res_y*py_step
        if np.less_equal(reset_factor,np.double(0.0)):
            px_step = res_x
            py_step = res_y
            # print('reset sd')

        #tolerance check both x and y must be sufficiently converged
        if(np.isclose(x_old, x, rtol=mytolerance)):
            xconv = True

        if(np.isclose(y_old, y, rtol=mytolerance)):
            yconv = True

        if(xconv and yconv):
            success = True
            output_history = solution_history
            break

    #After the for loop, if success is false we still need to write a history array
    if(xconv==False or yconv==False):
        success = False
        output_history = solution_history


    return success, x, y, output_history, k+1



#Uses O(H^6) accurate central difference to approximate the derivative wrt x
def differentiate_x(x,y,myfunc,H):
    
    #initialise coeff array for O(H^6)
    coeff = np.array([45, -9, 1], dtype=np.float64)

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
def differentiate_y(x,y,myfunc,H):
    
    #initialise coeff array for O(H^6)
    coeff = np.array([45, -9, 1], dtype=np.float64)

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



#Uses (O(H^2*H^2)) accurate mixed derivative formula to approximate the mixed derivative,
#def differentiate_xy(x,y,myfunc):

#    H = np.double(0.01) #use same step size to treat both variables

#    #Need func for all combinatoric factors of +- h
#    func_plus_plus   = OptFunc(x+H, y+H)
#    func_plus_minus  = OptFunc(x+H, y-H)
#    func_minus_plus  = OptFunc(x-H, y+H)
#    func_minus_minus = OptFunc(x-H, y-H)

#    #Evaluate functions at these points
#    f_plus_plus   = func_plus_plus.solve_for(myfunc)
#    f_plus_minus  = func_plus_minus.solve_for(myfunc)
#    f_minus_plus  = func_minus_plus.solve_for(myfunc)
#    f_minus_minus = func_minus_minus.solve_for(myfunc)

#    partial_xy = np.double(0.0)

#    #Evaluate mixed derivative
#    partial_xy = f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus

#    partial_xy /= (4*H*H)

#    return partial_xy

#Uses O(H^4) accurate mixed derivative for compute to second order
def differentiate_xy(x,y,myfunc,H):

    partial_xy = np.double(0.0)

    #Require all combinatorics of centred two point function
    #Notation is as follows
    #f_p1_m1 is f(x+iH,y-iH) the p is for plus, m for minus, the number after refers to
    #the step size factor so p1 is [var+H] and m2 means [var-(2*H)]

    #Define all combinatorics in groups of 4

    #[1,1]
    func_p1_p1 = OptFunc(x+H,y+H)
    func_m1_m1 = OptFunc(x-H,y-H)
    func_p1_m1 = OptFunc(x+H,y-H)
    func_m1_p1 = OptFunc(x-H,y+H)
    #Evaluate Derivative at these points
    f_p1_p1 = func_p1_p1.solve_for(myfunc)
    f_m1_m1 = func_m1_m1.solve_for(myfunc)
    f_p1_m1 = func_p1_m1.solve_for(myfunc)
    f_m1_p1 = func_m1_p1.solve_for(myfunc)

    partial_xy = 64.0*(f_m1_m1 + f_p1_p1 - f_p1_m1 - f_m1_p1)

    #[1,2]
    func_p1_p2 = OptFunc(x+H,y+(2*H))
    func_m1_m2 = OptFunc(x-H,y-(2*H))
    func_p1_m2 = OptFunc(x+H,y-(2*H))
    func_m1_p2 = OptFunc(x-H,y+(2*H))
    #Evaluate Derivative at these points
    f_p1_p2 = func_p1_p2.solve_for(myfunc)
    f_m1_m2 = func_m1_m2.solve_for(myfunc)
    f_p1_m2 = func_p1_m2.solve_for(myfunc)
    f_m1_p2 = func_m1_p2.solve_for(myfunc)

    partial_xy += 8.0*(f_p1_m2 + f_m1_p2 -f_m1_m2 - f_p1_p2)

    #[2,1]
    func_p2_p1 = OptFunc(x+(2*H),y+H)
    func_m2_m1 = OptFunc(x-(2*H),y-H)
    func_p2_m1 = OptFunc(x+(2*H),y-H)
    func_m2_p1 = OptFunc(x-(2*H),y+H)
    #Evaluate Derivative at these points
    f_p2_p1 = func_p2_p1.solve_for(myfunc)
    f_m2_m1 = func_m2_m1.solve_for(myfunc)
    f_p2_m1 = func_p2_m1.solve_for(myfunc)
    f_m2_p1 = func_m2_p1.solve_for(myfunc)

    partial_xy += 8.0*(f_p2_m1 + f_m2_p1 - f_m2_m1 - f_p2_p1)

    #[2,2]
    func_p2_p2 = OptFunc(x+(2*H),y+(2*H))
    func_m2_m2 = OptFunc(x-(2*H),y-(2*H))
    func_p2_m2 = OptFunc(x+(2*H),y-(2*H))
    func_m2_p2 = OptFunc(x-(2*H),y+(2*H))
    #Evaluate Derivative at these points
    f_p2_p2 = func_p2_p2.solve_for(myfunc)
    f_m2_m2 = func_m2_m2.solve_for(myfunc)
    f_p2_m2 = func_p2_m2.solve_for(myfunc)
    f_m2_p2 = func_m2_p2.solve_for(myfunc)

    partial_xy += 1.0*(f_p2_p2 + f_m2_m2 - f_p2_m2 - f_m2_p2)

    partial_xy /= (144.0*H*H)

    return partial_xy


#Uses (OH^6) accurate central difference to approximate the second derivative wrt x
#def differentiate_xx(x,y,myfunc):
#    H = np.double(0.01)

#    #intialise coeff array
#    coeff    =  np.zeros(4)
#    coeff[0] = -np.double(49.0)/np.double(18.0)
#    coeff[1] =  np.double(3.0)/np.double(2.0)
#    coeff[2] = -np.double(3.0)/np.double(20.0)
#    coeff[3] =  np.double(1.0)/np.double(90.0)

#    #initialise result to zero
#    partial_xx = np.double(0.0)

#    for i in range(0,4):
#        if i==0:
#            #require func only at x,y
#            func_stat = OptFunc(x,y)
#            f_stat = func_stat.solve_for(myfunc)
#            partial_xx += coeff[i]*f_stat
#            continue
#        #Now we require at both + - h
#        func_plus = OptFunc(x+(i*H), y)
#        func_minus = OptFunc(x-(i*H), y)
#        #Evaluate function at the point
#        f_plus = func_plus.solve_for(myfunc)
#        f_minus = func_minus.solve_for(myfunc)
#        #Add to total
#        partial_xx += coeff[i]*(f_plus + f_minus)

#    #Divide by H squared due to second derivative
#    partial_xx /= (H*H)

#    return partial_xx

#Use O(H^8) accurate central difference to approximate the second derivative wrt x
def differentiate_xx(x,y,myfunc,H):
    
    #intialise coeff array
    coeff    =  np.zeros(5)
    coeff[0] = -np.double(205.0)/np.double(72.0)
    coeff[1] =  np.double(8.0)/np.double(5.0)
    coeff[2] = -np.double(1.0)/np.double(5.0)
    coeff[3] =  np.double(8.0)/np.double(315.0)
    coeff[4] = -np.double(1.0)/np.double(560.0)

    #initialise result to zero
    partial_xx = np.double(0.0)

    for i in range(0,5):
        if i==0:
            #require func only at x,y
            func_stat = OptFunc(x,y)
            f_stat = func_stat.solve_for(myfunc)
            partial_xx += coeff[i]*f_stat
            continue
        #Now we require at both + - h
        func_plus = OptFunc(x+(i*H), y)
        func_minus = OptFunc(x-(i*H), y)
        #Evaluate function at the point
        f_plus = func_plus.solve_for(myfunc)
        f_minus = func_minus.solve_for(myfunc)
        #Add to total
        partial_xx += coeff[i]*(f_plus + f_minus)

    #Divide by H squared due to second derivative
    partial_xx /= (H*H)

    return partial_xx

#Uses (OH^6) accurate central difference to approximate the second derivative wrt y
#def differentiate_yy(x,y,myfunc):
#    H = np.double(0.01)

#    #intialise coeff array
#    coeff    =  np.zeros(4)
#    coeff[0] = -np.double(49.0)/np.double(18.0)
#    coeff[1] =  np.double(3.0)/np.double(2.0)
#    coeff[2] = -np.double(3.0)/np.double(20.0)
#    coeff[3] =  np.double(1.0)/np.double(90.0)

#    #initialise result to zero
#    partial_yy = np.double(0.0)

#    for i in range(0,4):
#        if i==0:
#            #require func only at x,y
#            func_stat = OptFunc(x,y)
#            f_stat = func_stat.solve_for(myfunc)
#            partial_yy += coeff[i]*f_stat
#            continue
#        #Now we require at both + - h
#        func_plus = OptFunc(x, y+(i*H))
#        func_minus = OptFunc(x, y-(i*H))
#        #Evaluate function at the point
#        f_plus = func_plus.solve_for(myfunc)
#        f_minus = func_minus.solve_for(myfunc)
#        #Add to total
#        partial_yy += coeff[i]*(f_plus + f_minus)

#    #Divide by H squared due to second derivative
#    partial_yy /= (H*H)

#    return partial_yy


#Use O(H^8) accurate central difference to approximate the second derivative wrt y
def differentiate_yy(x,y,myfunc,H):

    #intialise coeff array
    coeff    =  np.zeros(5)
    coeff[0] = -np.double(205.0)/np.double(72.0)
    coeff[1] =  np.double(8.0)/np.double(5.0)
    coeff[2] = -np.double(1.0)/np.double(5.0)
    coeff[3] =  np.double(8.0)/np.double(315.0)
    coeff[4] = -np.double(1.0)/np.double(560.0)

    #initialise result to zero
    partial_yy = np.double(0.0)

    for i in range(0,5):
        if i==0:
            #require func only at x,y
            func_stat = OptFunc(x,y)
            f_stat = func_stat.solve_for(myfunc)
            partial_yy += coeff[i]*f_stat
            continue
        #Now we require at both + - h
        func_plus = OptFunc(x, y+(i*H))
        func_minus = OptFunc(x, y-(i*H))
        #Evaluate function at the point
        f_plus = func_plus.solve_for(myfunc)
        f_minus = func_minus.solve_for(myfunc)
        #Add to total
        partial_yy += coeff[i]*(f_plus + f_minus)

    #Divide by H squared due to second derivative
    partial_yy /= (H*H)

    return partial_yy



















 


