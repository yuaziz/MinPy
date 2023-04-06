import numpy as np
from optimisation_functions import OptFunc

def bfgs_solver(parameters):
    """Perform BFGS (Quasi-Newton Update), polak-ribiere utilised exclusively (for now) for the beta update

    The secant method is used to impose the Quasi-Newton update of the approximate hessian

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
    xconv = False
    yconv = False
    success = False
    SIGMA_INIT = 0.01 #set as a constant SECANT method step parameter

    #Step size for numerical derivative evaluation
    STEP_SIZE=np.double(0.01) #set as constant for now, could be user defined

    #Intialise array which would hold the computed solutions, saved at every iteration
    solution_history = [[x,y]]
    # solution_history = np.empty((max_iter,3), dtype=np.float64)
    #history is indexed by row iter x y

    #Unlike NLCG, here we will store x and y in arrays and do the majority of operations with arrays.
    #x and y will be stored in a column vector of form
    #vec = ({[x],
    #        [y]})

    for k in range(0,max_iter):

        if(k==0):

            #Intialise solution vector
            sol_vec = np.zeros((2,1))
            sol_vec[0,0] = x
            sol_vec[1,0] = y

            #Initialise the approximate hessian to the 2X2 identity
            hess = np.eye(2,2)

            #We require the inverse here, already know the solution for the identity but call function anyway
            hess_inv = hessian_matrix_inverse(hess)

            #Intialise and poopulate the gradient vector
            grad_vec = np.zeros((2,1))
            grad_vec[0,0] = differentiate_x(x,y,function,STEP_SIZE)
            grad_vec[1,0] = differentiate_y(x,y,function,STEP_SIZE)

            #Intialise and compute search direction vector
            p_vec = np.zeros((2,1))
            p_vec = -np.matmul(hess_inv,grad_vec)

            #set initial alpha
            alpha = -SIGMA_INIT

            #Intialise and evaluate fprime_sig vector
            fprime_sig = np.zeros((2,1))
            fprime_sig[0,0] = differentiate_x(x+(SIGMA_INIT*p_vec[0,0]),y,function,STEP_SIZE)
            fprime_sig[1,0] = differentiate_y(x,y+(SIGMA_INIT*p_vec[1,0]),function,STEP_SIZE)

            #evaluate eta_prev and eta

            eta_prev = np.double(np.matmul(np.transpose(fprime_sig),p_vec))

            eta = np.double(np.matmul(np.transpose(grad_vec),p_vec))

            alpha_upper = eta

            alpha_lower = eta_prev - eta

            #Update alpha
            alpha *= (alpha_upper / alpha_lower)

            #Update sigma
            sigma = -alpha

            #Update solution vector, because we need to check the difference in solutions later
            #this is saved to a new vector
            sol_vec_new = np.zeros((2,1))
            sol_vec_new = sol_vec + np.multiply(alpha,p_vec)

            #Update the approximate hessian using this new solution
            x = sol_vec_new[0,0]
            y = sol_vec_new[1,0]

            #Require gradient vector at new solution point, intialise and populate
            grad_vec_new = np.zeros((2,1))
            grad_vec_new[0,0] = differentiate_x(x,y,function,STEP_SIZE)
            grad_vec_new[1,0] = differentiate_y(x,y,function,STEP_SIZE)

            #Require gradient difference, subtract new and old
            grad_vec_diff = np.subtract(grad_vec_new, grad_vec)

            #Also require solution vector difference
            sol_vec_diff = np.subtract(sol_vec_new, sol_vec)

            #Update the approximate hessian using new solution and diff vectors
            hess_update = hessian_matrix_update(grad_vec_diff,sol_vec_diff,hess)

            #Create a new hessian matrix from the original and update term
            hess_new = hess + hess_update

            #Set variables for the next iteration
            #sigma already set earlier
            hess = np.copy(hess_new)
            sol_vec = np.copy(sol_vec_new)
            grad_vec = np.copy(grad_vec_new)

            #Save to solution history
            # solution_history[k,0] = k
            # solution_history[k,1] = x
            # solution_history[k,2] = y
            solution_history.append([x,y])

        else:

            x = sol_vec[0,0]
            y = sol_vec[1,0]

            #Compute inverse hessian
            hess_inv = hessian_matrix_inverse(hess)

            #Compute search direction vector
            p_vec = -np.matmul(hess_inv,grad_vec)
            
            #Gradient vector
            grad_vec[0,0] = differentiate_x(x,y,function,STEP_SIZE)
            grad_vec[1,0] = differentiate_y(x,y,function,STEP_SIZE)

            #line search with fprime_sig vector
            fprime_sig[0,0] = differentiate_x(x+(sigma*p_vec[0,0]),y,function,STEP_SIZE)
            fprime_sig[1,0] = differentiate_y(x,y+(sigma*p_vec[1,0]),function,STEP_SIZE)

            #evaluate eta_prev and eta
            eta_prev = np.double(np.matmul(np.transpose(fprime_sig),p_vec))
            eta = np.double(np.matmul(np.transpose(grad_vec),p_vec))
            alpha_upper = eta
            alpha_lower = eta_prev - eta

            #Update alpha
            alpha *= (alpha_upper / alpha_lower)

            #Save x and y for conv check
            x_old = x
            y_old = y
            #Update solution
            sol_vec_new = sol_vec + np.multiply(alpha,p_vec)

            #Update the approximate hessian using this new solution
            x = sol_vec_new[0,0]
            y = sol_vec_new[1,0]
            
            #Require gradient vector at new solution point
            grad_vec_new[0,0] = differentiate_x(x,y,function,STEP_SIZE)
            grad_vec_new[1,0] = differentiate_y(x,y,function,STEP_SIZE)

            #Require gradient difference, subtract new and old
            grad_vec_diff = np.subtract(grad_vec_new, grad_vec)

            #Also require solution vector difference
            sol_vec_diff = np.subtract(sol_vec_new, sol_vec)

            #Update the approximate hessian using new solution and diff vectors
            hess_update = hessian_matrix_update(grad_vec_diff,sol_vec_diff,hess)

            #Create a new hessian matrix from the original and update term
            hess_new = hess + hess_update

            #Set variables for the next iteration
            sigma = -alpha
            hess = np.copy(hess_new)
            sol_vec = np.copy(sol_vec_new)
            grad_vec = np.copy(grad_vec_new)

            #Save to solution history
            # solution_history[k,0] = k
            # solution_history[k,1] = x
            # solution_history[k,2] = y
            solution_history.append([x,y])

            #tolerance check both x and y must be sufficiently converged
            if(np.isclose(x_old, x, rtol=mytolerance)):
                xconv = True

            if(np.isclose(y_old, y, rtol=mytolerance)):
                yconv = True

            if (xconv and yconv):
                success = True
                output_history = solution_history
                break

    #After the for loop, if success is false we still need to write a history array
    if(xconv==False or yconv==False):
        success = False
        output_history = solution_history

    return success, x, y, output_history, k


#Compute the inverse of the supplied hessian, return the inverse
def hessian_matrix_inverse(hessian):

    #Small number to compare the determinant
    eps = np.double(1.0e-16)

    #Evaluate determinant
    det = hessian[0,0]*hessian[1,1] - hessian[0,1]*hessian[1,0]

    #Error and exit if determinant is too small
    if ( np.less(np.absolute(det),eps) ):
        sys.exit('Error in bfgs : hessian_inverse, determinant is too small to evaluate')

    #Intialise and set the cofactor matrix
    cofactor = np.zeros((2,2))
    cofactor[0,0] =  hessian[1,1]
    cofactor[0,1] = -hessian[1,0]
    cofactor[1,0] = -hessian[0,1]
    cofactor[1,1] =  hessian[0,0]

    #Require transpose
    cofactor_t = np.transpose(cofactor)

    #Evaluate inverse
    hessian_inverse = np.divide(cofactor_t, det)

    return hessian_inverse

#Update the hessian matrix using the current hessian, diff solution and diff gradient vectors
#return an np.double hessian_update
def hessian_matrix_update(grad_diff, sol_diff, hess):

    #initialise hess_update with zeros
    hess_update = np.zeros((2,2))


    #term1 is a 2X2 grad_diff.T grad_diff / sol_diff . grad_diff
    term1_upper = np.matmul(grad_diff,np.transpose(grad_diff)) #2X2

    term1_lower = np.double(np.matmul(np.transpose(sol_diff),grad_diff))

    term1 = np.divide(term1_upper,term1_lower)


    #term2 is a 2X2 {hess . sol_diff . sol_diff_T . hess_T} / sol_diff_T.hess.sol_diff
    #split the numerator into two parts term2_upper1 and term2_upper2
    term2_upper1 = np.matmul(hess,sol_diff) #2X1
    term2_upper2 = np.matmul(np.transpose(sol_diff),hess) #1X2

    #Evaluate upper part of term2
    term2_upper = np.matmul(term2_upper1,term2_upper2) #2X2

    #Denominator of term2 invovles three terms, evaluate last two first
    term2_lower1 = np.matmul(hess,sol_diff) #2X1
    term2_lower  = np.double(np.matmul(np.transpose(sol_diff),term2_lower1))

    #Now evaluate term 2
    term2 = np.divide(term2_upper,term2_lower)


    #Finally evaluate hess_update
    hess_update = np.subtract(term1, term2)

    return hess_update


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











