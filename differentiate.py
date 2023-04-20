#This module houses all the functions which compute the various numerical derivatives we will require
# for both NLCG and BFGS solvers.

#We will require access to the optimisation functions module, since we need to evaluate the user 
# specified function for a number of points.

import numpy as np
from optimisation_functions import OptFunc


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


