import numpy as np

def write_output(success, x, y, history_array, num_iters):
    """
    Write an output file which contains all information

    """
    with open('output.dat','w') as out:
        out.writelines(r"""

  __  __ _       _____       
 |  \/  (_)     |  __ \      
 | \  / |_ _ __ | |__) |   _ 
 | |\/| | | '_ \|  ___/ | | |
 | |  | | | | | | |   | |_| |
 |_|  |_|_|_| |_|_|    \__, |
                        __/ |
                       |___/ 

        """)
        out.writelines('\n')
        out.writelines('#-------------------------------------------#\n')
        out.writelines(f'Converged Solution Found: {success!s}\n')
        out.writelines(f'Global Minimum x : {x:.6f}\n')
        out.writelines(f'Global Minimum y : {y:.6f}\n')
        out.writelines('#-------------------------------------------#\n')
