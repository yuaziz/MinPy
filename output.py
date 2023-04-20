import numpy as np

def write_output(success, x, y, history_array, num_iters):
    """
    Write an output file which contains all information

    """

    format_array = np.array(history_array)

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
        out.writelines('#-----------------------------------------------------#\n')
        out.writelines(f'Converged Solution Found: {success!s}\n')
        out.writelines(f'Global Minimum x : {x:.9f}\n')
        out.writelines(f'Global Minimum y : {y:.9f}\n')
        out.writelines(f'Completed in {num_iters} iterations\n')
        if not success:
            out.writelines('\n')
            out.writelines('**--------------------------------------------------**\n')
            out.writelines('WARNING solution(s) for x and/or y has not met\n')
            out.writelines('the convergence criteria for supplied tolerance.\n')
            out.writelines('Consider increasing max_iter, increasing tolerance,\n')
            out.writelines('using a different solver / line search / beta update.\n')
            out.writelines('**--------------------------------------------------**\n\n')
        out.writelines('#-----------------------------------------------------#\n')
        out.writelines('Writing solution history array\n')

        np.savetxt(out, format_array, delimiter='     ', fmt="%.9f")
        # for row in history_array:
        #     out.write(' '.join([str(a) for a in row]) + '\n')
        # np.savetxt('output.dat', history_array, delimiter=';', newline='\r\n', fmt='%.3f')

