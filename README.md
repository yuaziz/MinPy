# MinPy

## Table of contents

- [What's included](#whats-included)
- [Usage](#usage)
- [Parameter File](#parameter-file)
- [Example Calculation](#example-calculation)
- [Status and Further Work](#status-and-further-work)
- [Dependencies](#dependencies)
- [References](#references)
- [Copyright and license](#copyright-and-license)

The use of this script is to apply iterative methods to solve nonlinear optimisation problems. As of present, Non Linear Conjugate Gradient (NLCG) and the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm have been developed. Multivariate functions in the $(x,y)$ domain are considered.

Please note there is **no** relation between the name of this repository and that of the NumPy interface [MinPy](https://minpy.readthedocs.io/en/latest/ "https://minpy.readthedocs.io/en/latest/#").

## Whats Included

```text
- param.py
- nlcg.py
- bfgs.py
- output.py
- main.py
- optimisation_functions.py
```

## Usage

After setting up an appropriate parameter file, 'compute.param', run the 'main.py' script and supply a single parameter file as an argument::

```bash
python main.py compute.param
```

## Parameter File

The script expects a parameter file which provides the means for you to set the variables for the computation. These include the type of solver you wish to use and the optimisation function you wish to apply the iterative method in an attempt to find the global minimum.

Here all parameters accessible to you are listed, some are required for the computation to run. Others, if not specified, have a default setting.

* * *

### Solver

*REQUIRED*

Allowed values:

- NLCG
- BFGS

* * *

### Function

*REQUIRED*

More details of these functions may be found on the [Wiki](https://en.wikipedia.org/wiki/Test_functions_for_optimization)

Allowed values:

- Rosenbrock
- Rastrigin
- Ackley
- Beale
- Booth
- Matyas
- Himmelblau
- Easom

* * *

### x\_initial

*REQUIRED*

Starting value for the $x$ coordinate

### y\_initial

*REQUIRED*

Starting value for the $y$ coordinate

Both 'x\_initial' and 'y\_initial' must be within the allowed search domain for the function of interest.

- Rosenbrock $\longrightarrow$ $[x, y]\in[-100, 100]$
- Rastrigin $\longrightarrow$ $[x, y]\in[-5.12, 5.12]$
- Ackley $\longrightarrow$ $[x, y]\in[-5, 5]$
- Beale $\longrightarrow$ $[x,y]\in[-4.5, 4.5]$
- Booth $\longrightarrow$ $[x, y]\in[-10, 10]$
- Matyas $\longrightarrow$ $[x, y]\in[-10, 10]$
- Himmelblau $\longrightarrow$ $[x, y]\in[-5, 5]$
- Easom $\longrightarrow$ $[x, y]\in[-100, 100]$

* * *

### beta\_update

*OPTIONAL*

Specifies the formula of the $\beta$ update used in determining the search direction. Note the use of underscore instead of hyphenating.

Allowed Values:

- Fletcher\_Reeves
- Polak\_Ribiere

Default : 'fletcher\_reeves'

* * *

### line\_search

*OPTIONAL*

Defines the algorithm used to optimise find an $\alpha$ such that $f(X_{k} +\alpha p_{k} )< f_{k}$, where $p$ is the search direction updated by $\beta$.

Allowed Values:

- Secant
- Newton\_Raphson

Default : 'newton\_raphson'

**Note that for now, the 'Secant' method exclusively uses the 'polak\_ribiere' to update $\beta$. Also 'Newton\_Raphson' only uses 'fletcher\_reeves'.**

* * *

### max\_iter

*OPTIONAL*

Maximum number of iterations the solver will attempt before settling on the global minimum.

Default : 10000

* * *

### tolerance

*OPTIONAL*

Convergence tolerance which is satisfied when subsequent updates of both $x$ and $y$ lie within tolerance range.

Default : 1.0e-9

* * *

## Example Calculation

In this example, the global minimum of the Rosenbrock (banana) is determined from initial starting points of $x=4.0$ and $y=4.0$. The Nonlinear Conjugate Gradient solver is used, employing a Newton-Raphson line search along with a Fletcher-Reeves formula for the $\beta$ update. The parameter file 'compute.param' for this is written:

```text
Function    = Rosenbrock
x_initial   = 4.0
y_initial   = 4.0
solver      = nlcg
beta_update = fletcher_reeves
line_search = newton_raphson
max_iter    = 10000
tolerance   = 1.0e-9
```

Run this as follows:

```bash
python main.py compute.param
```

You can have 'compute.param' in any directory so long as you correctly define the path to 'main.py' and it is able to access all the other modules supplied in this repository.

Upon completion, a file 'output.dat' will contain the following:

```text
  __  __ _       _____       
 |  \/  (_)     |  __ \      
 | \  / |_ _ __ | |__) |   _ 
 | |\/| | | '_ \|  ___/ | | |
 | |  | | | | | | |   | |_| |
 |_|  |_|_|_| |_|_|    \__, |
                        __/ |
                       |___/ 

        
#-----------------------------------------------------#
Converged Solution Found: True
Global Minimum x : 0.999999808
Global Minimum y : 0.999999615
Completed in 28 iterations
#-----------------------------------------------------#
Writing solution history array
4.000000000     4.000000000
2.916639358     4.135377775
2.329824412     4.230130061
2.105688934     4.275516746
2.070047185     4.283691818
2.069117613     4.283774351
1.846349694     3.406880941
1.839780431     3.391176687
1.679408351     2.760094148
1.662523670     2.764958012
1.662129812     2.764848559
1.417946375     1.987822203
1.378735627     1.873579535
1.303486458     1.669472123
1.234853456     1.495280799
1.168724328     1.338289306
1.104758379     1.196951604
1.045736736     1.077079474
1.004406100     1.002547062
0.995252730     0.990197843
0.995109795     0.990220699
0.995114367     0.990260000
0.999797321     0.999675595
0.999970166     0.999939611
0.999969941     0.999939773
0.999973149     0.999945314
0.999999774     0.999999604
0.999999808     0.999999614
0.999999808     0.999999615
```

## Status and Further Work

\- Will aim to include more formulations for line searches and methods for the $\beta$ update, more flexibility for the user to pick and choose.
\- Implement plotting capability so the user can intuitively visualise algorithm behaviour as it attempts to reach global minima.

## Dependencies

Tested with Python 3.8.5

[NumPy](https://numpy.org/)

## References

Jonathan Richard Shewchuk's: [An Introduction to the Conjugate Gradient Method Without the Agonizing Pain](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)

[Test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization)

[Quasi-Newton Methods](http://www.seas.ucla.edu/~vandenbe/236C/lectures/qnewton.pdf)

[BFGS](https://www.cs.purdue.edu/homes/jhonorio/16spring-cs52000-quasinewton.pdf)

## Thanks

Thank you for checking out my work!

## Copyright and license

Copyright (C) 2023 Yusuf Aziz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see [GNU](https://www.gnu.org/licenses).
