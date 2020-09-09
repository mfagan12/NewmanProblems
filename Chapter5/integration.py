from typing import Tuple

def trapezoidal_rule(func: callable, a: float, 
                     b: float, N: int = 1000) -> float:
    '''
    Compute numerical integral of a function via the trapezoidal rule.

    Args:
        func (callable): one-variable function to integrate
        a (float): lower bound of integration
        b (float): upper bound of integration
        N (int, optional): number of sample points to use. Defaults to 1000.

    Returns:
        float: numerical approximation of integral.
    '''
    h = (b-a)/N
    return h * (func(a)/2 
              + func(b)/2 
              + sum((func(a + k*h) for k in range(1, N)))
               )

def simpsons_rule(func: callable, a: float, 
                  b: float, N: int = 1000) -> float:
    '''
    Compute numerical integral of a function via Simpson's rule.

    Args:
        func (callable): one-variable function to integrate
        a (float): lower bound of integration
        b (float): upper bound of integration
        N (int, optional): number of sample points to use. Defaults to 1000.

    Returns:
        float: numerical approximation of integral.
    '''
    h = (b-a)/N
    return (1/3) * h * (func(a) 
                        + func(b) 
                        + 4 * sum((func(a + k*h) for k in range(1, N, 2)))
                        + 2 * sum((func(a + k*h) for k in range(2, N-1, 2)))
                       )
    
def simple_adaptive_trapezoidal(func: callable, a: float, 
                         b: float, tol: float = 1/1000) -> float:
    '''
    Compute numerical integral of a function via the adaptive trapezoidal rule
    with constant width slices on the domain.

    Args:
        func (callable): one-variable function to integrate
        a (float): lower bound of integration
        b (float): upper bound of integration
        tol (float, optional): error tolerance of the integration. 
            Defaults to 1/1000.

    Returns:
        float: [description]
    '''
    raise NotImplementedError

def adaptive_trapezoidal(func: callable, a: float, 
                         b: float, tol: float = 1/1000) -> float:
    '''
    Compute numerical integral of a function via a general adaptive trapezoidal 
    rule.

    Args:
        func (callable): one-variable function to integrate
        a (float): lower bound of integration
        b (float): upper bound of integration
        tol (float, optional): error tolerance of the integration. 
            Defaults to 1/1000.

    Returns:
        float: [description]
    '''
    raise NotImplementedError

def recursive_adaptive_trapezoidal(func: callable, a: float, 
                         b: float, tol: float = 1/1000) -> float:
    '''
    Compute numerical integral of function between given endpoints via recursive
    adaptive trapezoidal rule with given tolerance, automatically computing
    non-uniform partition of integral to achieve specified tolerance.

    Args:
        func (callable): one-variable function to integrate
        a (float): lower bound of integration
        b (float): upper bound of integration
        tol (float, optional): error tolerance of the integration. 
            Defaults to 1/1000.

    Returns:
        float: numerical approximation of integral.
    '''
    def step(x1: float, x2: float, f1: float, f2: float) -> Tuple[float, float]:
        '''
        Helper function to compute one- and two-slice trapezoidal rule on a
        single starting slice.

        Args:
            x1 (float): left endpoint of slice.
            x2 (float): right endpoint of slice.
            f1 (float): value of integrand at left endpoint.
            f2 (float): value of integrand at right endpoint.

        Returns:
            Tuple[float, float]: Value of one- and two-slice trapezoidal rule
                between given endpoints, respectively.
        '''
        xm = (x1 + x2) / 2
        fm = func(xm)
        I_1 = (x2 - x1) * (f1 + f2) / 2
        I_2 = ((xm - x1) * (fm + f1) / 2 
               + (x2 - xm) * (fm + f2) / 2)
        return I_1, I_2
    
    I_1, I_2 = step(a, b, func(a), func(b))
    error = (1/3) * abs(I_2 - I_1)
    
    if error <= tol:
        return I_2
    else:
        mid = (a+b)/2
        return recursive_adaptive_trapezoidal(func, a, mid, tol/2) \
            + recursive_adaptive_trapezoidal(func, mid, b, tol/2)
   
def romberg_integrate():
    raise NotImplementedError
    
def adaptive_simpson(func: callable, a: float, 
                     b: float, tol: float = 1/1000) -> float:
    raise NotImplementedError

def recursive_adaptive_simpson(func: callable, a: float, 
                         b: float, tol: float = 1/1000) -> float:
    '''
    Compute numerical integral of function between given endpoints via recursive
    adaptive Simpson's rule with given tolerance, automatically computing
    non-uniform partition of integral to achieve specified tolerance.

    Args:
        func (callable): one-variable function to integrate
        a (float): lower bound of integration
        b (float): upper bound of integration
        tol (float, optional): error tolerance of the integration. 
            Defaults to 1/1000.

    Returns:
        float: numerical approximation of integral.
    '''
    def step(x1: float, x2: float, xm: float, 
             f1: float, f2: float, fm: float) -> Tuple[float, float]:
        '''
        Helper function to compute one- and two-slice Simpson's rule on a
        single starting slice.

        Args:
            x1 (float): left endpoint of slice.
            x2 (float): right endpoint of slice.
            f1 (float): value of integrand at left endpoint.
            f2 (float): value of integrand at right endpoint.

        Returns:
            Tuple[float, float]: Value of one- and two-slice Simpson's rule
                between given endpoints, respectively.
        '''
        h = (x2 - x1) / 2
        I_1 = (h/3) * (f1  + 4*fm + f2)
        I_2 = (h/6) * ((f1  + 4*func((x1 + xm)/2) + fm) + 
                       (fm  + 4*func((xm + x2)/2) + f2))
        return I_1, I_2
    
    mid = (a + b) / 2
    I_1, I_2 = step(a, b, mid, func(a), func(b), func(mid))
    error = (1/15) * abs(I_2 - I_1)
    
    if error <= tol:
        return I_2
    else:
        return recursive_adaptive_simpson(func, a, mid, tol/2) \
            + recursive_adaptive_simpson(func, mid, b, tol/2)

def gaussian_quadrature(func, a, b, N):
    def gaussian_weights(N):
        return points, weights
        raise NotImplementedError
    
    points, weights = gaussian_weights(N)
    raise NotImplementedError