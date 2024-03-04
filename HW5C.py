# region imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# endregion

# region functions
def ode_system(t, X, *params):
    '''
    The ode system is defined in terms of state variables.
    I have as unknowns:
    x: position of the piston (This is not strictly needed unless I want to know x(t))
    xdot: velocity of the piston
    p1: pressure on the right of the piston
    p2: pressure on the left of the piston
    For initial conditions, we see: x=x0=0, xdot=0, p1=p1_0=p_a, p2=p2_0=p_a
    :param X: The list of state variables.
    :param t: The time for this instance of the function.
    :param params: the list of physical constants for the system.
    :return: The list of derivatives of the state variables.
    '''
    # Unpack the parameters
    A, Cd, ps, pa, V, beta, rho, Kvalve, m, y = params

    # Unpack the state variables
    x, xdot, p1, p2 = X

    # Use equations from the assignment
    xddot = xdot
    p1dot = x
    p2dot = p1

    # Return the list of derivatives of the state variables
    return [xddot, xdot, p1dot, p2dot]

def main():
    # After some trial and error, I found all the action seems to happen in the first 0.02 seconds
    t = np.linspace(0, 0.02, 200)

    myargs = (4.909E-4, 0.6, 1.4E7, 1.0E5, 1.473E-4, 2.0E9, 850.0, 2.0E-5, 30, 0.002)

    # Because the solution calls for x, xdot, p1, and p2, I make these the state variables X[0], X[1], X[2], X[3]
    # ic=[x=0, xdot=0, p1=pa, p2=pa]
    pa = myargs[3]
    ic = [0, 0, pa, pa]

    # Call solve_ivp with ode_system as a callback
    sln = solve_ivp(ode_system, (t[0], t[-1]), ic, args=myargs, t_eval=t)

    # Unpack result into meaningful names
    xvals = sln.y[0]
    xdot_vals = sln.y[1]
    p1_vals = sln.y[2]
    p2_vals = sln.y[3]

    # Plot the result
    plt.subplot(2, 1, 1)
    plt.plot(t, xvals, 'r-', label='$x$')
    plt.ylabel('$x$')
    plt.legend(loc='upper left')

    ax2 = plt.twinx()
    ax2.plot(t, xdot_vals, 'b-', label='$\dot{x}$')
    plt.ylabel('$\dot{x}$')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(t, p1_vals, 'b-', label='$P_1$')
    plt.plot(t, p2_vals, 'r-', label='$P_2$')
    plt.legend(loc='lower right')
    plt.xlabel('Time, s')
    plt.ylabel('$P_1, P_2 (Pa)$')

    plt.show()
# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion
