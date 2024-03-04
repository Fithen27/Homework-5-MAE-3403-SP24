# region imports
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# endregion

# region functions
def ff(Re, rr, CBEQN=False):
    """
    This function calculates the friction factor for a pipe based on the
    notion of laminar, turbulent and transitional flow.
    :param Re: the Reynolds number under question.
    :param rr: the relative pipe roughness (expect between 0 and 0.05)
    :param CBEQN: boolean to indicate if I should use Colebrook (True) or laminar equation
    :return: the (Darcy) friction factor
    """
    if CBEQN:
        cb = lambda f: -2.0 * np.log((rr / 3.7) + (2.51 / (Re * np.sqrt(np.abs(f))))) - 1.0 / np.sqrt(np.abs(f))
        result = fsolve(cb, 0.02)
        return result[0]
    else:
        return 64 / Re

def plotMoody(plotPoint=False, pt=(0, 0)):
    """
    This function produces the Moody diagram for a Re range from 1 to 10^8 and
    for relative roughness from 0 to 0.05 (20 steps).  The laminar region is described
    by the simple relationship of f=64/Re whereas the turbulent region is described by
    the Colebrook equation.
    :return: just shows the plot, nothing returned
    """
    ReValsCB = np.logspace(3.6, 8, 100)
    ReValsL = np.logspace(np.log10(600.0), np.log10(2000.0), 20)
    ReValsTrans = np.logspace(3.3, 3.6, 50)
    rrVals = np.array(
        [0, 1E-6, 5E-6, 1E-5, 5E-5, 1E-4, 2E-4, 4E-4, 6E-4, 8E-4, 1E-3, 2E-3, 4E-3, 6E-3, 8E-8, 1.5E-2, 2E-2, 3E-2,
         4E-2, 5E-2])

    ffLam = np.array([64 / Re for Re in ReValsL])
    ffTrans = np.array([64 / Re for Re in ReValsTrans])

    ffCB = np.array([[ff(Re, relRough, True) for Re in ReValsCB] for relRough in rrVals])

    plt.loglog(ReValsL, ffLam, 'k-', label='Laminar')
    plt.loglog(ReValsTrans, ffTrans, 'k--', label='Transition')
    for nRelR in range(len(ffCB)):
        plt.loglog(ReValsCB, ffCB[nRelR], label=f'rr={rrVals[nRelR]}')
        plt.annotate(text=f'{rrVals[nRelR]:.2e}', xy=(ReValsCB[-1], ffCB[nRelR][-1]), xytext=(-20, -10),
                     textcoords='offset points', ha='right')

    plt.xlim(600, 1E8)
    plt.ylim(0.008, 0.10)
    plt.xlabel(r"Reynolds number $Re=Vd/V$", fontsize=16)
    plt.ylabel(r"Friction factor $f=h/(L/d * V^2/2g)$", fontsize=16)
    plt.text(2.5E8, 0.02, r"Relative roughness $\frac{\epsilon}{d}$", rotation=90, fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)
    ax.tick_params(axis='both', grid_linewidth=1, grid_linestyle='solid', grid_alpha=0.5)
    ax.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    plt.grid(which='both')
    if plotPoint:
        plt.plot(pt[0], pt[1], 'ro', markersize=12, markeredgecolor='red', markerfacecolor='none')

    plt.legend()
    plt.show()

def main():
    plotMoody()

# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion
