# region imports
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as pyplot
from math import floor, ceil
# endregion

# region functions
def RSquared(x,y,coeff):
    '''
    To calculate the R**2 value for a set of x,y data and a LeastSquares fit with polynomial having coefficients a
    :param x:
    :param y:
    :param a:
    :return:
    '''
    AvgY=np.mean(y) #calculates the average value of y
    SSTot=0
    SSRes=0
    ymodel=Poly(x,*coeff)
    for i in range(len(y)):
        SSTot+=(y[i]-AvgY)**2
        SSRes+=(y[i]-ymodel[i])**2
    RSq=#JES MISSING CODE
    return RSq

def Poly(xdata, *a):
    '''
    calculates the value for a polynomial given a value for xdata and the coefficients of the polynomial.
    f(x)=y=a[0]+a[1]x+a[2]x**2+a[3]x**3+...
    :param x: an array of x values for computing corresponding y values
    :param args: the tuple or list of coefficients (perhaps a numpy array)
    :return: the array of y values corresponding to xdata
    '''
    y = np.zeros_like(xdata)
    power = len(a)-1
    for i in range(power+1):
        for j in range(len(xdata)):
            x=xdata[j]
            c=a[i]
            y[j] += c*x**i
    return y

def PlotLeastSquares(x, y, coeff, showpoints=True, npoints=500):
    '''
    This makes a single formatted plot for a polynomial fit to the x,y data.
    :param x: the x data as a numpy array
    :param y: the y data as a numpy array
    :param coeff: the coefficients for the polynomial fit as a tuple or list
    :param showpoints: boolean indicating if we should show points or not
    :param npoints: number of points in the curve fit to plot
    :return: list of xvals and yvals for the plot.
    '''
    #find limits of the data
    Xmin=min(x)
    Xmax=max(x)
    Ymin=min(y)
    Ymax=max(y)

    if len(coeff)==0:
        coeff=LeastSquaresFit(x,y,1)
    #build numpy arrays for x and y for the curve fit
    xvals=np.linspace(Xmin,Xmax, npoints)
    yvals=np.array(Poly(xvals,*coeff))  # note: *coeff unpacks the tuple/list and passes to function expecting *args

    #calculate an r squared value
    RSq=RSquared(x,y,coeff)

    #make the plot
    pyplot.plot(xvals,yvals,linestyle='dashed',color='black',linewidth='2')
    pyplot.title(r'$R^2={:0.3f}$'.format(RSq)) #place r squared in title
    pyplot.xlim(floor(Xmin*10)/10,ceil(Xmax*10)/10)
    pyplot.ylim(floor(Ymin),ceil(Ymax*10)/10)
    if showpoints: pyplot.plot(x,y,linestyle='none', marker='o', markerfacecolor='white', markeredgecolor='black', markersize=10)
    pyplot.xlabel('X values')
    pyplot.ylabel('Y values')
    pyplot.gca().tick_params(axis='both',top=True, right=True, direction='in', grid_linewidth=1, grid_linestyle='dashed',grid_alpha=0.5)
    pyplot.show()
    return xvals, yvals

def LeastSquaresFit(x, y, power=1):
    """
    This is a convenience method that fits some x, y data with a polynomial of
    degree=power by calling scipy.optimize.curve_fit and returning the coefficients
    for the polynomial.
    :param x: the x-values of data points
    :param y: the y-values of data points
    :param power: the degree of the polynomial
    :return: the coefficients for the polynomial fit as an array
    """
    a = np.array([1 for x in range(power+1)])  # create a list for initial guess at coefficients
    coeff, cov = curve_fit(#JES MISSING CODE)
    return coeff

def main():
    """
    For use in testing and plotting curve_fit
    :return: nothing
    """
    x = np.array([0.05, 0.11, 0.15, 0.31, 0.46, 0.52, 0.70, 0.74, 0.82, 0.98, 1.17])
    y = np.array([0.956, 1.09, 1.332, 0.717, 0.771, 0.539, 0.378, 0.370, 0.306, 0.242, 0.104])
    #1. Call LeastSquaresFit to get coefficients for linear fit
    coeff1=LeastSquaresFit(#JES MISSING CODE)
    #
    #2. Call PlotLeastSquares for a linear fit
    linx,liny=PlotLeastSquares(x,y, coeff1, showpoints=True, npoints=500)
    RSqLin=RSquared(x,y,coeff1) #calculate RSqured for linear fit
    #
    # #3. Call LeastSquares for a Cubic fit
    coeff2=LeastSquaresFit(#JES MISSING CODE)

    # #4. Call PlotLeastSquares for a Cubic fit
    cubx,cuby=PlotLeastSquares(x,y,coeff2, showpoints=True, npoints=500)
    RSqCub=RSquared(x,y,coeff2) #calculate RSqured for cubic fit

    #
    # #5. Use results form 1 and 3 to plot the data points, the Linear fit and the Cubic fit all on one graph
    pyplot.plot(linx,liny, linewidth=2, linestyle='dashed', color='black', label=r'Linear fit ($R^2={:0.3f}$)'.format(RSqLin))
    pyplot.plot(cubx,cuby, linewidth=2, linestyle='dotted', color='black', label='Cubic fit ($R^2={:0.3f}$)'.format(RSqCub))
    pyplot.plot(x, y, linestyle='none', marker='o', markersize=10, markerfacecolor='white', markeredgecolor='black', label='Data')
    pyplot.xlabel('X values')
    pyplot.ylabel('Y values')
    pyplot.legend()

    pyplot.tick_params(axis='both', top=True, right=True, direction='in', grid_linewidth=1, grid_linestyle='dashed',grid_alpha=0.5)
    pyplot.show()

# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion