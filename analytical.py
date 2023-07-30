# This code contains the analytical methods to approximate the fixed points K*
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import plot
import function
from scipy.optimize import curve_fit
from scipy import integrate
import os


# Maybe rewrite rk4 using analytical_recursion() function in function.py....


# Runge-Kutta 4 method (from Symbolic Computation Coursework3):
def rk4(fn, initialVals, t0, tn, n):
    # using np.array for values allows us to use + and * for the elements
    vals = np.array(initialVals)
    dt = (tn - t0) / n
    tList = []
    # each aList element will be a list of vals for that component, ie [[x0,x1,x2,x3], [y0,y1,y2,y3]]
    aList = [[] for i in vals]
    for i in range(n + 1):
        t = t0 + i * dt  # current t val
        tList.append(t)
        for element, val in enumerate(vals):
            aList[element].append(val)
        k1 = dt * fn(t, vals)
        k2 = dt * fn(t + dt/2, vals + k1 / 2)
        k3 = dt * fn(t + dt/2, vals + k2 / 2)
        k4 = dt * fn(t + dt, vals + k3)
        vals = vals + (k1 + 2*k2 + 2*k3 + k4)/6
    return (tList, aList)

# Maybe also plot the method to show the fixed point convergence, similar to analytical recursion function I think.


