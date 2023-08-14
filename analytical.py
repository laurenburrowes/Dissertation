# This code contains the analytical methods to approximate the fixed points K*
import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
import tensorflow as tf
import plot as plot
import function as func
from scipy.optimize import curve_fit
from scipy import integrate
import os

# Using the analytical method in the book, pg 126.
# Want to find suitable Cw and Cb for a fixed point.
def fixed_pt_check(activation_function, start, step, step_count):
    perppar_array = []


    # 2nd derivative of rho^2:
    deriv_2 = lambda z: func.avg_rho_sq(activation_function(z)**2 *(z*z - k), k)

    # expectation of rho^2 where rho is the activation function (tanh(z)).
    #exp2 = func.avg_rho_sq(activation_function, k)

    # to find the differential of the activation function.
    rho_diff = lambda z: smp.sech(z)*smp.sech(z)
    
    
    k = start
    for i in range(0, step_count):
        # The expectation value of rho dashed squared, used for parallel susceptibility.
        exp_dash = func.avg_rho_sq(rho_diff, k)
        # The expectation value of rho^2(z^2-k):
        func2 = lambda  z: activation_function(z)**2 *(z*z - k)
        deriv_2 = func.avg_rho_sq(func2, k)
        # chi_perp over chi_parallel:
        perppar = (2*k * exp_dash[0]) / (deriv_2[0])
        perppar_array.append(perppar)
        print(k, perppar)
        k += step
    
    # Initially want to find values of Cw that satisfy chi_perp = chi_par = 1, then check that Cb is also greater than or equal to 0.

    return print(perppar_array)

# Using the second analytical method in the book, pg 126.
# Want to find suitable Cw and Cb for a fixed point K*.   
def fixed_pt_goal_seek(activation_function, start, threshold):
    # Target value of chi_perp/chi_parallel:
    target = 1

    # 2nd derivative of rho^2:
    deriv_2 = lambda z: func.avg_rho_sq(activation_function(z)**2 *(z*z - k), k)

    # expectation of rho^2 where rho is the activation function (tanh(z)).
    #exp2 = func.avg_rho_sq(activation_function, k)

    # to find the differential of the activation function.
    rho_diff = lambda z: smp.sech(z)*smp.sech(z)
    
    k_array = []
    k = start
    diff = threshold + 1
    x_array=[]
    i=0
    while abs(diff) > threshold:
        # The expectation value of rho dashed squared, used for perpendicular susceptibility.
        exp_dash = func.avg_rho_sq(rho_diff, k)
        # The expectation value of rho^2(z^2-k), used for parallel susceptibility:
        func2 = lambda  z: activation_function(z)**2 * (z*z - k)
        deriv_2 = func.avgrho(func2, k)
        # chi_perp / chi_parallel:
        perppar = (2*(k**2) * exp_dash[0]) / (deriv_2[0]) 
        diff = target - perppar # difference between target value (1) and the current value of chi_perp/chi_parallel
        print(k, perppar, diff)
        k_array.append(k)
        # changing the value of k based on the difference to target:
        k += diff/10
        x_array.append(i)
        i+=1 # number of iterations

    Cw = 1/exp_dash[0]
    Cb = k - (Cw * func.avg_rho_sq(activation_function, k)[0])
    print("Cw = ", Cw, "Cb = ", Cb, "K* = ", k)

    plot_type = plot.determine_plot_type(activation_function)

    plt.plot(x_array, k_array)
    plt.xlabel("Iteration")
    plt.ylabel("Kernel K")
    plt.title(f"Fixed point K* = {round(k, 2)},\n Cw = {round(Cw, 2)}, Cb = {round(Cb, 2)}")
    os.makedirs(f"Fixed Points/{plot_type}/Cb = {round(Cb,2)} Cw={round(Cw,2)} K*={round(k,2)}", exist_ok=True)
    plt.savefig(f"Fixed Points/{plot_type}/Cb = {round(Cb,2)} Cw={round(Cw,2)} K*={round(k,2)}/Start_Value = {start}.png")
    #plt.clf()
    
    return k, Cw, Cb 
    

