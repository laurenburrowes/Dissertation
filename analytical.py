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


# Using the second analytical method in the book, pg 126.
# Want to find suitable Cw and Cb for a fixed point K*.   
def fixed_pt_goal_seek(activation_function, start, threshold):
    # Target value of chi_perp/chi_parallel:
    target = 1

    # 2nd derivative of rho^2: Tanh:
    deriv_2_tanh = lambda z: func.avg_rho_sq(activation_function(z)**2 *(z*z - k), k)

    #Linear:
    deriv_2_lin = lambda z: func.avg_rho_sq(z**2 * (z*z - k), k)

    # expectation of rho^2 where rho is the activation function (tanh(z)).
    #exp2 = func.avg_rho_sq(activation_function, k)

    # to find the differential of the activation function. Tanh(z):
    rho_diff_tanh = lambda z: smp.sech(z)*smp.sech(z)

    # Linear = z:
    rho_diff_lin = 1

    # Determine plot type
    plot_type = plot.determine_plot_type(activation_function)

    k_array = []
    k = start
    diff = threshold + 1
    x_array=[]
    i=0
    while abs(diff) > threshold:
        # The expectation value of rho dashed squared, used for perpendicular susceptibility.
        if plot_type == "Tanh":
            exp_dash = func.avg_rho_sq(rho_diff_tanh, k)
            # The expectation value of rho^2(z^2-k), used for parallel susceptibility:
            func2 = lambda  z: activation_function(z)**2 * (z*z - k)
        if plot_type == "Linear":
            exp_dash = func.avg_rho_sq(rho_diff_lin, k)
             #The expectation value of rho^2(z^2-k), used for parallel susceptibility:
            func2 = lambda  z: z**2 * (z*z - k)
        deriv_2 = func.avgrho(func2, k)
        # chi_perp / chi_parallel:
        perppar = (2*(k**2) * exp_dash[0]) / (deriv_2[0]) 
        diff = target - perppar # difference between target value (1) and the current value of chi_perp/chi_parallel
        print(k, perppar, diff)
        k_array.append(k)
        # changing the value of k based on the difference to target:
        k -= abs(diff)/10
        x_array.append(i)
        i+=1 # number of iterations = number of layers

    Cw = 1/exp_dash[0]
    Cb = k - (Cw * func.avg_rho_sq(activation_function, k)[0])
    print("Cw = ", Cw, "Cb = ", Cb, "K* = ", k)

    plt.plot(x_array, k_array)
    plt.xlabel("Layer")
    plt.ylabel("Kernel K")
    plt.title(f"Fixed point K* = {round(k, 2)},\n Cw = {round(Cw, 2)}, Cb = {round(Cb, 2)}")
    os.makedirs(f"Fixed Points/{plot_type}/Cb = {round(Cb,2)} Cw = {round(Cw,2)} K* = {round(k,2)}", exist_ok=True)
    plt.savefig(f"Fixed Points/{plot_type}/Cb = {round(Cb,2)} Cw = {round(Cw,2)} K* = {round(k,2)}/Multiple Start Values.png")
    
    return k, Cw, Cb, k_array 

def zsq(activation_function, initialwidth, finalwidth, depth, start, threshold):
    #Finding the 2 point correlator from our data for K:
    # <z^2,(l)> = K^(l) + 1/n G^1,(l), where G^1,(l) = -1/6 for tanh(z) activation function:


    nums = finalwidth - initialwidth

    k_array = []
    # Taking the K values from our previous analysis
    ks = fixed_pt_goal_seek(activation_function, start, threshold)
    k_array.append(ks[3])

    print("K values", k_array)

    l = 0

    while l < depth:
        l += 1
        print("layer - ", l)
        # Initialising width each time:
        width = initialwidth - 1

        # Reinitialising arrays each time:
        zsql_array = []
        inversewidth_array = []
        # We want the plots PER LAYER, so take over multiple values of width when l is the same, I think loops the wrong way round.
        while finalwidth + 1 >= width:
            width += 1
            zsql = k_array[0][l] + (1/width) * (-1/6)
            inversewidth_array.append(1/width)            
            zsql_array.append(zsql)
            print("layer now - ", l)
        
        # G[1](l) value based on gradient:
        fitted = (zsql_array[nums] - zsql_array[0])/(inversewidth_array[nums] - inversewidth_array[0])

        #Line of best fit:
        fit_line_x = []
        fit_line_x.append(inversewidth_array[0])
        fit_line_x.append(inversewidth_array[nums])

        fit_line_y = []
        fit_line_y.append(zsql_array[0])
        fit_line_y.append(zsql_array[nums])

        #a, b = np.polyfit(inversewidth_array, zsql_array, 1)
        plt.scatter(inversewidth_array, zsql_array, marker="x", label = f"Plotted points using G(1)({l}) = {round(fitted,3)}")
        plt.plot(fit_line_x, fit_line_y, 'r--')
        #plt.plot(inversewidth_array, a*inversewidth_array + b, "r--") #Best fit, not working
        plt.plot(inversewidth_array, zsql_array, "r--") #Just a line through all the points.
        plt.legend()
        plt.xlabel("1/width")
        plt.ylabel("<Z^2>")
        plt.title(f"Layer {l},\n G^(0)({l}) = K^({l}) = {round(k_array[0][l],3)}, G^(1)({l}) = -1/6")
        os.makedirs(f"Plots for 2-pt correlator/Tanh/Comparison/ {initialwidth}w - {finalwidth}w, {depth}d", exist_ok=True)
        plt.savefig(f"Plots for 2-pt correlator/Tanh/Comparison/{initialwidth}w - {finalwidth}w, {depth}d/Layer = {l} for K(0) = {start}.png")
        plt.clf()
        
    return 
