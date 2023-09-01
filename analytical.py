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


    # the first derivative of the activation function = tanh(z):
    rho_diff_tanh = lambda z: smp.sech(z)*smp.sech(z)

    # the first derivative of the linear activation function = z:
    rho_diff_lin = 1


    # Determine plot type
    plot_type = plot.determine_plot_type(activation_function)

    k_array = []
    k = start
    diff = threshold + 1
    l_array=[]
    i=0
    while abs(diff) > threshold:
        # The expectation value of rho dashed squared, used for perpendicular susceptibility.
        if plot_type == "Tanh":
            exp_dash = func.avg_rho_sq(rho_diff_tanh, k)
            # Function used for the expectation value of rho^2(z^2-k), used for parallel susceptibility:
            func2 = lambda  z: activation_function(z)**2 * (z*z - k)
        if plot_type == "Linear":
            exp_dash = func.avg_rho_sq(rho_diff_lin, k)
             #Function used for the expectation value of rho^2(z^2-k), used for parallel susceptibility:
            func2 = lambda  z: z**2 * (z*z - k)
        #Expectation of rho^2(z^2-k):
        deriv_2 = func.avgrho(func2, k)
        # chi_perp / chi_parallel:
        perppar = (2*(k**2) * exp_dash[0]) / (deriv_2[0]) 
        diff = target - perppar # difference between target value, 1, and the current value of chi_perp/chi_parallel
        print(k, perppar, diff)
        k_array.append(k)
        # changing the value of k based on the difference to target:
        k -= abs(diff)/10
        l_array.append(i)
        i+=1 # number of iterations = number of layers

    Cw = 1/exp_dash[0]
    Cb = k - (Cw * func.avg_rho_sq(activation_function, k)[0])
    print("Cw = ", Cw, "Cb = ", Cb, "K* = ", k)


    # Uncomment if you want to output plots:

    # plt.plot(l_array, k_array)
    # plt.xlabel("Layer")
    # plt.ylabel("Kernel K")
    # plt.title(f"Fixed point K* = {round(k, 2)},\n Cw = {round(Cw, 2)}, Cb = {round(Cb, 2)}")
    # os.makedirs(f"Fixed Points/{plot_type}/Cb = {round(Cb,2)} Cw = {round(Cw,2)} K* = {round(k,2)}", exist_ok=True)
    # plt.savefig(f"Fixed Points/{plot_type}/Cb = {round(Cb,2)} Cw = {round(Cw,2)} K* = {round(k,2)}/Multiple Start Values.png")
    
    return k, Cw, Cb, k_array 



# Fixed point finding using the analytical recursion equation of K(l+1) in book
def recursion_k(activation_function, x, Cw, Cb, depth):

    plot_type = plot.determine_plot_type(activation_function)

    k_array = []
    l_array = []
    l = 1

    #Initial recursion eqn:
    kl = Cb + (Cw * (x**2))

    k_array.append(kl)
    l_array.append(l)

    for l in range(2, depth + 1):
        l_array.append(l)

        avg_rho, _, _ = func.avg_rho_sq(activation_function, kl)

        kl = Cb + (Cw * avg_rho)
        k_array.append(kl)

        # Uncomment if you want to plot:

        # plt.plot(l_array, k_array, "x")
        # plt.plot(l_array, k_array, 'r--')
        # plt.title("Recursion for kernel K(l)")
        # plt.suptitle(f"Activation Function - {plot_type}, Depth - {depth}, Input x = {x}")
        # plt.xlabel("Layer")
        # plt.ylabel("Kernel K")
        # os.makedirs(f"Analytical Plots/{plot_type}/K", exist_ok=True)
        # plt.savefig(f"Analytical Plots/{plot_type}/K/start - {x}, {depth}d.png")
        # plt.clf()

    
    return k_array


def recursion_v(activation_function, x, Cb, Cw, depth):
    plot_type = plot.determine_plot_type(activation_function)

    k_array = []
    v_array = []
    l_array = []
    voverk_array = []
    normv_array = []
    logcorrec_array = []
    diff_array = []

    k_array = recursion_k(activation_function, x, Cw, Cb, depth)

    # k_array goes from layer 1 to depth, so 0th entry is layer 1 value of K
    k1 = k_array[0]


    # <rho**4>(1):
    avg_rho41, _ = func.avg_rho_quar(activation_function, k1)
    # <rho**2>**2(1):
    _, _, avg_rho21 = (func.avg_rho_sq(activation_function, k1))

    v1 = 0

    # V for layer 2, since layer 1 V = 0:
    v2 = (Cw**2) * (avg_rho41 - avg_rho21**2)

    func2 = lambda  z: activation_function(z)**2 * (z*z - kl)


    # recursion relation fro the four-point vertex:
    # starting the recursion for 
    for l in range(1, depth + 1):
        l_array.append(l)
        kl = k_array[l-1]

        avg_rho4, _ = func.avg_rho_quar(activation_function, kl)
        _, _, avg_rho2 = func.avg_rho_sq(activation_function, kl)
        xpar = (Cw/(2*kl**2)) * func.avgrho(func2, kl)[0]

        if l == 1:
            vl = v1
        if l == 2:
            vl = v2
        if l > 2:
            vl = (xpar ** 2) * kl * vl + (Cw **2) * (avg_rho4 - (avg_rho2 ** 2))

        # Creating an array of V values per layer:
        v_array.append(vl)


        # Now want to calculate the 1/n dependance
        normv = (2/3) * l
        voverk = vl/(kl**2)
        logcorrec = (np.log(l))

        diff = voverk - normv
        diff_array.append(diff)

        voverk_array.append(voverk)
        normv_array.append(normv)
        # logcorrec_array.append(logcorrec)

    plt.plot(l_array, v_array, "x")
    plt.plot(l_array, v_array, 'r--')
    plt.title(f"Recursion for four-point vertex V(l)")
    plt.suptitle(f"Activation Function - {plot_type}, Depth - {depth}, Input x = {x}")
    plt.xlabel("Layer")
    plt.ylabel("4-Pt Vertex V")
    os.makedirs(f"Analytical Plots/{plot_type}/V", exist_ok=True)
    plt.savefig(f"Analytical Plots/{plot_type}/V/start - {x}, {depth}d.png")
    plt.clf()

    plt.plot(l_array, diff_array, "x")
    plt.plot(l_array, diff_array, 'r--')
    plt.title(f"Difference between $V(l)/K(l)^2$ and $2/3 * l$")
    plt.suptitle(f"Activation Function - {plot_type}, Depth - {depth}, Input x = {x}")
    plt.xlabel("Layer")
    plt.ylabel("Diff")
    os.makedirs(f"Analytical Plots/{plot_type}/Diff", exist_ok=True)
    plt.savefig(f"Analytical Plots/{plot_type}/Diff/start - {x}, {depth}d.png")
    plt.clf()

    plt.plot(l_array, voverk_array, "x")
    plt.plot(l_array, voverk_array, 'r--')
    plt.title(f"V(l)/K(l)^2")
    plt.suptitle(f"Activation Function - {plot_type}, Depth - {depth}, Input x = {x}")
    plt.xlabel("Layer")
    plt.ylabel("$V/K^2$")
    os.makedirs(f"Analytical Plots/{plot_type}/VoverK^2", exist_ok=True)
    plt.savefig(f"Analytical Plots/{plot_type}/VoverK^2/start - {x}, {depth}d.png")
    plt.clf()

    plt.plot(l_array, normv_array, "x")
    plt.plot(l_array, normv_array, 'r--')
    plt.title("$(2/3)l$")
    plt.suptitle(f"Activation Function - {plot_type}, Depth - {depth}, Input x = {x}")
    plt.xlabel("Layer")
    plt.ylabel("$V/K^2$")
    os.makedirs(f"Analytical Plots/{plot_type}/Normalised V", exist_ok=True)
    plt.savefig(f"Analytical Plots/{plot_type}/Normalised V/start - {x}, {depth}d.png")
    plt.clf()



    return v_array







