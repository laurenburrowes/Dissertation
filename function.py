#This code contains everything to do with the neural network and extracting its statistics
import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
import tensorflow as tf
import plot
from scipy.optimize import curve_fit
from scipy import integrate
import os

#Generates the weights and bias's for the network by drawing them at random from some normal distribution
def initialization(input_size_, hidden_size_, Mu_, STD_, nl, weightorbias):

    if weightorbias == "weight":
        Wname = 'W{}'.format(nl)
        out = tf.Variable(tf.random.normal([input_size_, hidden_size_], mean=Mu_, stddev=STD_), name=Wname)

    if weightorbias == "bias":
        bname = 'b{}'.format(nl)
        out = tf.Variable(tf.random.normal([hidden_size_], mean=Mu_, stddev=STD_), name=bname)

    return out


#This function propagates the signal through the network
def forward(X, i, activation_function, width_, depth_, Ws, Bs, Layer_out_matrix, Layer_zsq_matrix):

    with tf.GradientTape() as tape:
        tape.watch([param for param in [Ws[i, j] for j in range(depth_)] + [Bs[i, j] for j in range(depth_)]])

        #Setting the preactivation of the first hidden layer as the input data set
        A = X

        for l in range(depth_):
            layer_width = width_[l+1]
            #Calculating the preactivation and activation function for a given hidden layer
            Z = tf.add(tf.matmul(A, Ws[i, l]), Bs[i, l])
            A = activation_function(Z)

            zsq = 0

            for k in tf.get_static_value(Z[0]):
                z = k
                zsq += z**2

            
            Layer_zsq_matrix[i, l] = zsq/layer_width 

            #Appending the output distribution for each layer of the network
            Layer_out_matrix[i, l] = Z


        Z_last = Z

    return Z_last


#Used to exract data from the weights and bias generation, allows for histogram plot
def bootobj_extr(layer_, el1_, el2_, Nboot, Ws, Bs):

    A = np.zeros(Nboot)
    B = np.zeros(Nboot)

    for i in range(Nboot):
        matrW = Ws[i, layer_]
        A[i] = matrW[el1_, el2_]
        vecB = Bs[i, layer_]
        B[i] = vecB[el1_]

    return A, B


def avg_zl_sq(g):

    coeff = 1 / (np.sqrt(2 * math.pi * g))
    integrand = lambda z: (np.exp((-1 / 2) * (1 / g) * (z ** 2))) * (z ** 2)
    integral = integrate.quad(integrand, -np.inf, np.inf)
    value = coeff * integral[0]
    est_error = integral[1]

    return value, est_error


def zl_sq(inv_w, kl, g1l, correc):
    return kl + (inv_w * g1l) + (correc*(inv_w**2))


#This function initiates the neural network with the given inputs and gives the statistics of each step
def initialise_network(activation_function, width, depth, Cw, Cb, Mu, Nboot, rescale, x_input):

    print("\n----Initiating----")

    #Input and output layer size (number of neurons)
    input_size = 1
    output_size = 1

    plot_type = plot.determine_plot_type(activation_function)

    #Prints the networks input statistics
    net_stat = f"Nboot={Nboot}, Cb={Cb} & Cw={Cw}, Mu={round(Mu, 2)}"
    print(f"Input Gaussian Values: {net_stat}")

    #Creates an array with sizes of each layer as each value
    lsizes = np.concatenate(([input_size], np.repeat(width, depth - 1), [output_size]))

    #Prints the network architecture
    net_arch = f"Width = {width} & Depth = {depth}"
    print(f"Network Architecture: {net_arch} \n{lsizes}")

    #Generating the weights and bias matrices
    Ws = np.zeros((Nboot, depth), dtype=object)
    Bs = np.zeros((Nboot, depth), dtype=object)

    #Generating the weights and biases for each layer for all instantiations of the network
    #Nboot is the number of times the network is initiated
    for i in range(Nboot):
        for l in range(depth):
            if rescale == True:
                Bs[i, l] = initialization(lsizes[l], lsizes[l + 1], Mu, np.sqrt(Cb) / lsizes[l], l, weightorbias="bias")
                Ws[i, l] = initialization(lsizes[l], lsizes[l + 1], Mu, np.sqrt(Cw) / lsizes[l], l, weightorbias="weight")
            if rescale == False:
                Bs[i, l] = initialization(lsizes[l], lsizes[l + 1], Mu, np.sqrt(Cb), l, weightorbias="bias")
                Ws[i, l] = initialization(lsizes[l], lsizes[l + 1], Mu, np.sqrt(Cw)/(np.sqrt(lsizes[l])), l, weightorbias="weight")

    #Filtering the input bias and weight distribution data to plot on a histogram
    Whist, Bhist = bootobj_extr(1, 1, 1, Nboot=Nboot, Ws=Ws, Bs=Bs)

    #Plotting the bias input distribution
    #plot.bias_input_dist(x_input, Bhist, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    #Plotting the weight input distribution
    #plot.weight_input_dist(x_input, Whist, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    # Input quantities
    X = tf.constant([[30]], dtype=tf.float32)

    #L'th layer output matrix
    Lom = np.zeros((Nboot, depth), dtype=object)
    #z^2 matrix
    Lzsqm = np.zeros((Nboot, depth), dtype=object) 

    #Final layer output layer array
    output_array = []

    #Propagates the signal throughout the network for every instantiation
    for i in range(Nboot):
        Z_array = forward(X, i, activation_function, lsizes, depth, Ws=Ws, Bs=Bs, Layer_out_matrix=Lom, Layer_zsq_matrix=Lzsqm)
        output_array.append(Z_array.numpy()[0][0])

    #Plot the final layer output distribution
    #plot.final_out_dist(output_array, width, depth, Cw, Cb, Mu, Nboot, plot_type, rescale)

    #Arrays to store values for FWHM of fitted data
    FWHM_array = []
    STD_fit_array = []
    STD_data_array = []
    data_zsq = []

    #Filtering the array data for the output distribution of each layer so we can plot it
    for l in range(depth):
        Z = np.zeros(Nboot)
        zsq = 0
        zsq_std_array = []
        for i in range(Nboot):
            matrLom = Lom[i, l]
            Z[i] = matrLom[0, 0]
            # Calculating <Zsq> directly from the output data for every layer
            zsq += (Lzsqm[i, l])/Nboot
            data_zsq.append(zsq)


        #Plots the output distribution for every layer and appending each layers FWHM
        FWHM, STD_fit, STD_data = plot.output_dist_per_layer(Z, width, depth, l, Cw, Cb, Mu, Nboot, plot_type, rescale)
        FWHM_array.append(FWHM)
        STD_fit_array.append(STD_fit)
        STD_data_array.append(STD_data)

    #Plotting the FWHM vs depth of the network
    #plot.gaussian_width_and_depth(width, depth, Cw, Cb, Mu, Nboot, FWHM_array, plot_type)

    ### <Zsq> using the fitted output data and integral ###
        fit_zsq = []
        error_fit_zsq = []


    # Calculating the <Zsq> directly from the variance of the fitted gaussian output distributions for every layer
        for sigma in STD_fit_array:
            g = sigma**2
            outputs = avg_zl_sq(g)
            fit_zsq.append(outputs[0])
            error_fit_zsq.append(outputs[1])

    print("----Complete----")

    return fit_zsq, error_fit_zsq, data_zsq



def avg_rho_sq(activation_function, k):
    coeff = 1 / (np.sqrt(2 * math.pi * k))
    if activation_function == 1:
        integrand = lambda z: (np.exp((-1 / 2) * (1 / k) * (z ** 2))) * (1 ** 2)
    else:
        integrand = lambda z: (np.exp((-1 / 2) * (1 / k) * (z ** 2))) * (activation_function(z) ** 2)
    integral = integrate.quad(integrand, -np.inf, np.inf)
    value = coeff * integral[0]
    est_error = integral[1]

    return value, est_error, value**2


def avgrho(function, k):
    coeff = 1 / (np.sqrt(2 * math.pi * k))
    integrand = lambda z: (np.exp((-1 / 2) * (1 / k) * (z ** 2))) * (function(z))
    integral = integrate.quad(integrand, -np.inf, np.inf)
    value = coeff * integral[0]
    est_error = integral[1]

    return value, est_error

def avg_rho_quar(activation_function, k):
    coeff = 1/ (np.sqrt(2 * math.pi * k))
    if activation_function == 1:
        integrand = lambda z: (np.exp((-1 / 2) * (1 / k) * (z ** 2))) * (1 ** 2)
    else:
        integrand = lambda z: (np.exp((-1 / 2) * (1 / k) * (z ** 2))) * (activation_function(z) ** 4)
    integral = integrate.quad(integrand, -np.inf, np.inf)
    value = coeff * integral[0]
    est_error = integral[1]

    return value, est_error

def rho_diff(activation_function, diff_times):
    rhodiff = lambda z: smp.diff(activation_function(z), z, diff_times)

    return rhodiff[0]
