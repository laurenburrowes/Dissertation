import tensorflow as tf
import function as func
import analytical as ana

# Activation Function Options:
# [1.] tf.nn.tanh [2.] tf.nn.sigmoid [3.] tf.nn.elu [4.] tf.nn.relu [5.] tf.nn.selu [6.] tf.nn.gelu [7.] func.linear

####    func.initialise_network(activation_function, width, depth, Cw, Cb, Mu, Nboot, rescale)    ####
# Will instantiate and analyse a network "Nboot" times, plotting its output distributions layer by layer as well as the input weights and bias distributions.

#for i in range(3, 10):
#    for j in range(5, 30):
#        func.initialise_network(tf.nn.tanh, width=j, depth=i, Cw=1, Cb=0, Mu=0, Nboot=5000, rescale=False)

#### func.numerical_analysis(activation_function, initial_width, final_width, depth, Cw, Cb, Mu, Nboot, rescale)
# Instantiated the network and completes a numerical analysis to give values for g0l, g1l and the O(1/n^2) corrections all as a function of width.

#func.numerical_analysis(tf.nn.tanh, initial_width=10, final_width=15, depth=10, Cw=1, Cb=0, Mu=0, Nboot=1000, rescale=False)



#Showing the convergence to a fixed point K*.
# func.analytical_recursion(tf.nn.tanh, 100, 10, 1, 0)


#Analytical recursion finding fixed point(s)
step = 0.5
for i in range(3):
    i += step
    ana.fixed_pt_goal_seek(tf.nn.tanh, i, 0.01)
    if i != 0.5:
        i -= step
        ana.fixed_pt_goal_seek(tf.nn.tanh, i, 0.01)
ana.fixed_pt_goal_seek(tf.nn.tanh, 0.1, 0.01)


#ana.zsq(tf.nn.tanh, 8, 11, 10, 55, 0.01)
#func.numerical_analysis(tf.nn.tanh, 8, 15, 15, 1, 0, 0, 1000, False, 0.1, 0.01)
