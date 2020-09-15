#
# A model of odor concentration classification task
#
# Author: Naoki Hiratani (N.Hiratani@gmail.com)
#
import sys
from math import *

from data_loader import load_data
import jax.numpy as jnp
import jax.scipy.special as jscisp
import jax.ops as jops
from jax import grad
from jax import random
import matplotlib.pyplot as plt

concs = jnp.array([0.033, 0.1, 0.33, 1.0, 3.3, 10.0]) #odor concentration
clen = len(concs)
clrs = ['#00E5EE', '#E34234']

def phi(dtmp, sigma2tmp): #cumulative distribution
    cones = jnp.ones((clen))
    return 0.5*( cones + jscisp.erf( jnp.divide(dtmp, jnp.sqrt(2.0*sigma2tmp) ) ) )

def accuracy(cmin, r_low, r_high, a1, a2): #estimate the accuracy
    cones = jnp.ones((clen))
    
    mus = jnp.log( (1.0/cmin)*concs )
    sigma2s = a1*mus + a2*jnp.multiply(mus, mus)
    
    log_lowc_th = mus[2] + r_low*(mus[5]-mus[2])
    log_highc_th = mus[0] + r_high*(mus[3]-mus[0])
    
    lowc_accuracy = jnp.zeros((clen))
    highc_accuracy = jnp.zeros((clen))
    diff_target_low = 0.0; diff_target_high = 0.0
    
    lowc_phi = phi(log_lowc_th*cones - mus, sigma2s)
    highc_phi = phi(log_highc_th*cones - mus, sigma2s)
    
    e1 = jnp.array([0, 0, 0, 1.0, 1.0, 1.0])
    e2 = jnp.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    lowc_accuracy = e1 + jnp.multiply(e2, lowc_phi)
    highc_accuracy = e1 + jnp.multiply(e2, highc_phi)

    return lowc_accuracy, highc_accuracy

def loss(cmin, r_low, r_high, a1, a2): #calculate the loss
    target_low, target_high = load_data();
    lowc_accuracy, highc_accuracy = accuracy(cmin, r_low, r_high, a1, a2)
    
    return jnp.mean( jnp.square(lowc_accuracy-0.01*target_low) ) + jnp.mean( jnp.square(highc_accuracy-0.01*target_high) )

def plot_fig(ps): #plot
    target_low, target_high = load_data();
    lowc_accuracy, highc_accuracy = accuracy(ps[0], ps[1], ps[2], ps[3], ps[4])
    
    plt.rcParams.update({'font.size': 16})
    plt.plot(jnp.log(concs), lowc_accuracy, 'o-', color=clrs[0])
    plt.plot(jnp.log(concs), highc_accuracy, 'o-', color=clrs[1])
    plt.plot(jnp.log(concs), 0.01*target_low, 's--', color=clrs[0])
    plt.plot(jnp.log(concs), 0.01*target_high, 's--', color=clrs[1])
    plt.xticks(jnp.log(concs), concs)
    plt.show()

def fitting(eta, ilmax, ik):
    key = random.PRNGKey(ik) #key of the random number generator
    
    #ps: (cmin, r_low, r_high, a1, a2)
    plen = 5 #number of parameters
    pranges = jnp.array([[0.00003, 0.0, 0.0, 0.0, 0.0],\
                         [0.015,   1.0, 1.0, 1.0, 1.0]])
    pscales = jnp.array([0.01, 1.0, 1.0, 1.0, 0.1])
    ps = jnp.multiply(pscales, random.uniform(key, shape=([plen]), minval=0.0, maxval=1.0))
    
    grad_func = []; pgrads = []
    for q in range(plen):
        grad_func.append( grad(loss, argnums=q) )
        pgrads.append(0.0)
    
    #optimization
    for il in range(ilmax):
        for q in range(plen):
            pgrads[q] = grad_func[q](ps[0], ps[1], ps[2], ps[3], ps[4])
        ps = ps - eta*jnp.multiply(pscales, jnp.array(pgrads))
        ps = jnp.clip(ps, pranges[0], pranges[1])
    print('ps: ', ps)
    plot_fig(ps)


def main():
    param = sys.argv
    eta = float(param[1]) #learning rate
    ilmax = int(param[2]) #number of iterations
    ik = int(param[3]) #random number seed
    
    fitting(eta, ilmax, ik)

if __name__ == "__main__":
    main()


