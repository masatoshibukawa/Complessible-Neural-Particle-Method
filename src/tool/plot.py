"""
Author: masato shibukawa
Date: 2025-07-22
Email: shibukawa.masato@ac.jaxa.jp

Visualization utilities for neural particle methods simulation results.
Provides plotting functions for pressure, velocity, density fields,
and training loss visualization for fluid dynamics simulations.
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot(x, y, U_plot, filename, dir_path, t_step, lb, ub, args):

    '''Generic formatting'''

    cbformat = ticker.ScalarFormatter()  # create the formatter
    cbformat.set_powerlimits((-2, 2))

    f, ax = plt.subplots(1)

    plt.scatter(x, y, c=U_plot, cmap='hsv', s=1)
    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$y$', fontsize=20)
    plt.xlim(0, args.L * 4.1)
    plt.ylim(0, args.H * 1.1 )

    cb = plt.colorbar(format=cbformat)

    if(filename=='pressure_t'):
        cb.set_label(label="p", fontsize=20)
    elif(filename=='x-velocity_t'):
        cb.set_label(label='v_x', fontsize=20)
    elif(filename=='y-velocity_t'):
        cb.set_label(label='v_y', fontsize=20)
    elif(filename == 'velocity_magnitude_t'):
        cb.set_label(label='v', fontsize=20)
    elif(filename=='divergence_t'):
        cb.set_label(label='div,v', fontsize=20)
    elif(filename=='x-dist-net_t'):
        cb.set_label(label='D_x', fontsize=20)
    elif (filename == 'y-dist-net_t'):
        cb.set_label(label='D_y', fontsize=20)
    elif (filename == 'density_t'):
        cb.set_label(label='rho', fontsize=20)

    for l in cb.ax.yaxis.get_ticklabels():
        # l.set_weight("bold")
        l.set_fontsize(20)

    ax.tick_params(axis='both', labelsize=20)
    major_ticks = np.arange(lb[0], ub[0] + 1e-10, 0.5)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)

    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gca().set_aspect('equal', adjustable='box')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1.0, box.height])

    plt.savefig(dir_path + "/" + filename + str(t_step) + '.png')
    plt.close()

def plot_loss_adam(losses_adam, dir_path, t_step):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_adam, label='Adam')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(dir_path, f'loss_adam_{t_step}.png'))
    plt.close()

def plot_loss_lbfgs(losses_lbfgs, dir_path, t_step):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_lbfgs, label='L-BFGS')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(dir_path, f'loss_lbfgs_{t_step}.png'))
    plt.close()