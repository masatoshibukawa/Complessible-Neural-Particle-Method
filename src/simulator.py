"""
Author: masato shibukawa
Date: 2025-07-22
Email: shibukawa.masato@ac.jaxa.jp

Simulator class for Neural Particle Methods (NPM) and Compressible Neural Particle Methods (cNPM).
Manages the time-stepping simulation loop, handles model training at each time step,
and coordinates data I/O for fluid dynamics simulations.
"""

import os
import time
import torch
import scipy.io
import numpy as np
from model.NPM import *
from tool.plot import plot, plot_loss_adam, plot_loss_lbfgs
from tool.updateWall import updateWall

class Simulator:
    def __init__(self, path_dir, device, args, logger):
        '''Initial conditions'''
        self.args             = args
        self.eta              = args.eta 
        self.height           = args.H
        self.mu               = args.mu
        self.q                = args.q
        self.lbfgs_flg        = args.lbfgs_flg
        self.tl_flg           = args.tl_flg
        self.epoch            = args.epoch
        self.model_name       = args.model
        self.hidden_layers    = args.hidden_layers
        self.device           = device
        self.ic_bc_dir_path   = path_dir["ic_bc"]
        self.figure_dir_path  = path_dir["figure"]
        self.weights_dir_path = path_dir["weights"]
        self.loss_dir_path    = path_dir["loss"]
        self.IRK_dir_path     = path_dir["IRK"]
        self.this_dir_path    = os.path.dirname(os.path.abspath(__file__))
        self.logger           = logger
    
    def runDamBreak(self):
        '''Read time step from file'''
        TempDisc   = scipy.io.loadmat(self.ic_bc_dir_path + '/timeStep.mat')
        dt         = TempDisc['dt'][0,0]
        idt        = TempDisc['idt'][0]
        t_step     = int(idt[0])
        idt        = np.delete(idt, 0)
        scipy.io.savemat(self.ic_bc_dir_path + '/timeStep.mat', {'dt': dt, 'idt': idt})

        for arg in vars(self.args):
            self.logger.append(f'{arg}: {getattr(self.args, arg)}')
        self.logger.append('tstep: %d' % t_step)

        '''Read model parameter from file'''
        IC = scipy.io.loadmat(self.ic_bc_dir_path + '/IC_t' + str(t_step-1) + '.mat')['IC']
        BC = scipy.io.loadmat(self.ic_bc_dir_path + '/BC_t' + str(t_step-1) + '.mat')

        '''Initial conditions'''
        x        = IC[:, 0].flatten()[:, None]
        y        = IC[:, 1].flatten()[:, None]
        u0       = IC[:, 2:5] # vx:2, vy:3, p:4
        rho0     = IC[:, 5].flatten()[:, None]

        '''Boundary conditions'''
        id_u1_x   = BC['id_u1_x'][0]
        id_u1_y   = BC['id_u1_y'][0]
        id_u1_p   = BC['id_u1_p']
        id_free_p = BC['id_free_p']
        id_free_x = BC['id_free_x'][0]
        id_free_y = BC['id_free_y'][0]
        id_same   = BC['id_same'][0]

        sim_id_u1_p = np.intersect1d(id_same, id_u1_p) # (59) 自由表面に接している粒子のインデックス

        '''Lower and upper bounds'''
        lb = np.array([np.min(x), np.min(y)])
        ub = np.array([np.max(x), np.max(y)])

        '''build model'''
        if self.model_name == "NPM":
            #layers = [2, 60, 60, (3*(self.q+1)-1)]
            layers = [2] + self.hidden_layers + [(3*(self.q+1)-1)]
            Model = NeuralParticleMethods(
                    id_same, sim_id_u1_p,
                    x, y, rho0[id_same], u0,
                    layers, dt, lb, ub,
                    self.q, self.mu, t_step, self.device,
                    self.IRK_dir_path, self.lbfgs_flg, self.args, self.logger
                    )
        elif self.model_name == "gNPM":
            #layers = [2, 50, 50, 50, 50, (2*(self.q+1)+1)]
            layers = [2] + self.hidden_layers + [(2*(self.q+1)+1)]
            Model = generalNeuralParticleMethods(
                    id_same, sim_id_u1_p,
                    x, y, rho0[id_same], u0,
                    layers, dt, lb, ub,
                    self.q, self.mu, t_step, self.device,
                    self.IRK_dir_path, self.lbfgs_flg, self.args, self.logger
                    )
        elif self.model_name == "gNPM_old":
            #layers = [2, 50, 50, 50, 50, (2*(self.q+1)+1)]
            layers = [2] + self.hidden_layers + [(2*(self.q+1)+1)]
            Model = generalNeuralParticleMethods_old(
                    id_same, sim_id_u1_p,
                    x, y, rho0[id_same], u0,
                    layers, dt, lb, ub,
                    self.q, self.mu, t_step, self.device,
                    self.IRK_dir_path, self.lbfgs_flg, self.args, self.logger
                    )
        elif self.model_name == "cNPM":
            layers = [2] + self.hidden_layers + [(3*(self.q+1)+1)]
            Model = complessibleNeuralParticleMethods(
                    id_same, sim_id_u1_p,
                    x, y, rho0[id_same], u0,
                    layers, dt, lb, ub,
                    self.q, self.mu, t_step, self.device,
                    self.IRK_dir_path, self.lbfgs_flg, self.args, self.logger,
                    self.eta, self.height)
        elif self.model_name == "cNPM_old":
            layers = [2] + self.hidden_layers + [(2*(self.q+1))]
            Model = complessibleNeuralParticleMethods_old(
                    id_same, sim_id_u1_p,
                    x, y, rho0[id_same], u0,
                    layers, dt, lb, ub,
                    self.q, self.mu, t_step, self.device,
                    self.IRK_dir_path, self.lbfgs_flg, self.args, self.logger,
                    self.eta, self.height)
        else:
            assert False, "Model name is not correct"

        # 事前学習を行う
        if t_step > 1 and self.tl_flg and "cNPM" in self.model_name:
            params_file_name = f"weights_t{t_step - 1}.pth"
            params_file_path = os.path.join(self.weights_dir_path, params_file_name)
            weight_params    = torch.load(params_file_path, weights_only=True)
            Model.model.load_state_dict(weight_params)
            self.logger.append('Pre-trained model loaded')

        # 学習を行う
        start_time   = time.time()
        losses_adam  =  Model.train(self.epoch)
        elapsed_time = time.time() - start_time
        self.logger.append('elapsed_time: %.2f' % elapsed_time)
        torch.cuda.empty_cache()

        # 重み、損失を保存
        Model.save_model(os.path.join(self.weights_dir_path, 'weights_t' + str(t_step) + '.pth'))
        np.save(os.path.join(self.loss_dir_path, 'losses_adam_t' + str(t_step) + '.npy'), losses_adam)
        #plot_loss_adam(losses_adam,   self.loss_dir_path, t_step)

        ''' valiables explanation
        U1_x        : 次のステップのx速度 (60543, 5)
        U1_y        : 次のステップのy速度 (60543, 5)
        divU_out    : div v              (60543, 1)
        detF        : det F              (60543, 1)
        x_out       : 次のステップのx座標 (60543, 1)
        y_out       : 次のステップのy座標 (60543, 1)
        P_hat      : 次のステップの圧力  (60543, 1)
        '''

        predicted_states = Model.predict(x, y)
        
        U1_x     = predicted_states[0].detach().to('cpu').numpy()
        U1_y     = predicted_states[1].detach().to('cpu').numpy()
        x_out    = predicted_states[4].detach().to('cpu').numpy()
        y_out    = predicted_states[5].detach().to('cpu').numpy()
        P_hat    = predicted_states[6].detach().to('cpu').numpy()
        
        if "cNPM" in self.model_name:
            rho_out  = predicted_states[7].detach().to('cpu').numpy()
        else:
            divU_out = predicted_states[2].detach().to('cpu').numpy()
            rho_out  = rho0.copy()
        
        """
        Shape of U1_x    : (6843, 5)
        Shape of U1_y    : (6843, 5)
        Shape of divU_out: (6843, 1)
        Shape of x_out   : (6843, 1)
        Shape of y_out   : (6843, 1)
        Shape of P_hat   : (6843, 5)
        Shape of rho_out : (6843, 5)
        """

        '''Update solution
        U_x_plot    : 次のステップのx速度, n+1のみ (60543, 1)
        U_y_plot    : 次のステップのy速度, n+1のみ (60543, 1)
        '''

        U_x_plot   = np.zeros((len(y), 1))
        U_y_plot   = np.zeros((len(y), 1))
        P_plot     = np.zeros((len(y), 1))  
        rho_plot   = np.zeros((len(y), 1))

        U_x_plot[:, 0] = np.transpose(U1_x[:, self.q]).copy()
        U_y_plot[:, 0] = np.transpose(U1_y[:, self.q]).copy()
        P_plot         = P_hat.copy()
        rho_plot       = rho_out.copy()

        u0[id_free_x, 0] = U_x_plot[id_free_x, 0].copy() # 境界条件に該当しない全ての粒子の速度を更新
        u0[id_free_y, 1] = U_y_plot[id_free_y, 0].copy() # 境界条件に該当しない全ての粒子の速度を更新
        u0[:, 2]         = P_hat[:, 0].copy()            # 全粒子の圧力を更新
        
        '''Update coordinates (x,y) and density'''
        x[id_free_x] = x_out[id_free_x].copy()# 自由粒子のx座標を更新
        y[id_free_y] = y_out[id_free_y].copy() # 自由粒子のy座標を更新

        '''Update index list'''
        # 壁に接している粒子のインデックスを更新,境界での反射やその他の挙動を管理する
        x, y, id_u1_x, id_u1_y, u0 = updateWall(x, y, id_u1_x, id_u1_y, u0)

        for iter in range(0, id_u1_p.shape[0]):
            if(x[iter]<1e-3):
                id_u1_p = np.delete(id_u1_p, iter)

        id_free_x = np.arange(0, len(x), 1)
        id_free_x = np.delete(id_free_x, id_u1_x)

        id_free_y = np.arange(0, len(y), 1)
        id_free_y = np.delete(id_free_y, id_u1_y)

        '''Plot field'''
        U_mag_plot = np.sqrt(np.square(U_x_plot) + np.square(U_y_plot))

        plot(x, y, P_plot, 'pressure_t', self.figure_dir_path, t_step, lb, ub, self.args)
        plot(x, y, U_x_plot, 'x-velocity_t', self.figure_dir_path, t_step, lb, ub, self.args)
        plot(x, y, U_y_plot, 'y-velocity_t', self.figure_dir_path, t_step, lb, ub, self.args)
        plot(x, y, U_mag_plot, 'velocity_magnitude_t', self.figure_dir_path, t_step, lb, ub, self.args)
        plot(x[id_u1_p], y[id_u1_p], P_plot[id_u1_p], 'p-boun_t', self.figure_dir_path, t_step, lb, ub, self.args)
        plot(x[id_u1_x], y[id_u1_x], U_y_plot[id_u1_x], 'x-boun_t', self.figure_dir_path, t_step, lb, ub, self.args)
        plot(x[id_u1_y], y[id_u1_y], U_x_plot[id_u1_y], 'y-boun_t', self.figure_dir_path, t_step, lb, ub, self.args)

        if "cNPM" in self.model_name:
            plot(x, y, rho_plot, 'density_t', self.figure_dir_path, t_step, lb, ub, self.args)
        else:
            plot(x, y, divU_out[:, -1], 'divergence_t', self.figure_dir_path, t_step, lb, ub, self.args)

        '''Save result to file'''
        if "cNPM" in self.model_name:
            if P_hat.shape[1] == 1:
                P_hat = P_hat.reshape(-1, 1)  # P_hatを2次元に変換

        IC = np.concatenate((x, y, u0, rho_out), axis=1)
        scipy.io.savemat(os.path.join(self.ic_bc_dir_path, 'IC_t' + str(t_step) + '.mat'), 
                        {'IC': IC})
        scipy.io.savemat(os.path.join(self.ic_bc_dir_path, 'BC_t' + str(t_step) + '.mat'), 
                        {'id_same': id_same, 
                        'id_u1_x': id_u1_x, 
                        'id_u1_y': id_u1_y, 
                        'id_u1_p': id_u1_p, 
                        'id_free_x': id_free_x, 
                        'id_free_y': id_free_y, 
                        'id_free_p': id_free_p})

        return elapsed_time