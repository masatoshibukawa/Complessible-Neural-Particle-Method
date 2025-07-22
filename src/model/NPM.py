"""
Author: masato shibukawa
Date: 2025-07-22
Email: shibukawa.masato@ac.jaxa.jp

Neural Particle Methods (NPM) implementation with multiple model variants:
- NeuralParticleMethods: Standard incompressible NPM
- complessibleNeuralParticleMethods: Compressible NPM with density evolution
- generalNeuralParticleMethods: General NPM formulation (legacy variants available)

These models implement physics-informed neural networks for fluid dynamics simulations
using implicit Runge-Kutta time integration and Lagrangian particle tracking.

Based on the PINNs framework by Maziar Raissi: https://github.com/maziarraissi/PINNs
"""

import os
import torch
import numpy as np
from torch import nn
from model.PINNs import PhysicsInformedNNs
from model.EarlyStopping import EarlyStopping
import copy
class NeuralParticleMethods():
    def __init__(self, id_same, id_u1_p, x0, y0, rho0, u0, layers, dt, lb, ub, q, mu, t_step, device, dir_path_IRK, LBFGS_FLG, args, logger):
        
        self.logger = logger
        self.args   = args

        self.device = device
        self.t_step = t_step

        # fixed and free node indices
        self.id_u1_p = id_u1_p

        # material data
        self.rho0  = rho0
        self.mu    = mu

        # gravity acceleration
        self.b_x =  0.0
        self.b_y = -9.81

        # lower and upper bound
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)

        # initial coordinates and solution
        self.x0 = x0[id_same].flatten()[:, None]     # initial x-position
        self.y0 = y0[id_same].flatten()[:, None]     # initial y-position

        # domain height and width for distance function
        self.width  = np.max(x0) - np.min(x0)
        self.height = np.max(y0) - np.min(y0)
        self.u0_x = u0[id_same, 0].flatten()[:, None]    # initial x-velocity
        self.u0_y = u0[id_same, 1].flatten()[:, None]    # initial y-velocity
        self.u0   = u0

        # boundary pressure
        self.x1_p = x0[id_u1_p].flatten()[:, None]          # boundary x-position
        self.y1_p = y0[id_u1_p].flatten()[:, None]          # boundary y-position

        # others
        self.layers = layers
        self.dt     = dt
        self.q      = max(q,1)
        self.activation_func = nn.Tanh()

        self.logger.append(f"device : {self.device}")
        if device == 'cuda':
            self.model = torch.nn.DataParallel(PhysicsInformedNNs(layers, self.ub, self.lb, self.activation_func)).to(self.device)
        else:
            self.model = PhysicsInformedNNs(layers, self.ub, self.lb, self.activation_func).to(self.device)
        #self.model           = torch.jit.script(self.model)
        self.optimizer       = torch.optim.Adam(self.model.parameters(), lr = args.learning_rate)
        self.optimizer_LBFGS = torch.optim.LBFGS(self.model.parameters(), 
                                        lr               = 1.0, 
                                        max_iter         = 100000,
                                        max_eval         = 50000,
                                        history_size     = 100,
                                        tolerance_grad   = 1e-5, 
                                        tolerance_change = 1.0 * np.finfo(float).eps, 
                                        line_search_fn   = "strong_wolfe")
        self.iter            = 0
        
        self.early_stopping  = EarlyStopping(patience=args.early_stopping_patience,
                                            verbose=False, 
                                            )
        self.criteria        = nn.MSELoss(reduction='sum')
        self.LBFGS_FLG       = LBFGS_FLG
        self.best_parameters = copy.deepcopy(self.model.state_dict())
        
        # Load IRK weights
        IRK_path         = os.path.join(dir_path_IRK, f"Butcher_IRK{q}.txt")
        tmp              = np.float32(np.loadtxt(IRK_path, ndmin=2))
        self.IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))
        self.IRK_times   = tmp[q**2+q:]

        # convert to tensor
        self.x0_t   = torch.tensor(self.x0,   dtype=torch.float32, device=device, requires_grad=True)
        self.y0_t   = torch.tensor(self.y0,   dtype=torch.float32, device=device, requires_grad=True)
        self.rho0_t = torch.tensor(self.rho0, dtype=torch.float32, device=device, requires_grad=True)
        self.u0_x_t = torch.tensor(self.u0_x, dtype=torch.float32, device=device, requires_grad=True)
        self.u0_y_t = torch.tensor(self.u0_y, dtype=torch.float32, device=device, requires_grad=True)
        self.u0_t   = torch.tensor(self.u0,   dtype=torch.float32, device=device, requires_grad=True)
        self.x1_p_t = torch.tensor(self.x1_p, dtype=torch.float32, device=device, requires_grad=True)
        self.y1_p_t = torch.tensor(self.y1_p, dtype=torch.float32, device=device, requires_grad=True)
        self.IRK_weights_t = torch.tensor(self.IRK_weights, dtype=torch.float32, device=device, requires_grad=True)

    def fwd_gradients_U_x(self, U, x):
        dummy = torch.ones_like(U, requires_grad=True)
        g = torch.autograd.grad(
            U, x, 
            grad_outputs = dummy,
            create_graph = True,
            )[0]
        dU_dx = torch.autograd.grad(
            g, dummy,
            grad_outputs = torch.ones_like(x, requires_grad=True),
            create_graph = True,
            )[0]
        return dU_dx

    def _distance_function(self, X, Y, c_rank = 1):
        '''Distance function and boundary extension'''
        if c_rank == 1:
            D_x    = X.clone().detach() / self.width
            D_x_x  = 1 / self.width
            D_x_y  = 0
        elif c_rank == 2:
            D_x    = -4 * torch.pow(X.clone().detach(), 2) / self.width**2 + 4 * X.clone().detach() / self.width
            D_x_x  = -8 * X.clone().detach() / self.width**2 + 4 / self.width
            D_x_y  = 0
        else:
            self.logger.append('Error: c_rank = %d has not been developed!!' % (c_rank))
        D_y    = Y.clone().detach() / self.height
        D_y_x  = 0
        D_y_y  = 1 / self.height
        return D_x, D_x_x, D_x_y, D_y, D_y_x, D_y_y
    
    def _fetch_output(self, U1, D_x, D_y):
        '''RK velocity stages & solution'''
        U1_hat_x = U1[:, 0:self.q+1]
        U1_hat_y = U1[:, (self.q+1):(2*self.q+2)]

        U1_x  = D_x*U1[:, 0:(self.q+1)] #[799, 21]
        U1_y  = D_y*U1[:, (self.q+1):(2*self.q + 2)] #[799, 21]

        '''Pressure RK stages'''
        p_stages   = U1[:, (2*self.q+2):(3*self.q+2)]      # RK pressure stages
        return U1_hat_x, U1_hat_y, U1_x, U1_y, p_stages

    def _fetch_output_for_output_solution(self, U1, D_x, D_y):
        '''RK velocity stages & solution'''
        U1_hat_x = U1[:, 0:self.q+1]
        U1_hat_y = U1[:, (self.q+1):(2*self.q+2)]

        U1_x  = D_x * U1_hat_x
        U1_y  = D_y * U1_hat_y

        '''Pressure RK stages'''
        p_stages   = U1[:, (2*self.q+2):(3*self.q+2)]      # RK pressure stages
        return U1_hat_x, U1_hat_y, U1_x, U1_y, p_stages

    def _compute_F_inv(self, D_x, D_x_x, D_x_y, D_y, D_y_x, D_y_y, U1_hat_x, U1_hat_y, X, Y):
        '''Grad v'''
        dot_F_11 = D_x*self.fwd_gradients_U_x(U1_hat_x, X) + D_x_x*U1_hat_x
        dot_F_12 = D_x*self.fwd_gradients_U_x(U1_hat_x, Y) + D_x_y*U1_hat_x
        dot_F_21 = D_y*self.fwd_gradients_U_x(U1_hat_y, X) + D_y_x*U1_hat_y
        dot_F_22 = D_y*self.fwd_gradients_U_x(U1_hat_y, Y) + D_y_y*U1_hat_y

        '''Delta F'''
        F_11 = self.dt*torch.matmul(dot_F_11[:, 0:self.q], self.IRK_weights_t.T) + 1
        F_12 = self.dt*torch.matmul(dot_F_12[:, 0:self.q], self.IRK_weights_t.T)
        F_21 = self.dt*torch.matmul(dot_F_21[:, 0:self.q], self.IRK_weights_t.T)
        F_22 = self.dt*torch.matmul(dot_F_22[:, 0:self.q], self.IRK_weights_t.T) + 1

        '''Inverse of deformation gradient'''
        detF = torch.multiply(F_11, F_22) - torch.multiply(F_12, F_21)

        F_inv_11 =  torch.divide(F_22, detF)
        F_inv_12 = -torch.divide(F_12, detF)
        F_inv_21 = -torch.divide(F_21, detF)
        F_inv_22 =  torch.divide(F_11, detF)

        F_inv = [F_inv_11, F_inv_12, F_inv_21, F_inv_22]
        dot_F = [dot_F_11, dot_F_12, dot_F_21, dot_F_22]

        return F_inv, dot_F, detF
    
    def _compute_F_inv_for_output_solution(self, D_x, D_x_x, D_x_y, D_y, D_y_x, D_y_y, U1_hat_x, U1_hat_y, X, Y):
        b_j  = self.IRK_weights_t[self.q:(self.q+1),:].T
        '''Grad v'''
        dot_F_11 = D_x * self.fwd_gradients_U_x(U1_hat_x, X) + D_x_x*U1_hat_x
        dot_F_12 = D_x * self.fwd_gradients_U_x(U1_hat_x, Y) + D_x_y*U1_hat_x
        dot_F_21 = D_y * self.fwd_gradients_U_x(U1_hat_y, X) + D_y_x*U1_hat_y
        dot_F_22 = D_y * self.fwd_gradients_U_x(U1_hat_y, Y) + D_y_y*U1_hat_y

        '''Delta F'''
        F_11 = self.dt * torch.matmul(dot_F_11[:, 0:self.q], b_j) + 1
        F_12 = self.dt * torch.matmul(dot_F_12[:, 0:self.q], b_j)
        F_21 = self.dt * torch.matmul(dot_F_21[:, 0:self.q], b_j)
        F_22 = self.dt * torch.matmul(dot_F_22[:, 0:self.q], b_j) + 1

        '''Inverse of deformation gradient'''
        detF     = torch.multiply(F_11, F_22) - torch.multiply(F_12, F_21)
        F_inv_11 =  torch.divide(F_22, detF)
        F_inv_12 = -torch.divide(F_12, detF)
        F_inv_21 = -torch.divide(F_21, detF)
        F_inv_22 =  torch.divide(F_11, detF)

        F_inv = [F_inv_11, F_inv_12, F_inv_21, F_inv_22]
        dot_F = [dot_F_11, dot_F_12, dot_F_21, dot_F_22]
        return F_inv, dot_F, detF
    
    def balance_equations(self, X, Y, RHO_0):
        '''Distance function and boundary extension'''
        D_x, D_x_x, D_x_y, D_y, D_y_x, D_y_y = self._distance_function(X, Y)

        '''Neural network'''
        U1 = self.model(torch.cat((X, Y), 1))

        '''RK velocity stages & solution'''
        U1_hat_x, U1_hat_y, U1_x, U1_y, p_stages = self._fetch_output(U1, D_x, D_y)

        '''Grad v, F_inv, detF'''
        F_inv, dot_F, detF = self._compute_F_inv(D_x, D_x_x, D_x_y, D_y, D_y_x, D_y_y, U1_hat_x, U1_hat_y, X, Y)
        F_inv_11 = F_inv[0]
        F_inv_12 = F_inv[1]
        F_inv_21 = F_inv[2]
        F_inv_22 = F_inv[3]
        dot_F_11 = dot_F[0]
        dot_F_12 = dot_F[1]
        dot_F_21 = dot_F[2]
        dot_F_22 = dot_F[3]

        '''!!! Momentum equation: only stages go into acceleration computation!!'''
        '''Density inverse'''
        invRho = torch.divide(1.0, RHO_0)

        '''Grad p'''
        Grad_p_X = self.fwd_gradients_U_x(-p_stages, X)
        Grad_p_Y = self.fwd_gradients_U_x(-p_stages, Y)

        '''Contact acceleration'''
        grad_p_x = torch.multiply(F_inv_11[:, 0:self.q], Grad_p_X) + torch.multiply(F_inv_21[:, 0:self.q], Grad_p_Y)
        grad_p_y = torch.multiply(F_inv_12[:, 0:self.q], Grad_p_X) + torch.multiply(F_inv_22[:, 0:self.q], Grad_p_Y)

        '''Contact acceleration'''
        x_stages = X + self.dt*torch.matmul(U1_x[:, 0:self.q], self.IRK_weights_t[0:self.q, :].T) # [799, 20]
        y_stages = Y + self.dt*torch.matmul(U1_y[:, 0:self.q], self.IRK_weights_t[0:self.q, :].T) # [799, 20]
        eps = 1e7

        # left boundary
        normal_left  = -1.0
        gap_left     = (x_stages - 0.0)*normal_left
        active_left  = 0.5*(1.0 + torch.sign(gap_left))
        penalty_left = eps*gap_left*active_left*normal_left

        # bottom boundary
        normal_bottom  = -1.0
        gap_bottom     = (y_stages - 0.0)*normal_bottom
        active_bottom  = 0.5*(1.0 + torch.sign(gap_bottom))
        penalty_bottom = eps*gap_bottom*active_bottom*normal_bottom

        '''Momentum equation in x and y'''
        acce_x = self.b_x + torch.multiply(invRho, grad_p_x) - penalty_left   #- penalty_right
        acce_y = self.b_y + torch.multiply(invRho, grad_p_y) - penalty_bottom

        '''Time integration''' # v_n_i = v_n+1 - dt*sum_i a_ji f(xi_i) \\ v_n_q+1 = v_n+1 - dt*sum_j b_j f(xi_j)
        U0_x = U1_x - self.dt*torch.matmul(acce_x, self.IRK_weights_t.T) #[799, 21]
        U0_y = U1_y - self.dt*torch.matmul(acce_y, self.IRK_weights_t.T) #[799, 21]

        '''!!! Mass balance'''
        '''Spatial velocity gradient l'''
        l_11 = torch.multiply(dot_F_11, F_inv_11) + torch.multiply(dot_F_12, F_inv_21)
        l_22 = torch.multiply(dot_F_21, F_inv_12) + torch.multiply(dot_F_22, F_inv_22)
        
        divU = l_11 + l_22
        return U0_x, U0_y, divU
    
    def output_solution(self, X, Y):
        D_x, D_x_x, D_x_y, D_y, D_y_x, D_y_y = self._distance_function(X, Y)

        '''Neural network'''
        U1 = self.model(torch.cat([X ,Y], 1))

        U1_hat_x, U1_hat_y, U1_x, U1_y, p_stages = self._fetch_output_for_output_solution(U1, D_x, D_y)

        '''Updated Position and pressure'''
        b_j  = self.IRK_weights_t[self.q:(self.q+1),:].T
        x    = X + self.dt*torch.matmul(U1_x[:, 0:self.q], b_j)
        y    = Y + self.dt*torch.matmul(U1_y[:, 0:self.q], b_j)
        p_na = torch.matmul(p_stages, b_j)

        '''Grad v, F_inv, detF'''
        F_inv, dot_F, detF = self._compute_F_inv_for_output_solution(D_x, D_x_x, D_x_y, D_y, D_y_x, D_y_y, U1_hat_x, U1_hat_y, X, Y)
        F_inv_11 = F_inv[0]
        F_inv_12 = F_inv[1]
        F_inv_21 = F_inv[2]
        F_inv_22 = F_inv[3]
        dot_F_11 = dot_F[0]
        dot_F_12 = dot_F[1]
        dot_F_21 = dot_F[2]
        dot_F_22 = dot_F[3]

        '''Spatial velocity gradient l'''
        l_11 = torch.multiply(dot_F_11[:, self.q:(self.q+1)], F_inv_11) + torch.multiply(dot_F_12[:, self.q:(self.q+1)], F_inv_21)
        l_22 = torch.multiply(dot_F_21[:, self.q:(self.q+1)], F_inv_12) + torch.multiply(dot_F_22[:, self.q:(self.q+1)], F_inv_22)

        '''div v = tr l'''
        divU = l_11 + l_22
        return U1_x, U1_y, divU, detF, x, y, p_na # N x (2*(q+1) +1)
    
    def net_U1_p(self, x1_p, y1_p):
        U1 = self.model(torch.cat([x1_p, y1_p], 1))
        U1_p = U1[:, 2*(self.q+1):3*(self.q+1)] # [59, 4]
        return U1_p
    
    def compute_Loss(self, criteria):
        loss_p    = criteria(self.U1_p_pred, torch.zeros_like(self.U1_p_pred))
        loss_mass = criteria(self.divU_pred, torch.zeros_like(self.divU_pred))
        loss_x    = criteria(self.U0_x_pred, self.u0_x_t.expand(-1, self.q+1)) * (self.q + 1)
        loss_y    = criteria(self.U0_y_pred, self.u0_y_t.expand(-1, self.q+1)) * (self.q + 1)
        return loss_p + loss_mass + loss_x + loss_y, loss_p, loss_mass, loss_x, loss_y
        
    def _closure(self):
        self.x1_p_t = self.x1_p_t.contiguous().clone().detach().requires_grad_(True).to(self.device)
        self.y1_p_t = self.y1_p_t.contiguous().clone().detach().requires_grad_(True).to(self.device)

        self.U0_x_pred, self.U0_y_pred, self.divU_pred = self.balance_equations(self.x0_t, self.y0_t, self.rho0_t)
        self.U1_p_pred = self.net_U1_p(self.x1_p_t, self.y1_p_t)
        loss, loss_p, loss_mass, loss_vx, loss_vy  = self.compute_Loss(self.criteria)
        self.optimizer_LBFGS.zero_grad()
        loss.backward()

        self.iter += 1
        if self.iter % 1000 == 0:
            self.logger.append('LBFGS : Iter %d, Loss %.3e, Loss_p %.3e, Loss_rho %.3e, Loss_vx %.3e, Loss_vy %.3e' % (self.iter, loss.item(), loss_p.item(), loss_mass.item(), loss_vx.item(), loss_vy.item()))
        return loss
    
    def train_LBFGS(self):
        self.model.load_state_dict(self.best_parameters)
        loss_before, loss_p, loss_mass, loss_vx, loss_vy = self.compute_Loss(self.criteria)
        self.logger.append('LBFGS before : Loss %.3e, Loss_p %.3e, Loss_rho %.3e, Loss_vx %.3e, Loss_vy %.3e' % (loss_before.item(), loss_p.item(), loss_mass.item(), loss_vx.item(), loss_vy.item()))

        self.optimizer_LBFGS.step(self._closure)
        
        loss_after, loss_p, loss_mass, loss_vx, loss_vy = self.compute_Loss(self.criteria)
        self.logger.append('LBFGS after : Loss %.3e, Loss_p %.3e, Loss_rho %.3e, Loss_vx %.3e, Loss_vy %.3e' % (loss_after.item(), loss_p.item(), loss_mass.item(), loss_vx.item(), loss_vy.item()))

        if loss_after.item() < loss_before.item():
            self.best_parameters = copy.deepcopy(self.model.state_dict())
            self.logger.append('LBFGS : Update best parameters')
        else:
            self.model.load_state_dict(self.best_parameters)
            self.logger.append('LBFGS : Not update best parameters')
        return loss_after.item()

    def train_adam(self, epochs):
        self.model.train()
        losses_adam = []

        minimum_loss = 1e10
        flg_best = False

        for epoch in range(epochs + 1):
            self.x1_p_t = self.x1_p_t.contiguous().clone().detach().requires_grad_(True).to(self.device)
            self.y1_p_t = self.y1_p_t.contiguous().clone().detach().requires_grad_(True).to(self.device)

            self.optimizer.zero_grad()
            self.U0_x_pred, self.U0_y_pred, self.divU_pred = self.balance_equations(self.x0_t, self.y0_t, self.rho0_t)
            self.U1_p_pred = self.net_U1_p(self.x1_p_t, self.y1_p_t)
            loss, loss_p, loss_mass, loss_vx, loss_vy = self.compute_Loss(self.criteria)

            if epoch % 1000 == 0:
                self.logger.append('Adam : Epoch %d, Loss %.3e, Loss_p %.3e, Loss_rho %.3e, Loss_vx %.3e, Loss_vy %.3e' % (epoch, loss.item(), loss_p.item(), loss_mass.item(), loss_vx.item(), loss_vy.item()))
                if flg_best:
                    self.logger.append('Adam : Update best parameters')
                    flg_best = False

            # save the best model
            if epoch > 2000 and loss.item() < minimum_loss:
                minimum_loss = loss.item()
                self.best_parameters = copy.deepcopy(self.model.state_dict())
                flg_best = True

            if self.args.early_stopping_flg:
                self.early_stopping(loss, self.model)
                if self.early_stopping.early_stop:
                    self.logger.append("Early stopping")
                    break
            losses_adam.append(loss.item())

            loss.backward() 
            self.optimizer.step()

        return losses_adam
    
    def train(self, epochs):
        losses_adam = []
        losses_adam = self.train_adam(epochs)
        if self.LBFGS_FLG:
            self.train_LBFGS()
        return losses_adam

    def predict(self, x_star, y_star):
        '''
        self.U1_x_out : 次のステップのx速度 
        self.U1_y_out : 次のステップのy速度
        self.divU_out : div v
        self.detF_out : det F
        self.x_out    : 次のステップのx座標
        self.y_out    : 次のステップのy座標
        self.p_out    : 次のステップの圧力
        '''
        x_star = torch.tensor(x_star, dtype=torch.float32, device=self.device, requires_grad=True)
        y_star = torch.tensor(y_star, dtype=torch.float32, device=self.device, requires_grad=True)
        # 最も良いパラメータをロード
        self.model.load_state_dict(self.best_parameters)
        self.model.eval()
        return self.output_solution(x_star, y_star)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

class complessibleNeuralParticleMethods(NeuralParticleMethods):
    def __init__(self, id_same, id_u1_p, x0, y0, rho0_hat, u0, layers, dt, lb, ub, q, mu, t_step, device, THIS_DIR_PATH, LBFGS_FLG, args, logger, eta, height):
        super().__init__(id_same, id_u1_p, x0, y0, rho0_hat, u0, layers, dt, lb, ub, q, mu, t_step, device, THIS_DIR_PATH, LBFGS_FLG, args, logger)

        self.optimizer_LBFGS = torch.optim.LBFGS(self.model.parameters(), 
                                        max_iter = 1000000,
                                        max_eval = 50000,
                                        history_size = 50,
                                        tolerance_change = 10 * np.finfo(float).eps, # gNPMの設定:10
                                        line_search_fn = "strong_wolfe")
        
        # material data
        self.eta = eta

        # domain height
        self.height = height
        self.logger.append('This is complessibleNeuralParticleMethods')

    def _fetch_output(self, U1, D_x, D_y):
        U1_hat_x = U1[:, 0:self.q+1]
        U1_hat_y = U1[:, (self.q+1):(2*self.q+2)]
        U1_x     = D_x*U1[:, 0:(self.q+1)] 
        U1_y     = D_y*U1[:, (self.q+1):(2*self.q+2)]
        p_stages = U1[:, (2*self.q+2):(3*self.q+3)]  
        return U1_hat_x, U1_hat_y, U1_x, U1_y, p_stages
    
    def _equation_of_state_scaled(self, p_hat):
        g     = 9.81 #速度計算に使うので正の値
        gamma = 7.0
        vf    = np.sqrt(2 * g * self.height)
        cs    = vf/np.sqrt(self.eta)
        return torch.pow(gamma * p_hat/ (cs**2) + 1, 1 / gamma)
    
    def _compute_Lapracian_v(self, F_inv, dot_F, X, Y):
        F_inv_11 = F_inv[0]
        F_inv_12 = F_inv[1]
        F_inv_21 = F_inv[2]
        F_inv_22 = F_inv[3]
        dot_F_11 = dot_F[0]
        dot_F_12 = dot_F[1]
        dot_F_21 = dot_F[2]
        dot_F_22 = dot_F[3]

        l_11 = torch.multiply(dot_F_11, F_inv_11) + torch.multiply(dot_F_12, F_inv_21)
        l_12 = torch.multiply(dot_F_11, F_inv_12) + torch.multiply(dot_F_12, F_inv_22)
        l_21 = torch.multiply(dot_F_21, F_inv_11) + torch.multiply(dot_F_22, F_inv_21)
        l_22 = torch.multiply(dot_F_21, F_inv_12) + torch.multiply(dot_F_22, F_inv_22)

        dl_11dx = self.fwd_gradients_U_x(l_11, X)
        dl_12dx = self.fwd_gradients_U_x(l_12, X)
        dl_21dx = self.fwd_gradients_U_x(l_21, X)
        dl_22dx = self.fwd_gradients_U_x(l_22, X)
        dl_11dy = self.fwd_gradients_U_x(l_11, Y)
        dl_12dy = self.fwd_gradients_U_x(l_12, Y)
        dl_21dy = self.fwd_gradients_U_x(l_21, Y)
        dl_22dy = self.fwd_gradients_U_x(l_22, Y)

        dvdxx   = torch.multiply(dl_11dx, F_inv_11) \
                + torch.multiply(dl_11dy, F_inv_21) \
                + torch.multiply(dl_12dx, F_inv_12) \
                + torch.multiply(dl_12dy, F_inv_22) 
        dvdyy   = torch.multiply(dl_21dx, F_inv_11) \
                + torch.multiply(dl_21dy, F_inv_21) \
                + torch.multiply(dl_22dx, F_inv_12) \
                + torch.multiply(dl_22dy, F_inv_22)

        """
        F_inv_11 shape: torch.Size([799, 5])
        F_inv_12 shape: torch.Size([799, 5])
        F_inv_21 shape: torch.Size([799, 5])
        F_inv_22 shape: torch.Size([799, 5])
        dot_F_11 shape: torch.Size([799, 5])
        dot_F_12 shape: torch.Size([799, 5])
        dot_F_21 shape: torch.Size([799, 5])
        dot_F_22 shape: torch.Size([799, 5])
        dl_11dx  shape: torch.Size([799, 5])
        dl_12dx  shape: torch.Size([799, 5])
        dl_21dx  shape: torch.Size([799, 5])
        dl_22dx  shape: torch.Size([799, 5])
        dl_11dy  shape: torch.Size([799, 5])
        dl_12dy  shape: torch.Size([799, 5])
        dl_21dy  shape: torch.Size([799, 5])
        dl_22dy  shape: torch.Size([799, 5])
        dvdxx    shape: torch.Size([799, 5])
        dvdyy    shape: torch.Size([799, 5])
        """

        return dvdxx, dvdyy

    def balance_equations(self, X, Y):
        ''''''
        '''Distance function and boundary extension'''
        D_x, D_x_x, D_x_y, D_y, D_y_x, D_y_y = self._distance_function(X, Y , c_rank = 1)

        '''Neural network'''
        U1 = self.model(torch.cat((X, Y), 1))

        '''RK velocity stages & solution'''
        U1_hat_x, U1_hat_y, U1_x, U1_y, p1_hat = self._fetch_output(U1, D_x, D_y)

        '''Computing F'''
        F_inv, dot_F, _ = self._compute_F_inv(D_x, D_x_x, D_x_y, 
                                                D_y, D_y_x, D_y_y, 
                                                U1_hat_x, U1_hat_y, 
                                                X, Y)
        F_inv_11 = F_inv[0]
        F_inv_12 = F_inv[1]
        F_inv_21 = F_inv[2]
        F_inv_22 = F_inv[3]
        dot_F_11 = dot_F[0]
        dot_F_12 = dot_F[1]
        dot_F_21 = dot_F[2]
        dot_F_22 = dot_F[3]

        '''mass balance'''
        l_11 = torch.multiply(dot_F_11, F_inv_11) + torch.multiply(dot_F_12, F_inv_21)
        l_22 = torch.multiply(dot_F_21, F_inv_12) + torch.multiply(dot_F_22, F_inv_22)
        divU = l_11 + l_22

        '''!!! Momentum equation: only stages go into acceleration computation!!'''
        '''Density inverse'''
        rho1_hat    = self._equation_of_state_scaled(p1_hat)
        invrho1_hat = torch.reciprocal(rho1_hat[:, 0:self.q]) # [799, 4]

        '''Grad p'''
        Grad_p_X = self.fwd_gradients_U_x(-p1_hat[:, 0:self.q], X)
        Grad_p_Y = self.fwd_gradients_U_x(-p1_hat[:, 0:self.q], Y)
        grad_p_x = torch.multiply(F_inv_11[:, 0:self.q], Grad_p_X) + torch.multiply(F_inv_21[:, 0:self.q], Grad_p_Y)
        grad_p_y = torch.multiply(F_inv_12[:, 0:self.q], Grad_p_X) + torch.multiply(F_inv_22[:, 0:self.q], Grad_p_Y)

        '''Contact acceleration'''
        x_stages = X + self.dt*torch.matmul(U1_x[:, 0:self.q], self.IRK_weights_t[0:self.q, :].T) 
        y_stages = Y + self.dt*torch.matmul(U1_y[:, 0:self.q], self.IRK_weights_t[0:self.q, :].T) 
        
        eps          = 1e7
        # left boundary
        normal_left  = -1.0
        gap_left     = (x_stages - 0.0)*normal_left
        active_left  = 0.5*(1.0 + torch.sign(gap_left))
        penalty_left = eps*gap_left*active_left*normal_left

        # bottom boundary
        normal_bottom  = -1.0
        gap_bottom     = (y_stages - 0.0)*normal_bottom
        active_bottom  = 0.5*(1.0 + torch.sign(gap_bottom))
        penalty_bottom = eps*gap_bottom*active_bottom*normal_bottom

        '''Momentum equation in x and y'''
        '''Laplacian of velocity'''
        dvdxx, dvdyy = self._compute_Lapracian_v(F_inv, dot_F, X, Y)

        '''Momentum equation in x and y'''
        vis_x        = self.mu / self.args.rho_init * torch.multiply(invrho1_hat, dvdxx[:, 0:self.q]) # [799, 4]
        vis_y        = self.mu / self.args.rho_init * torch.multiply(invrho1_hat, dvdyy[:, 0:self.q]) # [799, 4]

        acce_x = self.b_x + torch.multiply(invrho1_hat, grad_p_x) + vis_x - penalty_left    # [799, 4]
        acce_y = self.b_y + torch.multiply(invrho1_hat, grad_p_y) + vis_y - penalty_bottom  # [799, 4]

        '''Time integration'''
        U0_x = U1_x - self.dt*torch.matmul(acce_x, self.IRK_weights_t.T) # [799, 5]
        U0_y = U1_y - self.dt*torch.matmul(acce_y, self.IRK_weights_t.T) # [799, 5]

        drho1_dt_hat = -1 * torch.multiply(rho1_hat[:, 0:self.q], divU[:, 0:self.q])  
        rho0_hat     = rho1_hat - self.dt*torch.matmul(drho1_dt_hat, self.IRK_weights_t.T) # [799, 5]

        # Shape information
        """
        U1_hat_x shape : torch.Size([799, 5])
        U1_hat_y shape : torch.Size([799, 5])
        D_x shape      : torch.Size([799, 1])
        D_y shape      : torch.Size([799, 1])
        U1_x shape     : torch.Size([799, 5])
        U1_y shape     : torch.Size([799, 5])
        p1_hat shape   : torch.Size([799, 5])
        F_inv_11 shape : torch.Size([799, 5])
        F_inv_12 shape : torch.Size([799, 5])
        F_inv_21 shape : torch.Size([799, 5])
        F_inv_22 shape : torch.Size([799, 5])
        dot_F_11 shape : torch.Size([799, 5])
        dot_F_12 shape : torch.Size([799, 5])
        dot_F_21 shape : torch.Size([799, 5])
        dot_F_22 shape : torch.Size([799, 5])
        div_U shape    : torch.Size([799, 5])
        rho1_hat shape : torch.Size([799, 5])
        invrho1_hat shape : torch.Size([799, 4])
        grad_p_x shape : torch.Size([799, 4])
        grad_p_y shape : torch.Size([799, 4])
        x_stages shape : torch.Size([799, 4])
        y_stages shape : torch.Size([799, 4])
        dvdxx shape    : torch.Size([799, 5])
        dvdyy shape    : torch.Size([799, 5])
        vis_x shape    : torch.Size([799, 4])
        vis_y shape    : torch.Size([799, 4])
        acce_x shape   : torch.Size([799, 4])
        acce_y shape   : torch.Size([799, 4])
        U0_x shape     : torch.Size([799, 5])
        U0_y shape     : torch.Size([799, 5])
        IRK_weights_t shape : torch.Size([5, 4])
        drho1_dt_hat shape  : torch.Size([799, 4])
        rho0_hat shape      : torch.Size([799, 5])
        """
        return U0_x, U0_y, rho0_hat

    def output_solution(self, X, Y):
        '''Distance function and boundary extension'''
        D_x, _, _, D_y, _, _,  = self._distance_function(X, Y)

        '''Neural network'''
        U1 = self.model(torch.cat([X ,Y], 1))

        '''RK velocity stages & solution'''
        _, _, U1_x, U1_y, p_out_hat = self._fetch_output(U1, D_x, D_y)


        '''!!! Momentum equation: only stages go into acceleration computation!!'''
        '''Density inverse'''
        rho_out_hat = self._equation_of_state_scaled(p_out_hat)

        '''Updated Position, pressure and density'''
        b_j  = self.IRK_weights_t[self.q:(self.q+1),:].T
        x    = X + self.dt*torch.matmul(U1_x[:, 0:self.q], b_j)
        y    = Y + self.dt*torch.matmul(U1_y[:, 0:self.q], b_j)
        p    = torch.matmul(p_out_hat[:, 0:self.q], b_j)
        rho  = torch.matmul(rho_out_hat[:, 0:self.q], b_j)

        """
        D_x shape      : torch.Size([6843, 1])
        D_y shape      : torch.Size([6843, 1])
        U1_x shape     : torch.Size([6843, 5])
        U1_y shape     : torch.Size([6843, 5])
        p_out_hat shape: torch.Size([6843, 5])
        rho_out_hat shape: torch.Size([6843, 5])
        x shape        : torch.Size([6843, 1])
        y shape        : torch.Size([6843, 1])
        p shape        : torch.Size([6843, 1])
        rho shape      : torch.Size([6843, 1])
        """
        return U1_x, U1_y, None, None, x, y, p, rho

    def net_U1_p(self, x1_p, y1_p):
        U1 = self.model(torch.cat([x1_p, y1_p], 1))
        p_hat = U1[:, (2*self.q+2):(3*self.q+3)]      # RK pressure stages
        #p_hat shape: torch.Size([59, 5])
        return p_hat

    def compute_Loss(self, criteria):
        loss_p   = criteria(self.U1_p_hat_pred, torch.zeros_like(self.U1_p_hat_pred))
        loss_rho = criteria(self.rho0_hat_pred, self.rho0_t.expand(-1, self.q+1)) * (self.q + 1)
        loss_vx  = criteria(self.U0_x_pred,     self.u0_x_t.expand(-1, self.q+1)) * (self.q + 1)
        loss_vy  = criteria(self.U0_y_pred,     self.u0_y_t.expand(-1, self.q+1)) * (self.q + 1)
        """
        U1_p_hat_pred shape : torch.Size([59,  5])
        rho0_hat_pred shape : torch.Size([799, 5])
        rho0_t shape        : torch.Size([799, 1])
        U0_x_pred shape     : torch.Size([799, 5])
        u0_x_t shape        : torch.Size([799, 1])
        U0_y_pred shape     : torch.Size([799, 5])
        u0_y_t shape        : torch.Size([799, 1])
        """
        return loss_p + loss_rho + loss_vx + loss_vy, loss_p, loss_rho, loss_vx, loss_vy
    
    def train_adam(self, epochs):
        self.model.train()
        losses_adam = []

        minimum_loss = 1e10
        flg_best = False

        for epoch in range(epochs + 1):
            
            self.x0_t   = self.x0_t.contiguous().clone().detach().requires_grad_(True).to(self.device)
            self.y0_t   = self.y0_t.contiguous().clone().detach().requires_grad_(True).to(self.device)
            self.rho0_t = self.rho0_t.contiguous().clone().detach().requires_grad_(True).to(self.device)
            self.ub     = self.ub.contiguous().clone().detach().requires_grad_(True).to(self.device)
            self.lb     = self.lb.contiguous().clone().detach().requires_grad_(True).to(self.device)
            self.u0_x_t = self.u0_x_t.contiguous().clone().detach().requires_grad_(True).to(self.device)
            self.u0_y_t = self.u0_y_t.contiguous().clone().detach().requires_grad_(True).to(self.device)
            self.x1_p_t = self.x1_p_t.contiguous().clone().detach().requires_grad_(True).to(self.device)
            self.y1_p_t = self.y1_p_t.contiguous().clone().detach().requires_grad_(True).to(self.device)
            self.IRK_weights_t = self.IRK_weights_t.contiguous().clone().detach().requires_grad_(True).to(self.device)
            

            self.optimizer.zero_grad()
            self.U0_x_pred, self.U0_y_pred, self.rho0_hat_pred = self.balance_equations(self.x0_t, self.y0_t)
            self.U1_p_hat_pred = self.net_U1_p(self.x1_p_t, self.y1_p_t)
            loss, loss_p, loss_rho, loss_vx, loss_vy = self.compute_Loss(self.criteria)
            if epoch % 1000 == 0:
                self.logger.append('Adam : Epoch %d, Loss %.3e, Loss_p %.3e, Loss_rho %.3e, Loss_vx %.3e, Loss_vy %.3e' % (epoch, loss.item(), loss_p.item(), loss_rho.item(), loss_vx.item(), loss_vy.item()))
                if flg_best:
                        self.logger.append('Adam : Update best parameters')
                        flg_best = False

            # save the best model
            if epoch > 2000 and loss.item() < minimum_loss:
                minimum_loss = loss.item()
                self.best_parameters = copy.deepcopy(self.model.state_dict())
                flg_best = True

            if self.args.early_stopping_flg:
                self.early_stopping(loss, self.model)
                if self.early_stopping.early_stop:
                    self.logger.append("Early stopping")
                    break
            losses_adam.append(loss.item())

            loss.backward() 
            self.optimizer.step()

        return losses_adam

    def predict(self, x_star, y_star):
        '''
        self.U1_x_out : 次のステップのx速度 
        self.U1_y_out : 次のステップのy速度
        self.divU_out : div v
        self.detF_out : det F
        self.x_out    : 次のステップのx座標
        self.y_out    : 次のステップのy座標
        self.p_out    : 次のステップの圧力
        '''
        x_star = torch.tensor(x_star, dtype=torch.float32, device=self.device, requires_grad=True)
        y_star = torch.tensor(y_star, dtype=torch.float32, device=self.device, requires_grad=True)
        self.model.load_state_dict(self.best_parameters)
        self.model.eval()
        return self.output_solution(x_star, y_star)

