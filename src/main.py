"""
Author: masato shibukawa
Date: 2025-07-22
Email: shibukawa.masato@ac.jaxa.jp

Main execution script for Neural Particle Methods (NPM) and Compressible Neural Particle Methods (cNPM).
This script handles command-line arguments, initializes simulations, and orchestrates the time-stepping loop
for fluid dynamics simulations using Physics-Informed Neural Networks.
"""

import os
import torch
import scipy.io 
import numpy as np
import tool.init as init
import time
import argparse
#from simulator import Simulator
from simulator import Simulator
from tool.plot import plot
from tool.logger import Logger

# このフォルダの絶対パス
THIS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# 今フォルダの名前
THIS_DIR_NAME = os.path.basename(THIS_DIR_PATH)

# GPUの確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
print('Using device:', device)

# pytorchのデバッグAPIを無効化する
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

# cuDNNの自動チューニングを有効化:再現性がなくなるため無効化
#torch.backends.cudnn.benchmark = True

def arg_parser():
    # argparseで設定を受け取る
    parser = argparse.ArgumentParser(description='Run Neural Particle Methods simulation.')
    # モデルの設定
    parser.add_argument('--model', type=str, default=r'NPM', help='Model name')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[60, 60], help='List of hidden layer sizes')

    # ダムの形状
    parser.add_argument('--L', type=float, default=0.146, help='Length of the domain')
    parser.add_argument('--H', type=float, default=0.292, help='Height of the domain')

    # パーティクルの設置
    parser.add_argument('--dl', type=float, default=0.007, help='Distance between particles')
    parser.add_argument('--particle_distribution', type=str, choices=['grid', 'random'], default='grid', help='学習用の粒子の配置がグリッド状か、ランダムに配置するかを設定')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for particle distribution')
    parser.add_argument('--refine_times', type=int, default=10, help='Number of times to refine the particle distribution')

    # 流体の物性値
    parser.add_argument('--rho_init', type=float, default=997, help='Initial density')
    parser.add_argument('--mu',  type=float, default=0.001016, help='Dynamic viscosity')
    parser.add_argument('--eta', type=float, default=0.01, help='Tait parameter')

    # Runge-Kuttaの次数
    parser.add_argument('--q', type=int, default=4, help='Parameter Q')

    # シミュレーションの時間設定
    parser.add_argument('--t_start', type=float, default=0, help='Start time')
    parser.add_argument('--dt', type=float, default=0.01, help='Time increment')
    parser.add_argument('--t_max', type=int, default=20, help='Number of time steps')

    # 最適化の設定
    parser.add_argument('--epoch', type=int, default=20000, help='Number of epochs for Adam optimizer')
    parser.add_argument('--lbfgs_flg', type=int, default=0, help='Flag to use LBFGS optimizer')
    parser.add_argument('--tl_flg', type=int, default=0, help='Flag for transfer learning')
    parser.add_argument('--early_stopping_flg', type=int, default=0, help='Flag for early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=1000, help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for early stopping')

    # その他の設定
    parser.add_argument('--database_path', type=str, default=r"./database", help='Path to the database')
    parser.add_argument('--result_directory_name', type=str, default=r"result", help='Result directory name')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()

    # 時間増分、時間ステップ数、開始時間を設定
    dt         = args.dt
    t_nn_steps = args.t_max
    t_start    = args.t_start
    idt = np.arange(t_start+1, t_start+t_nn_steps+2, 1, dtype=np.float32)

    # 結果を保存するディレクトリを作成
    dir_name_result  = args.result_directory_name
    dir_path_result  = os.path.join(args.database_path, THIS_DIR_NAME, dir_name_result)
    dir_path_ic_bc   = os.path.join(dir_path_result, 'ic_bc')
    dir_path_figure  = os.path.join(dir_path_result, 'figure')
    dir_path_weights = os.path.join(dir_path_result, 'weights')
    dir_path_loss    = os.path.join(dir_path_result, 'loss')
    os.makedirs(dir_path_result, exist_ok=True)
    os.makedirs(dir_path_ic_bc, exist_ok=True)
    os.makedirs(dir_path_figure, exist_ok=True)
    os.makedirs(dir_path_weights, exist_ok=True)
    os.makedirs(dir_path_loss, exist_ok=True)

    logger = Logger(os.path.join(dir_path_result, 'train_log.log'), True)


    # 初期化
    id_same, id_free_x, id_u1_x, id_free_y, id_u1_y, id_free_p, id_u1_p, x, y, Exact = init.dam_break_initialize(
        dir_path_ic_bc, dir_path_figure, args
    )
    u0 = Exact.T # (60543, 3)

    # Lower and upper bounds
    lb = np.array([np.min(x), np.min(y)])
    ub = np.array([np.max(x), np.max(y)])

    # rhoを初期化、997で初期化
    if 'cNPM' in args.model:
        rho0 = np.ones((len(x), 1)).flatten()[:, None] # scaling
    else:
        rho0 = np.ones((len(x), 1)).flatten()[:, None] * args.rho_init

    '''Save initial data to file'''
    IC = np.concatenate((x, y, u0, rho0), axis=1) #(60543, 7)

    # 初期データを保存
    os.makedirs(dir_path_ic_bc, exist_ok=True)
    scipy.io.savemat(dir_path_ic_bc + '/timeStep.mat', {'dt': dt, 'idt': idt})
    scipy.io.savemat(os.path.join(dir_path_ic_bc, 'IC_t0.mat'), 
                    {'IC': IC})
    scipy.io.savemat(os.path.join(dir_path_ic_bc, 'BC_t0.mat'), 
                    {'id_same': id_same, 
                    'id_u1_x': id_u1_x, 
                    'id_u1_y': id_u1_y, 
                    'id_u1_p': id_u1_p, 
                    'id_free_x': id_free_x, 
                    'id_free_y': id_free_y, 
                    'id_free_p': id_free_p})

    '''plot initial data'''
    if(t_start==0):
        U_x_plot = np.zeros((len(y), 1))
        U_y_plot = np.zeros((len(y), 1))
        P_hat    = np.ones((len(y), 1))

        U_mag_plot = np.sqrt(np.square(U_x_plot) + np.square(U_y_plot))

        # plot
        plot(x[id_same], y[id_same], P_hat[id_same], 'pressure_t', dir_path_figure, 0, lb, ub, args)
        plot(x[id_same], y[id_same], U_x_plot[id_same], 'x-velocity_t',dir_path_figure, 0, lb, ub, args)
        plot(x[id_same], y[id_same], U_y_plot[id_same], 'y-velocity_t',dir_path_figure, 0, lb, ub, args)
        plot(x[id_same], y[id_same], U_mag_plot[id_same], 'velocity_magnitude_t',dir_path_figure, 0, lb, ub, args)
        plot(x[id_same], y[id_same], P_hat[id_same], 'divergence_t',dir_path_figure, 0, lb, ub, args)
        plot(x[id_same], y[id_same], rho0[id_same], 'density_t', dir_path_figure, 0, lb, ub, args)

    dir_path_IRK = os.path.join(THIS_DIR_PATH, 'IRK')
    dir_table = {
        "figure": dir_path_figure, 
        "weights": dir_path_weights, 
        "loss": dir_path_loss, 
        "ic_bc": dir_path_ic_bc, 
        "result": dir_path_result,
        "IRK" : dir_path_IRK,
    }
    simulator = Simulator(dir_table, device, args, logger)

    start = time.time()
    for i in range(1, t_nn_steps+1):
        train_time = simulator.runDamBreak()
        torch.cuda.empty_cache()
        with open(os.path.join(dir_path_result, 'train_time_log.csv'), mode='a') as f:
            f.write(f'{i},{train_time}\n')
    end   = time.time()
    elapsed_time = end - start

    with open(os.path.join(dir_path_result, 'total_time.txt'), mode='w') as f:
        f.write(f'{elapsed_time:.3f} [sec]\n')
        f.write(f'{elapsed_time/60:.3f} [min]\n')
        f.write(f'{elapsed_time/3600:.3f} [hour]\n')

