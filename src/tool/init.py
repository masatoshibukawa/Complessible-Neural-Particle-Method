"""
Author: masato shibukawa
Date: 2025-07-22
Email: shibukawa.masato@ac.jaxa.jp

Initialization utilities for dam break simulation setup.
Handles particle generation, initial conditions, boundary conditions,
and configuration file management for neural particle methods.
"""

import os
import yaml
import shutil
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def copy_file(source_file, destination_folder):
    """
    指定のファイルを指定のフォルダにコピーする関数

    Args:
        source_file (str): コピー元のファイルパス
        destination_folder (str): コピー先のフォルダパス

    Returns:
        str: コピーされたファイルのパス
    """
    destination_folder = os.path.join(destination_folder, os.path.basename(source_file))
    # ファイルをコピーする
    copied_file = shutil.copy(source_file, destination_folder)
    
    return copied_file

def read_yml(file_path):
    """
    yamlファイルを読み込む関数
    input : file_path - str, yamlファイルのパス
    output: data - dict, yamlファイルのデータ
    """
    with open(file_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
    return data

def make_dirs_for_output(dir_path):
    """
    出力用のディレクトリを作成する関数
    input : dir_path - str, 出力用のディレクトリパス
    output: dir_path_posi - str, 位置の出力用ディレクトリパス
            dir_path_velo - str, 速度の出力用ディレクトリパス
            dir_path_pres - str, 圧力の出力用ディレクトリパス
            dir_path_IC_BC - str, 初期条件と境界条件の出力用ディレクトリパス

    """
    dir_path_posi = os.path.join(dir_path, "figure", "position")
    dir_path_velo = os.path.join(dir_path, "figure", "velocity")
    dir_path_pres = os.path.join(dir_path, "figure", "pressure")
    dir_path_IC_BC = os.path.join(dir_path, "IC_BC")
    os.makedirs(dir_path_posi, exist_ok=True)
    os.makedirs(dir_path_velo, exist_ok=True)
    os.makedirs(dir_path_pres, exist_ok=True)
    os.makedirs(dir_path_IC_BC, exist_ok=True)
    
    return dir_path_posi, dir_path_velo, dir_path_pres, dir_path_IC_BC

def plotInitialCondition(x, y, u0, idu1, BC_type, x_min, x_max, y_min, y_max, dir_path):

    # U_plot = np.zeros((len(y), 1))
    # for i in idu1:
    #     U_plot[i, 0] = 1

    f, ax = plt.subplots(1)

    plt.scatter(x[idu1], y[idu1], c=u0[idu1], cmap='hsv', s=1)
    plt.xlim(x_min, x_max*2)
    plt.ylim(y_min, y_max*1.1)
    cb = plt.colorbar()
    for l in cb.ax.yaxis.get_ticklabels():
        # l.set_weight("bold")
        l.set_fontsize(20)

    ax.tick_params(axis='both', labelsize=20)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(dir_path, BC_type + '.png'))
    plt.close()

def createList(x, y, x_min, x_max, y_min, y_max, dir_path):
    '''
    グリッドのエッジを削除する
    グリッドの原点(x, y) = (0, 0)を削除する
    '''
    id_u1_x = np.where(x == x_min)[0] # (40)
    id_u1_y = np.where(y == y_min)[0] # (20)

    id_common = np.intersect1d(id_u1_x, id_u1_y)[0]
    x = np.delete(x, id_common) # (799)
    y = np.delete(y, id_common) # (799)

    '''
    速度の境界条件に該当する点を取得する
    壁に接している点を取得する
    '''
    id_u1_x = np.where(x == x_min)[0] # (39)
    id_u1_y = np.where(y == y_min)[0] # (39)

    '''
    圧力の境界条件に該当する点を取得する
    自由表面に接している点を取得する
    '''
    id_u1_p_x = np.where(y == y_max)[0] # (20)
    id_u1_p_y = np.where(x == x_max)[0] # (40)

    # 重複したidを削除する
    common_value = np.intersect1d(id_u1_p_x, id_u1_p_y)
    id_common    = np.where(id_u1_p_x==common_value)[0]
    id_u1_p_x    = np.delete(id_u1_p_x, id_common)
    id_u1_p      = np.concatenate([id_u1_p_x, id_u1_p_y], axis=0) # (59)
    '''Flatten position vector'''
    x = x.flatten()[:, None] # (799, 1)
    y = y.flatten()[:, None] # (799, 1)

    '''
    自由粒子のインデックスを取得する
    自由粒子とは、境界条件に該当しない粒子のこと
    '''

    id_free_x = np.arange(0, len(x), 1) # (799)
    id_free_x = np.delete(id_free_x, id_u1_x) # (760)

    id_free_y = np.arange(0, len(x), 1)
    id_free_y = np.delete(id_free_y, id_u1_y) # (780)

    id_free_p = np.arange(0, len(x), 1)
    id_free_p = np.delete(id_free_p, id_u1_p) # (740)

    '''
    PINNsの出力の初期値
    xの速度、yの速度、圧力の初期値を0にする
    '''
    T = np.zeros((3, len(x))) # (3, 799)

    '''plot initial condition'''
    plotInitialCondition(x, y, T[0 ,:], id_u1_x, 'x-boun', x_min, x_max, y_min, y_max, dir_path)
    plotInitialCondition(x, y, T[1 ,:], id_u1_y, 'y-boun', x_min, x_max, y_min, y_max, dir_path)
    plotInitialCondition(x, y, T[2 ,:], id_u1_p, 'p-boun', x_min, x_max, y_min, y_max, dir_path)

    return x, y, id_u1_x, id_free_x, id_u1_y, id_free_y, id_u1_p, id_free_p, T

def refine(x_coarse, nx):

    x_fine = np.empty(0)
    for i in range(0, len(x_coarse) - 1):
        tmp = np.linspace(x_coarse[i], x_coarse[i + 1], nx)
        x_fine = np.append(x_fine, tmp)

    x_fine = np.unique(x_fine)

    return x_fine

def dam_break_initialize(dir_path_timeStep, dir_path_figure, args):
    """
    dam_breakの初期条件を設定する関数
    input : config_path - str, configファイルのパス
            dir_path - str, 出力用のディレクトリパス
    output: dir_path_posi - str, 位置の出力用ディレクトリパス
            dir_path_velo - str, 速度の出力用ディレクトリパス
            dir_path_pres - str, 圧力の出力用ディレクトリパス
            dir_path_IC_BC - str, 初期条件と境界条件の出力用ディレクトリパス
    """

    L  = args.L
    H  = args.H
    dl = args.dl

    t_nn_steps = args.t_max
    t_start    = args.t_start
    dt         = args.dt
    idt        = np.arange(t_start + 1, t_start + t_nn_steps + 2, 1, dtype=np.float32)

    # timestepファイルの作成
    scipy.io.savemat(dir_path_timeStep + '/timeStep.mat', {"dt": dt, "idt": idt})

    # 全粒子の配置と、分類分けを行う
    x_min, x_max = 0, L
    y_min, y_max = 0, H
    
    '''(x, y) simulation'''
    nx = int(L/dl) # 20
    ny = int(H/dl) # 40
    x_tmp = np.linspace(x_min, x_max, num = nx) # (20)
    y_tmp = np.linspace(y_min, y_max, num = ny) # (40)
    x_sim, y_sim = np.meshgrid(x_tmp, y_tmp) # (40, 20), (40, 20)

    x_sim = x_sim.flatten()[:, None].copy()
    y_sim = y_sim.flatten()[:, None].copy()

    '''(x, y) alphashape'''
    refine_num = args.refine_times
    x_tmp_2 = refine(x_tmp, refine_num) # (172) 20個の点を10個の点に分割して、重複した18個の点を削除
    y_tmp_2 = refine(y_tmp, refine_num) # (352)
    x_alpha, y_alpha = np.meshgrid(x_tmp_2, y_tmp_2) #(60543, 1), (60543, 1)

    # x_sim, y_simから原点を削除したx, yを返す
    x_sim, y_sim = createList(x_sim, y_sim, x_min, x_max, y_min, y_max, dir_path_figure)[0:2] # (799, 1), (799, 1)
    x_alpha, y_alpha, id_u1_x, id_free_x, id_u1_y, id_free_y, id_u1_p, id_free_p, T = createList(x_alpha, y_alpha, x_min, x_max, y_min, y_max, dir_path_figure)

    '''
    x_alpha,    (60543, 1) # 全粒子のx座標
    y_alpha,    (60543, 1) # 全粒子のy座標
    id_u1_x,    (351,)     # 速度の境界条件を満たす粒子のインデックス、x=0で壁に接している粒子のインデックス
    id_free_x,  (60192,)   # 自由粒子のインデックス、x=0で壁に接していない粒子のインデックス
    id_u1_y,    (171,)     # 速度の境界条件を満たす粒子のインデックス、y=0で壁に接している粒子のインデックス
    id_free_y,  (60372,)   # 自由粒子のインデックス、y=0で壁に接していない粒子のインデックス
    id_u1_p,    (523,)     # 圧力の境界条件を満たす粒子のインデックス、自由表面に接している粒子のインデックス
    id_free_p   (60020,)   # 自由粒子のインデックス、自由表面に接していない粒子のインデックス
    T,          (3, 60543)
    '''
    x_alpha = x_alpha.flatten()[:, None] # (60543, 1)
    y_alpha = y_alpha.flatten()[:, None] # (60543, 1)

    x_sim   = x_sim.flatten()[:, None] # (799, 1)
    y_sim   = y_sim.flatten()[:, None] # (799, 1)

    '''
    PINNsで学習する用のサブデータセットを作成する
    学習を行うのはx_simの粒子のみなので
    x_alphaの内、x_simに含まれる粒子のインデックスを取得する
    学習時は、x_alpha[id_sama]を使うことで、x_simの粒子を使った学習を実現できる
    '''
    common_x  = np.intersect1d(x_alpha, x_sim) # (20)
    id_same_x = np.unique(np.where(x_alpha == common_x)[0]) # (7039,)

    common_y  = np.intersect1d(y_alpha, y_sim) # (40)
    id_same_y = np.unique(np.where(y_alpha == common_y)[0]) # (6879,)

    if args.particle_distribution == "grid":
        #このx_sim == x_alpha[id_same]は全てtrueを返す
        id_same = np.intersect1d(id_same_x, id_same_y) # (799)
    elif args.particle_distribution == "random":
        #このx_sim == x_alpha[id_same]は全てtrueを返す
        id_same_uniform = np.intersect1d(id_same_x, id_same_y) # (799)

        id_same_fs = np.intersect1d(id_same_uniform, id_u1_p) # (523)

        # id_same_uniformからid_u1_pを排除
        id_same_filtered = np.setdiff1d(np.arange(len(x_alpha)), id_u1_p)

        # ランダムに指定数抽出
        num_random_samples = nx * ny - id_same_fs.shape[0]
        np.random.seed(args.random_seed)
        id_same_random = np.random.choice(id_same_filtered, num_random_samples, replace=False)

        # 重複したidを削除して統合
        id_same = np.unique(np.concatenate((id_same_random, id_same_fs)))
    else:
        assert False, "particle_distribution is invalid"

    plt.scatter(x_alpha[id_same], y_alpha[id_same], c='b', s=3)
    plt.xlim(x_min, x_max*2)
    plt.ylim(y_min, y_max*1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(dir_path_figure + '/subset.png')

    '''
    id_same     (799,)     # x_simに該当する粒子のインデックス
    id_free_x,  (60192,)   # 自由粒子のインデックス、x=0で壁に接していない粒子のインデックス
    id_u1_x,    (351,)     # 速度の境界条件を満たす粒子のインデックス、x=0で壁に接している粒子のインデックス
    id_free_y,  (60372,)   # 自由粒子のインデックス、y=0で壁に接していない粒子のインデックス
    id_u1_y,    (171,)     # 速度の境界条件を満たす粒子のインデックス、y=0で壁に接している粒子のインデックス
    id_free_p   (60020,)   # 自由粒子のインデックス、自由表面に接していない粒子のインデックス
    id_u1_p,    (523,)     # 圧力の境界条件を満たす粒子のインデックス、自由表面に接している粒子のインデックス
    x_alpha,    (60543, 1) # 全粒子のx座標
    y_alpha,    (60543, 1) # 全粒子のy座
    T,          (3, 60543) # PINNsの出力の初期値
    '''

    return id_same, id_free_x, id_u1_x, id_free_y, id_u1_y, id_free_p, id_u1_p, x_alpha, y_alpha, T