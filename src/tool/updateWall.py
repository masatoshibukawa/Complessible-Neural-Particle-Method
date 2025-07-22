"""
Author: masato shibukawa
Date: 2025-07-22
Email: shibukawa.masato@ac.jaxa.jp

Wall boundary condition update utilities for neural particle methods.
Handles particle interactions with domain boundaries, including collision detection,
boundary reflections, and constraint enforcement for fluid-structure interaction.
"""

import numpy as np
import tool.init as init
from tool.plot import plot #描画関数

def updateWall(x0, y0, id_u1_x_0, id_u1_y_0, u0_0):
    x = x0.copy()
    y = y0.copy()
    id_u1_x = id_u1_x_0.copy()
    id_u1_y = id_u1_y_0.copy()
    u0 = u0_0.copy()

    tol   = 1e-3
    for i in range(0, len(id_u1_x)):
        id = id_u1_x[i]

        if (y[id]<tol) & (u0[id, 1]<=0):
            print('shift X-boun (%.5f, %.5f) ' % (x[id].item(), y[id].item()))
            if x[id]==0:
                y[id] = 0
                x[id] = tol

                u0[id, 0] = -u0[id, 1]
                u0[id, 1] = 0
            print('after shift: (%.5f, %.5f)' % (x[id].item(), y[id].item()))
            '''switch id'''
            id_u1_y = np.append(id_u1_y, id)
            id_u1_x = np.delete(id_u1_x, i)
            break

    for i in range(0, len(id_u1_y)):
        id = id_u1_y[i]

        if (x[id]<tol) & (u0[id, 0]<=0):
            print('shift Y-boun (%.5f, %.5f)' % (x[id].item(), y[id].item()))

            x[id] = 0
            y[id] = tol

            u0[id, 1] = -u0[id, 0]
            u0[id, 0] = 0

            print('after shift: (%.5f, %.5f)' % (x[id].item(), y[id].item()))

            '''switch id'''
            id_u1_x = np.append(id_u1_x, id)
            id_u1_y = np.delete(id_u1_y, i)
            break

    return x, y, id_u1_x, id_u1_y, u0