'''
use shooting to solve Legendre polynomial ODE
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

from scipy.integrate import solve_ivp
import scipy.optimize as opt

def dydx(x, _y, l):
    y, y_x = _y
    y_xx = (2 * x * y_x - l * (l + 1) * y) / (1 - x**2)
    return [y_x, y_xx]

def wrons(l, args, xmid=0, eps=1e-6):
    yl0 = 1 # arbitrary normalization
    yl0_x = -l / 2
    yr0 = 1
    yr0_x = -l / 2
    retl = solve_ivp(dydx, (-1 + eps, xmid), (yl0, yl0_x), args=[l], **args)
    retr = solve_ivp(dydx, (1 - eps, xmid), (yr0, yr0_x), args=[l], **args)
    ylf, ylf_x = retl.y[:, -1]
    yrf, yrf_x = retr.y[:, -1]
    return ylf * yrf_x - ylf_x * yrf

def get_y(l, args, xmid=0, eps=1e-6):
    yl0 = 1 # arbitrary normalization
    yl0_x = -l / 2
    yr0 = 1
    yr0_x = -l / 2
    retl = solve_ivp(dydx, (-1 + eps, xmid), (yl0, yl0_x), args=[l], **args)
    retr = solve_ivp(dydx, (1 - eps, xmid), (yr0, yr0_x), args=[l], **args)
    ylf, ylf_x = retl.y[:, -1]
    yrf, yrf_x = retr.y[:, -1]
    mat = np.array([[ylf, -yrf], [ylf_x, -yrf_x]])
    eigs, eigvs = np.linalg.eig(mat)
    z_idx = np.argmin(np.abs(eigs))
    mults = eigvs[ :, z_idx]
    return (
        np.concatenate((retl.t, retr.t[ ::-1])),
        np.concatenate((mults[0] * retl.y[0,  :], mults[1] * retr.y[0,  ::-1])))

if __name__ == '__main__':
    args = dict(atol=1e-9, rtol=1e-9)

    # plot W(l)
    l_arr = np.linspace(0.5, 2.5, 21)
    w_arr = [wrons(l, args) for l in l_arr]
    plt.plot(l_arr, w_arr)
    plt.xlabel('l')
    plt.ylabel('det W')
    plt.tight_layout()
    plt.savefig('1legendre', dpi=200)
    plt.close()

    # plot an eigen
    for y0 in range(2, 7):
        l_crit = opt.brenth(wrons, y0 - 0.2, y0 + 0.2, args=args)
        x_crit, y_crit = get_y(l_crit, args)
        plt.plot(x_crit, y_crit, label='%.5f' % l_crit)
    plt.legend()
    plt.tight_layout()
    plt.savefig('1sols', dpi=200)
    plt.close()
