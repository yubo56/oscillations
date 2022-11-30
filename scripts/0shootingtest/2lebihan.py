'''
use shooting to solve Le Bihan Eqs
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

xmid = 0.03
eps = 1e-4

def dydx(x, y, wsq, l, Vg_x, U_x, c1_x, As_x):
    y1, y2, y3, y4 = y
    Vg = Vg_x(x)
    U = U_x(x)
    c1 = c1_x(x)
    As = As_x(x)

    return np.array([
        y1 * (Vg - 3) + (l * (l + 1) / (c1 * wsq) - Vg) * y2 + Vg * y3,
        (c1 * wsq - As) * y1 + (As - U + 1) * y2 + As * y3,
        (1 - U) * y3 + y4,
        U * As * y1 + U * Vg * y2 + (l * (l + 1) - U * Vg) * y3 + U * y4,
    ]) / x

def wrons(wsq, l, Vg_x, U_x, c1_x, As_x, xmid=xmid, eps=eps, **kwargs):
    '''
    we need two linearly independent solutions; yc1, yc2, ys1, ys2
    '''
    f1 = 1
    f2 = 2
    g1 = 1
    g2 = 2
    yc1_0 = [
        eps**(l - 2),
        c1_x(eps) * wsq / l * eps**(l - 2),
        f1 * eps**(l - 2),
        f1 * l * eps**(l - 2),
    ]
    yc2_0 = [
        eps**(l - 2),
        c1_x(eps) * wsq / l * eps**(l - 2),
        f2 * eps**(l - 2),
        l * eps**(l - 2),
    ]
    ys1_0 = [
        1,
        2,
        g1,
        -g1 * (l + 1),
    ]
    ys2_0 = [
        1,
        2,
        g2,
        -g2 * (l + 1),
    ]
    args = [wsq, l, Vg_x, U_x, c1_x, As_x]
    retc1 = solve_ivp(dydx, (eps, xmid), yc1_0, args=args, **kwargs)
    retc2 = solve_ivp(dydx, (eps, xmid), yc2_0, args=args, **kwargs)
    rets1 = solve_ivp(dydx, (1, xmid), ys1_0, args=args, **kwargs)
    rets2 = solve_ivp(dydx, (1, xmid), ys2_0, args=args, **kwargs)
    wronskian = np.array([
        retc1.y[ :,-1],
        retc2.y[ :,-1],
        rets1.y[ :,-1],
        rets2.y[ :,-1]])
    det = np.linalg.det(wronskian)
    return det

def get_y(wsq, l, Vg_x, U_x, c1_x, As_x, xmid=xmid, eps=eps, **kwargs):
    '''
    we need two linearly independent solutions; yc1, yc2, ys1, ys2
    '''
    f1 = -1
    f2 = 1
    g1 = -1
    g2 = 1
    yc1_0 = [
        eps,
        c1_x(eps) * wsq / l * eps,
        f1 * eps,
        f1 * l * eps,
    ]
    yc2_0 = [
        eps,
        c1_x(eps) * wsq / l * eps,
        f2 * eps,
        l * eps,
    ]
    ys1_0 = [
        1,
        2,
        g1,
        -g1 * (l + 1),
    ]
    ys2_0 = [
        1,
        2,
        g2,
        -g2 * (l + 1),
    ]
    args = [wsq, l, Vg_x, U_x, c1_x, As_x]
    retc1 = solve_ivp(dydx, (eps, xmid), yc1_0, args=args, dense_output=True, **kwargs)
    retc2 = solve_ivp(dydx, (eps, xmid), yc2_0, args=args, dense_output=True,**kwargs)
    rets1 = solve_ivp(dydx, (1, xmid), ys1_0, args=args, dense_output=True,**kwargs)
    rets2 = solve_ivp(dydx, (1, xmid), ys2_0, args=args, dense_output=True,**kwargs)
    retc1 = solve_ivp(dydx, (eps, xmid), yc1_0, args=args, dense_output=True,**kwargs)
    retc2 = solve_ivp(dydx, (eps, xmid), yc2_0, args=args, dense_output=True,**kwargs)
    rets1 = solve_ivp(dydx, (1, xmid), ys1_0, args=args, dense_output=True,**kwargs)
    rets2 = solve_ivp(dydx, (1, xmid), ys2_0, args=args, dense_output=True,**kwargs)
    wronskian = np.array([
        retc1.y[ :,-1],
        retc2.y[ :,-1],
        -rets1.y[ :,-1],
        -rets2.y[ :,-1]]).T
    eigs, eigvs = np.linalg.eig(wronskian)
    z_idx = np.argmin(np.abs(eigs))
    mults = np.real(eigvs[ :, z_idx])

    c_t = np.sort(np.concatenate((retc1.t, retc2.t)))
    y_c1 = retc1.sol(c_t)
    y_c2 = retc2.sol(c_t)
    s_t = np.sort(np.concatenate((rets1.t, rets2.t)))
    y_s1 = rets1.sol(s_t)
    y_s2 = rets2.sol(s_t)

    y_crit = np.concatenate((
        mults[0] * y_c1 + mults[1] * y_c2,
        mults[2] * y_s1 + mults[3] * y_s2), axis=1)
    y_crit *= np.sign(y_crit[0, -1])
    x_crit = np.concatenate((c_t, s_t))
    return x_crit, y_crit

def sweep_test():
    solve_args = dict(atol=1e-9, rtol=1e-9, method='DOP853')

    def C(x):
        ''' simple interp function for testing purposes '''
        return lambda y: x
    struct_args = dict(
        l=3, Vg_x=C(1.5), U_x=C(1.1), c1_x=C(1), As_x=C(1))

    # plot W(l)
    wsq_arr = np.linspace(2, 10, 101)
    w_arr = np.array([wrons(wsq, **struct_args, **solve_args)
                      for wsq in wsq_arr])
    plt.plot(wsq_arr, w_arr)
    plt.yscale('symlog', linthresh=1e7)
    plt.xlabel('wsq')
    plt.ylabel('det W')
    plt.tight_layout()
    plt.savefig('2sweep', dpi=200)
    plt.close()

    # plot some eigens
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2,
        figsize=(10, 10),
        sharex=True)

    sign_changes = np.where(w_arr[1: ] / w_arr[ :-1] < 0)[0]
    cands = wsq_arr[sign_changes]
    print('%d sign changes detected' % len(cands))
    for y0, ax in zip(cands, [ax1, ax2, ax3, ax4, ax5, ax6]):
        print(y0)
        opt_func = lambda wsq: wrons(wsq, **struct_args, **solve_args)
        wsq_crit = opt.brenth(opt_func, y0 - 0.1, y0 + 0.1, rtol=1e-12)
        x_crit, y_crit = get_y(wsq_crit, **struct_args, **solve_args)
        ax.loglog(x_crit, y_crit[0], c='g')
        ax.loglog(x_crit, -y_crit[0], c='g', ls='--')
        ax.set_title(r'$\omega^2 = %.5f$' % wsq_crit)
    ax5.set_xlabel(r'$x$')
    plt.tight_layout()
    plt.savefig('2sols', dpi=200)
    plt.close()

if __name__ == '__main__':
    sweep_test()
