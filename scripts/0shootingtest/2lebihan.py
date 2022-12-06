'''
use shooting to solve Le Bihan Eqs
'''
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
import os, pickle, lzma

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.optimize as opt

XMID = 0.03
EPS = 1e-2

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

def wrons(wsq, l, Vg_x, U_x, c1_x, As_x, xmid=XMID, eps=EPS, atol=1e-9,
          rtol=1e-9, method='DOP853', **kwargs):
    '''
    we need two linearly independent solutions; yc1, yc2, ys1, ys2
    '''
    f1 = 1
    f2 = 2
    g1 = 1
    g2 = 2
    yc_0_base = 1 # eps**(l - 2)
    yc1_0 = [
        yc_0_base,
        c1_x(eps) * wsq / l * yc_0_base,
        f1 * yc_0_base,
        f1 * l * yc_0_base,
    ]
    yc2_0 = [
        yc_0_base,
        c1_x(eps) * wsq / l * yc_0_base,
        f2 * yc_0_base,
        l * yc_0_base,
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
    kwargs = dict(atol=atol, rtol=rtol, method=method, **kwargs)
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
    print(wsq, det)
    return det

def get_y(wsq, l, Vg_x, U_x, c1_x, As_x, xmid=XMID, eps=EPS, **kwargs):
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

G = 6.67e-8
def build_polytrope(n=3, eps=EPS, M=2e33, R=7e10, method='DOP853',
                    atol=1e-9, rtol=1e-9, **args):
    '''
    follow https://www.astro.princeton.edu/~gk/A403/polytrop.pdf Eq 6

    2x * qp + x^2 * qp' = -q^n * x^2
    q' = qp
    '''
    if n >= 5:
        raise ValueError('Unbound star with n = %d' % n)

    y0 = [1, 0]
    def dqdx(x, y, n):
        q, qp = y
        return [
            qp,
            -q**n - 2 * qp / x,
        ]

    def stop_cond(x, y, n):
        return y[0]
    stop_cond.terminal = True

    ret = solve_ivp(dqdx, (eps, np.inf), y0, args=[n],
                    events=[stop_cond], method=method, atol=atol, rtol=rtol,
                    dense_output=True, **args)

    # x = ret.t
    # q, qp = ret.y
    # q = np.maximum(q, np.zeros_like(q)) # regularize

    x = np.geomspace(ret.t[0], ret.t[-1], 1000)
    q, qp = ret.sol(x)
    q = np.maximum(q, np.zeros_like(q)) # regularize
    # solved dimensionless Lane-Emden, now scale for a given M & R
    N_n = (
        (4 * np.pi)**(1 / n) / (n + 1)
        * (-x[-1]**2 * qp[-1])**((1 - n) / n)
        * x[-1]**((n - 3) / n))
    K = R**((3 - n) / n) * M**((n - 1) / n) * G * N_n
    rho_c = (
        (R / x[-1] / (K / G * (n + 1) / (4 * np.pi))**(1/2))**(2 * n / (1 - n))
    )
    alpha = (K *(n + 1) * rho_c**((1 - n) / n) / (4 * np.pi * G))**(1/2)
    rho_avg = 3 * M / (4 * np.pi * R**3)
    # print(x[-1], rho_c, rho_c / rho_avg, N_n) # all correct

    r = alpha * x
    rho = rho_c * q**n
    dr_left = np.concatenate(([0], r[1: ] - r[ :-1]))
    dr_right = np.concatenate((r[1: ] - r[ :-1], [0]))
    dm_left = rho * 4 * np.pi * r**2 * dr_left
    dm_right = rho * 4 * np.pi * r**2 * dr_right
    dm = (dm_left + dm_right) / 2 # trapezoidal rule effectively

    c_sq = (1 + 1 / n) * K * rho**(1 / n) # = Gamma * P / rho
    c_sq[-1] = c_sq[-2]
    m_r = np.cumsum(dm)
    g = G * m_r / r**2
    Vg = g * r / c_sq
    U = 4 * np.pi * rho * r**3 / m_r
    c1 = (r / R) / (m_r / M)
    As = np.zeros_like(r)
    rnorm = r / R

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    #     2, 2,
    #     figsize=(8, 8),
    #     sharex=True)
    # ax1.plot(rnorm, Vg, 'bo')
    # ax1.set_ylabel('Vg')
    # ax2.plot(rnorm, U * rnorm, 'bo')
    # ax2.set_ylabel('U * r')
    # ax3.plot(rnorm, c1 * rnorm**2, 'bo')
    # ax3.set_ylabel('c1 * r^2')
    # ax4.plot(rnorm, rho / rho_c, 'bo')
    # ax4.set_ylabel('rho / rho_c')
    # plt.tight_layout()
    # plt.savefig('/tmp/foo')
    # plt.close()

    return rnorm, Vg, U, c1, As

def wrons_nokw(wsq, l, Vg_x, U_x, c1_x, As_x, atol, rtol, method):
    return wrons(wsq, l, Vg_x, U_x, c1_x, As_x,
                 atol=atol, rtol=rtol, method=method)
def opt_func(y0, l, Vg_x, U_x, c1_x, As_x):
    my_opt = lambda wsq: wrons_nokw(wsq, l, Vg_x, U_x, c1_x, As_x,
                                      1e-9, 1e-9, 'DOP853')
    return opt.brenth(my_opt, y0 - 0.1, y0 + 0.1)
def sweep_polytrope(n=3):
    x, Vg, U, c1, As = build_polytrope(n)
    l = 1
    Vg_x = interp1d(x, Vg)
    U_x = interp1d(x, U)
    c1_x = interp1d(x, c1)
    As_x = interp1d(x, As)

    # plot W(wsq)
    wsq_arr = np.linspace(2, 20, 201)
    pkl_fn = '2polytrope.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        with Pool(16) as p:
            args = [
                (wsq, l, Vg_x, U_x, c1_x, As_x, 1e-9, 1e-9, 'DOP853')
                for wsq in wsq_arr
            ]
            w_arr = p.starmap(wrons_nokw, args)
        w_arr = np.array(w_arr)

        sign_changes = np.where(w_arr[1: ] / w_arr[ :-1] < 0)[0]
        cands = wsq_arr[sign_changes]
        print('%d sign changes detected' % len(cands))

        x_crit_lst = []
        y_crit_lst = []
        with Pool(16) as p:
            args = [
                (y0, l, Vg_x, U_x, c1_x, As_x)
                for y0 in cands
            ]
            wsq_crits = p.starmap(opt_func, args)
        for wsq_crit in wsq_crits:
            x_crit, y_crit = get_y(wsq_crit, l, Vg_x, U_x, c1_x, As_x,
                                   rtol=1e-9, atol=1e-9, method='DOP853')
            x_crit_lst.append(x_crit)
            y_crit_lst.append(y_crit)
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((w_arr, cands, wsq_crits, x_crit_lst, y_crit_lst), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            (w_arr, cands, wsq_crits, x_crit_lst, y_crit_lst) = pickle.load(f)

    wsq_crits = np.array(wsq_crits)
    plt.plot(wsq_arr, w_arr)
    plt.yscale('symlog', linthresh=1e4)
    plt.xlabel('wsq')
    plt.ylabel('det W')
    plt.tight_layout()
    plt.savefig('2sweep_pt', dpi=200)
    plt.close()

    # plot some eigens
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2,
        figsize=(10, 10),
        sharex=True)
    nu_mhzs = np.sqrt(
        G * 2e33 / (4 * np.pi**2 * 7e10**3) * wsq_crits) * 1e6
    for ax, nu_mhz, x_crit, y_crit in zip(
            [ax1, ax2, ax3, ax4, ax5, ax6], nu_mhzs, x_crit_lst, y_crit_lst):
        ax.loglog(x_crit, y_crit[0], c='g')
        ax.loglog(x_crit, -y_crit[0], c='g', ls='--')
        ax.set_title(r'$\nu = %.4f\;\mathrm{\mu Hz}$' % nu_mhz)
    ax5.set_xlabel(r'$x$')
    plt.tight_layout()
    plt.savefig('2sols_pt', dpi=200)
    plt.close()


if __name__ == '__main__':
    # sweep_test()

    # ret = build_polytrope()
    sweep_polytrope()
