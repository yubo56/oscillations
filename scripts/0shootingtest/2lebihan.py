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

XMID = 0.3
EPS = 1e-3
MSUN = 1.989e33
RSUN = 6.9599e10
MHZ = 1e6

def dydx(x, y, wsq, l, Vg_x, U_x, c1_x, As_x):
    y1, y2, y3, y4 = y
    Vg = Vg_x(x)
    U = U_x(x)
    c1 = c1_x(x)
    As = As_x(x)

    return np.array([
        y1 * (Vg - 3) + (l * (l + 1) / (c1 * wsq) - Vg) * y2 + Vg * y3,
        (c1 * wsq - As) * y1 + (As - U + 1) * y2 - As * y3,
        (1 - U) * y3 + y4,
        U * As * y1 + U * Vg * y2 + (l * (l + 1) - U * Vg) * y3 - U * y4,
    ]) / x

def wrons(wsq, l, Vg_x, U_x, c1_x, As_x, xmid=XMID, eps=EPS, atol=1e-9,
          rtol=1e-9, method='DOP853', **kwargs):
    '''
    we need two linearly independent solutions; yc1, yc2, ys1, ys2
    '''
    f1 = 1
    f2 = -1
    g1 = 1
    g2 = -1
    yc_0_base = eps**(l - 2)
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
        1 + g1,
        g1,
        -g1 * (l + 1),
    ]
    ys2_0 = [
        1,
        1 + g2,
        g2,
        -g2 * (l + 1),
    ]
    kwargs = dict(atol=atol, rtol=rtol, method=method, **kwargs)
    args = [wsq, l, Vg_x, U_x, c1_x, As_x]
    retc1 = solve_ivp(dydx, (eps, xmid), yc1_0, args=args, **kwargs)
    retc2 = solve_ivp(dydx, (eps, xmid), yc2_0, args=args, **kwargs)
    rets1 = solve_ivp(dydx, (1, xmid), ys1_0, args=args, **kwargs)
    rets2 = solve_ivp(dydx, (1, xmid), ys2_0, args=args, **kwargs)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2,
        figsize=(8, 8),
        gridspec_kw={'height_ratios': [1, 1]})
    ax1.loglog(retc1.t,  retc1.y[1])
    ax1.loglog(retc1.t, -retc1.y[1], ls='--')
    ax2.loglog(retc2.t,  retc2.y[1])
    ax2.loglog(retc2.t, -retc2.y[1], ls='--')
    ax3.loglog(rets1.t,  rets1.y[1])
    ax3.loglog(rets1.t, -rets1.y[1], ls='--')
    ax4.loglog(rets2.t,  rets2.y[1])
    ax4.loglog(rets2.t, -rets2.y[1], ls='--')
    plt.savefig('/tmp/foo')
    plt.close()

    wronskian = np.array([
        retc1.y[ :,-1],
        retc2.y[ :,-1],
        -rets1.y[ :,-1],
        -rets2.y[ :,-1]])
    det = np.linalg.det(wronskian)
    print('%.15f' % wsq, det)
    return det

def shoot(wsq, l, Vg_x, U_x, c1_x, As_x, xmid=XMID, eps=EPS, atol=1e-9,
          rtol=1e-9, method='DOP853', **kwargs):
    '''
    Same interface as wrons, but return the deviation from the BCs at the outer
    boundary
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
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    #     2, 2,
    #     figsize=(8, 8),
    #     gridspec_kw={'height_ratios': [1, 1]})
    # ax1.plot(retc1.t, retc1.y[0])
    # ax2.plot(retc2.t, retc2.y[0])
    # ax3.plot(rets1.t, rets1.y[0])
    # ax4.plot(rets2.t, rets2.y[0])
    # plt.savefig('/tmp/foo')
    # plt.close()

    wronskian = np.array([
        retc1.y[ :,-1],
        retc2.y[ :,-1],
        -rets1.y[ :,-1],
        -rets2.y[ :,-1]])
    det = np.linalg.det(wronskian)
    print('%.15f' % wsq, det)
    return det

def get_y(wsq, l, Vg_x, U_x, c1_x, As_x, xmid=XMID, eps=EPS, **kwargs):
    '''
    we need two linearly independent solutions; yc1, yc2, ys1, ys2
    '''
    f1 = 1e-3
    f2 = -1e-3
    g1 = 1e-3
    g2 = -1e-3
    yc_0_base = 1e3 # eps**(l - 2)
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
        1 + g1,
        g1,
        -g1 * (l + 1),
    ]
    ys2_0 = [
        1,
        1 + g2,
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
        opt_wsq = lambda wsq: wrons(wsq, **struct_args, **solve_args)
        wsq_crit = opt.brenth(opt_wsq, y0 - 0.1, y0 + 0.1, rtol=1e-12)
        x_crit, y_crit = get_y(wsq_crit, **struct_args, **solve_args)
        ax.loglog(x_crit, y_crit[0], c='g')
        ax.loglog(x_crit, -y_crit[0], c='g', ls='--')
        ax.set_title(r'$\omega^2 = %.5f$' % wsq_crit)
    ax5.set_xlabel(r'$x$')
    plt.tight_layout()
    plt.savefig('2sols', dpi=200)
    plt.close()

G = 6.67e-8
def build_polytrope(n=3, eps=EPS, M=MSUN, R=RSUN, method='DOP853',
                    atol=1e-9, rtol=1e-9, plot=False, **args):
    '''
    follow my notes
    '''
    if n >= 5:
        raise ValueError('Unbound star with n = %d' % n)

    y0 = [1, -eps / 3]
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
    Gamma1 = 5/3
    Gamma = 1 + 1 / n

    # x = ret.t
    # q, qp = ret.y
    # q = np.maximum(q, np.zeros_like(q)) # regularize

    x = np.geomspace(ret.t[0], ret.t[-1], 1000)[ :-1]
    # drop last point since a lot of stuff vanishes here
    q, qp = ret.sol(x)

    x1 = x[-1]
    rho_c = (M / (4 * np.pi * R**3 / 3)) * (-x1 / (3 * qp[-1]))
    K = (R / x1)**2 * rho_c**((n - 1) / n) * 4 * np.pi * G / (n + 1)
    alpha = (K * (n + 1) * rho_c**((1 - n) / n) / (4 * np.pi * G))**(1/2)

    rho = rho_c * q**n
    m_r = -4 * np.pi * alpha**3 * rho_c * x**2 * qp
    g = -4 * np.pi * G * alpha * rho_c * qp
    U = -q**n * x / qp
    c1 = (x / x1) / (qp / qp[-1])
    c_sq = Gamma1 * K * rho_c**(1 / n) * q
    As = -(1 / Gamma - 1 / Gamma1) * (n + 1) * x * qp / q
    Vg = -(1 / Gamma1) * (n + 1) * x * qp / q
    r = alpha * x
    rnorm = r / R

    ###### OLD: USED Princeton astro class notes
    # solved dimensionless Lane-Emden, now scale for a given M & R
    # N_n = (
    #     (4 * np.pi)**(1 / n) / (n + 1)
    #     * (-x[-1]**2 * qp[-1])**((1 - n) / n)
    #     * x[-1]**((n - 3) / n))
    # K = R**((3 - n) / n) * M**((n - 1) / n) * G * N_n
    # rho_c = (
    #     (R / x[-1] / (K / G * (n + 1) / (4 * np.pi))**(1/2))**(2 * n / (1 - n))
    # )
    # alpha = (K * (n + 1) * rho_c**((1 - n) / n) / (4 * np.pi * G))**(1/2)
    # rho_avg = 3 * M / (4 * np.pi * R**3)
    # # print(x[-1], rho_c, rho_c / rho_avg, N_n) # all correct

    # r = alpha * x
    # rho = rho_c * q**n
    # m_r = (
    #     4 * np.pi * ((K * (n + 1)) / (G * 4 * np.pi))**(3/2)
    #     * rho_c**((3 - n) / (2 * n))
    #     * (-x**2 * qp))
    # m_r[1] = m_r[0] # to avoid singularity

    # c_sq = Gamma1 * K * rho**(1 / n) # = Gamma1 * P / rho
    # c_sq[-1] = c_sq[-2] # to avoid singularity
    # g = G * m_r / r**2
    # Vg = g * r / c_sq
    # U = 4 * np.pi * rho * r**3 / m_r
    # c1 = (r / R) / (m_r / M)
    # As = g * r / (K * rho**(1 / n)) * (1 / (1 + 1 / n) - 1 / Gamma1)
    # As[-1] = As[-2] # to avoid singularity
    # rnorm = r / R

    if plot == True:
        kwargs = dict(c='k', marker='o', markersize=1, lw=0.5, ls='')

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6),
              (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(
            4, 3,
            figsize=(12, 12),
            sharex=True)
        ax1.semilogy(rnorm, q, **kwargs)
        ax1.set_ylabel(r'$\theta$')
        dqdr = np.diff(q) / np.diff(rnorm)
        ax2.semilogy((rnorm[1: ] + rnorm[ :-1]) / 2, -dqdr, **kwargs)
        ax2.set_ylabel(r'$-d\theta/dr$')
        ax3.semilogy(rnorm[ :-1], rho[ :-1] / rho_c, **kwargs)
        ax3.set_ylabel(r'$\rho / \rho_c$')
        ax4.semilogy(rnorm, m_r, **kwargs)
        ax4.set_ylabel(r'$M_r$')
        ax5.semilogy(rnorm, g, **kwargs)
        ax5.set_ylabel(r'$g$')
        ax6.semilogy(rnorm, c_sq, **kwargs)
        ax6.set_ylabel(r'$c^2$')

        ax7.semilogy(rnorm, Vg, **kwargs)
        ax7.set_ylabel(r'$V_g$')
        ax8.semilogy(rnorm, U, **kwargs)
        ax8.set_ylabel(r'$U r$')
        ax9.semilogy(rnorm, c1, **kwargs)
        ax9.set_ylabel(r'$c_1$')

        ax10.semilogy(rnorm, As, **kwargs)
        ax10.set_ylabel(r'$A^*$')
        # ax1.set_xlim(-eps, 10 * eps)
        ax1.set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig('/tmp/foo', dpi=300)
        plt.close()

    Vg_x = interp1d(rnorm, Vg)
    U_x = interp1d(rnorm, U)
    c1_x = interp1d(rnorm, c1)
    As_x = interp1d(rnorm, As)
    return rnorm, Vg_x, U_x, c1_x, As_x

def wrons_nokw(wsq, l, Vg_x, U_x, c1_x, As_x, xmid, eps, atol, rtol, method):
    return wrons(wsq, l, Vg_x, U_x, c1_x, As_x,
                 xmid=xmid, eps=eps, atol=atol, rtol=rtol, method=method)
def opt_func(yl, yr, l, Vg_x, U_x, c1_x, As_x, xmid=XMID, eps=EPS, atol=1e-9,
             rtol=1e-9, method='DOP853'):
    my_opt = lambda wsq: wrons_nokw(wsq, l, Vg_x, U_x, c1_x, As_x,
                                      xmid, eps, atol, rtol, method)
    try:
        return opt.brenth(my_opt, yl, yr, xtol=rtol)
    except ValueError:
        print('ERROR: f(a) f(b) have opposite signs', yl, yr)
        raise
def sweep_polytrope(n=3, l=1, wsq_arr=np.linspace(2, 20, 201), nthreads=16,
                    xmid=XMID, eps=EPS,
                    atol=1e-9, rtol=1e-9, method='DOP853', fn='2polytrope'):
    x, Vg_x, U_x, c1_x, As_x = build_polytrope(n, eps=eps)
    dwsq = np.mean(np.diff(wsq_arr))

    # plot W(wsq)
    pkl_fn = '%s.pkl' % fn
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        with Pool(nthreads) as p:
            args = [
                (wsq, l, Vg_x, U_x, c1_x, As_x, xmid, eps, atol, rtol, method)
                for wsq in wsq_arr
            ]
            w_arr = p.starmap(wrons_nokw, args)
        w_arr = np.array(w_arr)

        sign_changes = np.where(w_arr[1: ] / w_arr[ :-1] < 0)[0]
        cands_left = wsq_arr[sign_changes]
        cands_right = wsq_arr[sign_changes + 1]
        print('%d sign changes detected' % len(cands_left))

        x_crit_lst = []
        y_crit_lst = []
        with Pool(nthreads) as p:
            args = [
                (yl, yr, l, Vg_x, U_x, c1_x, As_x, xmid, eps, atol, rtol, method)
                for yl, yr in zip(cands_left, cands_right)
            ]
            wsq_crits = p.starmap(opt_func, args)
        for wsq_crit in wsq_crits:
            x_crit, y_crit = get_y(wsq_crit, l, Vg_x, U_x, c1_x, As_x,
                                   rtol=rtol, atol=atol, method=method)
            x_crit_lst.append(x_crit)
            y_crit_lst.append(y_crit)
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((w_arr, cands_left, cands_right, wsq_crits, x_crit_lst, y_crit_lst), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            (w_arr, cands_left, cands_right, wsq_crits, x_crit_lst, y_crit_lst) = pickle.load(f)

    wsq_crits = np.array(wsq_crits)
    plt.semilogy(wsq_arr, w_arr, 'k')
    plt.semilogy(wsq_arr, -w_arr, 'k--')
    plt.xlabel('wsq')
    plt.ylabel('det W')
    plt.tight_layout()
    plt.savefig('%s_sweep' % fn, dpi=200)
    plt.close()

    # plot some eigens
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2,
        figsize=(10, 10),
        sharex=True)
    nu_mhzs = np.sqrt(
        G * MSUN / (4 * np.pi**2 * RSUN**3) * wsq_crits) * MHZ
    for ax, nu_mhz, x_crit, y_crit in zip(
            [ax1, ax2, ax3, ax4, ax5, ax6], nu_mhzs, x_crit_lst, y_crit_lst):
        ax.loglog(x_crit, y_crit[0], c='g')
        ax.loglog(x_crit, -y_crit[0], c='g', ls='--')
        ax.set_title(r'$\nu = %.4f\;\mathrm{\mu Hz}$' % nu_mhz)
    ax5.set_xlabel(r'$x$')
    plt.tight_layout()
    plt.savefig('%s_sols' % fn, dpi=200)
    plt.close()


if __name__ == '__main__':
    # sweep_test()

    # build_polytrope(plot=True)

    x, Vg_x, U_x, c1_x, As_x = build_polytrope(3)
    # print('Vg (0, inf):', Vg_x(EPS), Vg_x(1))
    # print('U (3, 0):', U_x(EPS), U_x(1))
    # print('c1 (>0, 1):', c1_x(EPS), c1_x(1))
    # print('As (0, inf):', As_x(EPS), As_x(1))
    # wrons(74, 2, Vg_x, U_x, c1_x, As_x)

    # x, Vg_x, U_x, c1_x, As_x = build_polytrope(3)
    # acc = opt_func(73, 76, 2, Vg_x, U_x, c1_x, As_x, method='DOP853',
    #                atol=1e-12, rtol=1e-12)
    # print(acc)
    # print(np.sqrt(G * MSUN / (4 * np.pi**2 * RSUN**3) * acc) * MHZ)
    # x, y = get_y(acc, 2, Vg_x, U_x, c1_x, As_x, method='DOP853', atol=1e-12,
    #              rtol=1e-12)
    # plt.loglog(x, y[0], 'k')
    # plt.loglog(x, -y[0], 'k--')
    # plt.axvline(XMID, c='b')
    # plt.tight_layout()
    # plt.savefig('/tmp/foo')
    # plt.close()

    sweep_polytrope(wsq_arr=np.linspace(5, 100, 128), l=1, nthreads=8,
                    atol=1e-8, rtol=1e-8, method='DOP853',
                    fn='2polyl1')
    sweep_polytrope(wsq_arr=np.linspace(5, 100, 128), l=1, nthreads=8,
                    atol=1e-10, rtol=1e-10, method='DOP853',
                    fn='2polyl1_10')
    sweep_polytrope(wsq_arr=np.linspace(5, 100, 128), l=1, nthreads=8,
                    atol=1e-10, rtol=1e-10, method='DOP853', eps=1e-4,
                    fn='2polyl1_10eps4')
    sweep_polytrope(wsq_arr=np.linspace(5, 100, 128), l=1, nthreads=8,
                    atol=1e-10, rtol=1e-10, method='DOP853', eps=1e-2,
                    fn='2polyl1_10eps2')
