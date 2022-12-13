'''
Instead of using interp1d for the polytropic stellar structure, use the
OdeSolution. This is much more accurate and much faster too, though it requires
a bit of finangling


According to my testing, instead implementing this by wrapping the OdeSolution
with a wrapper class that evaluates Vg, U, c1, As from q/q' of the Polytrope
solution slows the code down by 50%! Any overhead in the interpolation is
expensive.
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
import scipy.optimize as opt

XMID = 0.3
EPS = 1e-7

G = 6.67232e-8
MSUN = 1.989e33
RSUN = 6.9599e10
MHZ = 1e6

def dydx(x, y, wsq, l, ptrope_sol, x1):
    '''
    NB: though the variable name is ptrope_sol, it just needs to be any function
    that takes xi \in (0, x1) as its argument and returns Vg, U, c1_base, As
    '''
    y1, y2, y3, y4 = y
    xi = x * x1
    Vg, U, c1_base, As = ptrope_sol(xi)[ :4]
    c1 = c1_base / ptrope_sol(x1)[2]

    return np.array([
        y1 * (Vg - 3) + (l * (l + 1) / (c1 * wsq) - Vg) * y2 + Vg * y3,
        (c1 * wsq - As) * y1 + (As - U + 1) * y2 - As * y3,
        (1 - U) * y3 + y4,
        U * As * y1 + U * Vg * y2 + (l * (l + 1) - U * Vg) * y3 - U * y4,
    ]) / x

def wrons(wsq, l, ptrope_sol, x1, xmid=XMID, eps=EPS, atol=1e-9,
          rtol=1e-9, method='DOP853', **kwargs):
    '''
    we need two linearly independent solutions; yc1, yc2, ys1, ys2
    '''
    f1 = 1
    f2 = -1
    g1 = 1
    g2 = -1
    yc_0_base = eps**(l - 2)
    c1_eps = ptrope_sol(eps / x1)[2]
    yc1_0 = [
        yc_0_base,
        c1_eps * wsq / l * yc_0_base,
        f1 * yc_0_base,
        f1 * l * yc_0_base,
    ]
    yc2_0 = [
        yc_0_base,
        c1_eps * wsq / l * yc_0_base,
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
    args = [wsq, l, ptrope_sol, x1]
    retc1 = solve_ivp(dydx, (eps, xmid), yc1_0, args=args, **kwargs)
    retc2 = solve_ivp(dydx, (eps, xmid), yc2_0, args=args, **kwargs)
    rets1 = solve_ivp(dydx, (1 - eps, xmid), ys1_0, args=args, **kwargs)
    rets2 = solve_ivp(dydx, (1 - eps, xmid), ys2_0, args=args, **kwargs)
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
    # print('%.15f' % wsq, det)
    return det

def get_y(wsq, l, ptrope_sol, x1, xmid=XMID, eps=EPS, **kwargs):
    '''
    we need two linearly independent solutions; yc1, yc2, ys1, ys2
    '''
    f1 = 1e-3
    f2 = -1e-3
    g1 = 1e-3
    g2 = -1e-3
    yc_0_base = 1e3 # eps**(l - 2)
    c1_eps = ptrope_sol(eps / x1)[2]
    yc1_0 = [
        yc_0_base,
        c1_eps * wsq / l * yc_0_base,
        f1 * yc_0_base,
        f1 * l * yc_0_base,
    ]
    yc2_0 = [
        yc_0_base,
        c1_eps * wsq / l * yc_0_base,
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
    args = [wsq, l, ptrope_sol, x1]
    retc1 = solve_ivp(dydx, (eps, xmid), yc1_0, args=args, dense_output=True, **kwargs)
    retc2 = solve_ivp(dydx, (eps, xmid), yc2_0, args=args, dense_output=True,**kwargs)
    rets1 = solve_ivp(dydx, (1 - eps, xmid), ys1_0, args=args, dense_output=True,**kwargs)
    rets2 = solve_ivp(dydx, (1 - eps, xmid), ys2_0, args=args, dense_output=True,**kwargs)
    retc1 = solve_ivp(dydx, (eps, xmid), yc1_0, args=args, dense_output=True,**kwargs)
    retc2 = solve_ivp(dydx, (eps, xmid), yc2_0, args=args, dense_output=True,**kwargs)
    rets1 = solve_ivp(dydx, (1 - eps, xmid), ys1_0, args=args, dense_output=True,**kwargs)
    rets2 = solve_ivp(dydx, (1 - eps, xmid), ys2_0, args=args, dense_output=True,**kwargs)
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

def build_polytrope(n=3, eps=EPS, M=MSUN, R=RSUN, method='DOP853',
                    npts=1000, atol=1e-9, rtol=1e-9, plot=False, **args):
    '''
    follow my notes


    let the ODE solver also keep track of our quantities of interest:
    '''
    if n >= 5:
        raise ValueError('Unbound star with n = %d' % n)

    Gamma1 = 5/3
    Gamma = 1 + 1 / n
    Vg0 = (n + 1) * eps**2 / (3 * Gamma1)
    U0 = 3
    c1_base0 = -3
    A0 = (n + 1) * eps**2 / 3 * (1 / Gamma - 1 / Gamma1)
    y0 = [Vg0, U0, c1_base0, A0, 1, -eps / 3]
    def dqdx(x, y, n):
        q, qp = y[-2: ]
        qpp = -q**n - 2 * qp / x
        # Vg', U', c1_base', A'
        return [
            -(n + 1) / Gamma1 * (qp / q + x * qpp / q - x * qp**2 / q**2),
            -(n * x * q**(n - 1) + q**n / qp - x * q**n * qpp / qp**2),
            1 / qp - x * qpp / qp**2,
            -(n + 1)* (1 / Gamma - 1 / Gamma1)
                * (qp / q + x * qpp / q - x * qp**2 / q**2),
            qp,
            qpp,
        ]

    def stop_cond(x, y, n):
        return y[0]
    stop_cond.terminal = True

    ret = solve_ivp(dqdx, (eps, np.inf), y0, args=[n],
                    events=[stop_cond], method=method, atol=atol, rtol=rtol,
                    dense_output=True, **args)

    return ret.t, ret.sol

def wrons_nokw(wsq, l, ptrope_sol, x1, xmid, eps, atol, rtol, method):
    return wrons(wsq, l, ptrope_sol, x1,
                 xmid=xmid, eps=eps, atol=atol, rtol=rtol, method=method)
def opt_func(yl, yr, l, ptrope_sol, x1, xmid=XMID, eps=EPS, atol=1e-9,
             rtol=1e-9, method='DOP853'):
    my_opt = lambda wsq: wrons_nokw(wsq, l, ptrope_sol, x1,
                                      xmid, eps, atol, rtol, method)
    try:
        return opt.brenth(my_opt, yl, yr, xtol=rtol)
    except ValueError:
        print('ERROR: f(a) f(b) do not have opposite signs', yl, yr)
        raise
def sweep_polytrope(n=3, l=1, wsq_arr=np.linspace(2, 20, 201), nthreads=16,
                    xmid=XMID, eps=EPS,
                    tol=1e-9, method='DOP853', fn='2polytrope'):
    x, ptrope_sol = build_polytrope(n, eps=eps)
    x1 = x[-1]
    dwsq = np.mean(np.diff(wsq_arr))
    print('(l, tol, eps, xmid)', l, tol, eps, xmid)

    # plot W(wsq)
    pkl_fn = '%s.pkl' % fn
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        with Pool(nthreads) as p:
            args = [
                (wsq, l, ptrope_sol, x1, xmid, eps, tol, tol, method)
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
                (yl, yr, l, ptrope_sol, x1, xmid, eps, tol, tol, method)
                for yl, yr in zip(cands_left, cands_right)
            ]
            wsq_crits = p.starmap(opt_func, args)
        for wsq_crit in wsq_crits:
            x_crit, y_crit = get_y(wsq_crit, l, ptrope_sol, x1,
                                   eps=eps, rtol=tol, atol=tol, method=method)
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
    for n, nu in enumerate(nu_mhzs):
        print(n + 1, nu)
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
    # tol = 1e-10
    # eps = 1e-7
    # xmid = 0.3
    # x, ptrope_sol = build_polytrope(3, atol=tol, rtol=tol, eps=eps)

    # l = 1
    # acc = opt_func(49, 55, l, ptrope_sol, x[-1], method='DOP853',
    #                atol=tol, rtol=tol, eps=eps, xmid=xmid)
    # print(acc)
    # print(np.sqrt(G * MSUN / (4 * np.pi**2 * RSUN**3) * acc) * MHZ)

    os.makedirs('3sweeps', exist_ok=True)
    kws = dict(wsq_arr = np.linspace(5, 5000, 1024), nthreads=8,
               method='DOP853')
    sweep_polytrope(l=1, tol=1e-8, fn='3sweeps/3ptpl1', **kws)
    sweep_polytrope(l=1, tol=1e-10, fn='3sweeps/3ptpl1_10', **kws)
    sweep_polytrope(l=1, tol=1e-12, fn='3sweeps/3ptpl1_12', **kws)
    sweep_polytrope(l=1, tol=1e-10, eps=1e-5, fn='3sweeps/3ptpl1_10ep5', **kws)
    sweep_polytrope(l=1, tol=1e-10, eps=1e-3, fn='3sweeps/3ptpl1_10ep3', **kws)

    sweep_polytrope(l=2, tol=1e-10, fn='3sweeps/3ptpl2_10', **kws)
    sweep_polytrope(l=3, tol=1e-10, fn='3sweeps/3ptpl3_10', **kws)

    kws_g = dict(wsq_arr = np.linspace(5, 5e-3, 1024), nthreads=8,
               method='DOP853')
    sweep_polytrope(l=2, tol=1e-12, fn='3sweeps/3ptgl1_12', **kws_g)
