'''
extended to general structures: first, broken polytrope
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
plt.rc('figure', figsize=(8.0, 8.0), dpi=300)
import os, pickle, lzma

from scipy.integrate import solve_ivp
import scipy.optimize as opt

XMID = 0.3
EPS = 1e-7

GSUN = 6.67232e-8
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
    c1_eps = ptrope_sol(eps / x1)[2] / ptrope_sol(x1)[2]
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
    c1_eps = ptrope_sol(eps / x1)[2] / ptrope_sol(x1)[2]
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

def build_polytrope(n=3, eps=EPS, method='DOP853',
                    npts=1000, atol=1e-9, rtol=1e-9, **args):
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
    x, ptrope_sol = build_broken_polytrope(n, eps=eps)
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
    nu_mhzs = np.sqrt(
        GSUN * MSUN / (4 * np.pi**2 * RSUN**3) * wsq_crits) * MHZ

    plt.semilogy(wsq_arr, w_arr, 'k')
    plt.semilogy(wsq_arr, -w_arr, 'k--')
    plt.xlabel('wsq')
    plt.ylabel('det W')
    plt.tight_layout()
    plt.savefig('%s_sweep' % fn)
    plt.close()

    # print mode orders & freqs
    for x_crit, y_crit, nu in zip(x_crit_lst, y_crit_lst, nu_mhzs):
        # get the mode order (ignore innermost points)
        offset = 3
        y1_zero_idxs = np.where(
            y_crit[0, offset + 1: ] / y_crit[0, offset:-1] < 0)[0]
        n = (
            0 if y_crit[0, offset] * y_crit[1, offset] > 0
            else 1
        )
        for z_idx in y1_zero_idxs:
            z_idx += offset
            dy1 = y_crit[0, z_idx + 1] - y_crit[0, z_idx]
            y2 = np.mean(y_crit[1, z_idx:z_idx + 2])
            n -= int(np.sign(dy1 * y2))
        print(n, nu)

    # plot some eigens
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2,
        figsize=(10, 10),
        sharex=True)
    for ax, nu_mhz, x_crit, y_crit in zip(
            [ax1, ax2, ax3, ax4, ax5, ax6], nu_mhzs, x_crit_lst, y_crit_lst):
        ax.semilogy(x_crit, y_crit[0], c='g')
        ax.semilogy(x_crit, -y_crit[0], c='g', ls='--')
        ax.semilogy(x_crit, y_crit[1], c='r')
        ax.semilogy(x_crit, -y_crit[1], c='r', ls='--')
        ax.set_title(r'$\nu = %.4f\;\mathrm{\mu Hz}$' % nu_mhz)
    ax5.set_xlabel(r'$x$')
    plt.tight_layout()
    plt.savefig('%s_sols' % fn)
    plt.close()

def plot_polytrope(x, ptrope_sol, fn, n=3, G=GSUN, R=RSUN, M=MSUN, n_pts=1000):
    xi1 = x.max()
    xi = np.linspace(x.min(), xi1, n_pts)
    Vg, U, c1_base, As = ptrope_sol(xi)[ :4]
    c1 = c1_base / ptrope_sol(xi1)[2]

    Gamma = 1 + 1 / n

    r = xi / xi1 * R
    x = xi / xi1
    g = (G * M / R**3) * r / c1
    cs2 = g * r / Vg
    rho0 = U * g / (4 * np.pi  * G * r)
    P0 = (rho0 / Gamma) / (As / (g * r) + 1 / cs2)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2,
        figsize=(8, 8),
        sharex=True)
    ax1.plot(x, g / 1e2, 'ko') # cm -> m
    ax2.plot(x, np.sqrt(cs2) / 1e5, 'ko') # cm -> km
    ax3.plot(x, rho0, 'ko')
    ax4.plot(x, P0 / 1e12, 'ko') # g/(cm.s^2) = 0.1 N / m^2 = 1e6 bar -> Mbar
    ax1.set_ylabel(r'$g$ [$\mathrm{m/s^2}$]')
    ax2.set_ylabel(r'$c_s$ [$\mathrm{km/s}$]')
    ax3.set_ylabel(r'$\rho_0$ [$\mathrm{g/cm^3}$]')
    ax4.set_ylabel(r'$P$ [Mbar]')
    ax3.set_xlabel(r'$r / R$')
    ax4.set_xlabel(r'$r / R$')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()

class PolytropeInterpolator(object):
    """
    Interpolates a core-envelope two-part polytrope
    """
    def __init__(self, xi_i, xi_e, sol_i, sol_e, R_i, R_e, M_i, M_e, R_c):
        '''
        R_i * xi_i / xi_i[-1] = r coordinates within the interior
        R_e * xi_e / xi_e[-1] = r coordinates within the env
        --> thus, R_e = actual radius

        R_c in same units as R_e
        '''
        super(PolytropeInterpolator, self).__init__()
        (
            self.xi_i, self.xi_e, self.sol_i, self.sol_e,
            self.R_i, self.R_e, self.M_i, self.M_e, self.R_c
        ) = xi_i, xi_e, sol_i, sol_e, R_i, R_e, M_i, M_e, R_c

    def get_i(self, r):
        # need to rescale quantities for correct jump conditions
        xi1_e = self.xi_e[-1]
        xi1_i = self.xi_i[-1]
        sol_i = self.sol_i(r / self.R_i * xi1_i)

        sol_i_c = self.sol_i(self.R_c / self.R_i * xi1_i)
        sol_e_c = self.sol_e(self.R_c / self.R_e * xi1_e)

        c_end_i = sol_i_c[2]
        c_end_e = sol_e_c[2]

        Vg, U, c1_base, As = sol_i[ :4]
        c_mult = c_end_e / c_end_i
        U_mult = (
            self.M_i / self.M_e * (self.R_e / self.R_i)**3
            * c_mult
        )
        Vg_mult = (1 / c_mult) / (self.M_i / self.M_e * (self.R_e / self.R_i)**3)
        As_mult = (1 / c_mult) / (self.M_i / self.M_e * (self.R_e / self.R_i)**3)
        return (Vg_mult * Vg, U_mult * U, c_mult * c1_base,
                As_mult * As, *sol_i[4: ])

    def get_e(self, r):
        return self.sol_e(r / self.R_e * self.xi_e[-1])

    def __call__(self, xi):
        if (xi > self.xi_e[-1]).any():
            raise ValueError('xi exceeds xi_1=%f' % self.xi_e[-1])
        r = xi / self.xi_e[-1] * self.R_e

        if not np.isscalar(xi):
            ret = np.zeros((6, len(xi)))
            int_idxs = np.where(r < self.R_c)[0]
            ret[:, int_idxs] = self.get_i(r[int_idxs])
            env_idxs = np.where(r >= self.R_c)[0]
            ret[:, env_idxs] = self.get_e(r[env_idxs])
            return ret

        if r < self.R_c:
            return self.get_i(r)
        else:
            return self.get_e(r)

def build_broken_polytrope(n_i=3, n_e=3, Rc_guess=0.5, rho_fact=2,
                           eps=EPS,
                           method='DOP853', npts=1000, atol=1e-9, rtol=1e-9,
                           **args):
    '''
    n_i, n_e = interior, env ptrope indicies
    Rc_guess = "guess" for the core radius. Specifically, the interior ends
        once the x = Rc_guess / R(interior model)
        * NB: this may not actually result in a core radius =
            Rc_guess, since the total interior + env radius may not be equal to
            that estimated by the interior alone
    rho_fact = rho_i / rho_e @ i-e boundary (C_rho in my notes)
    '''
    # calculate the full polytrope solutions for both indicies
    if n_i >= 5 or n_e >= 5:
        raise ValueError('Unbound star with n = %d' % n)

    Gamma1 = 5/3
    Gamma_i = 1 + 1 / n_i
    Gamma_e = 1 + 1 / n_e
    Vg0_i = (n_i + 1) * eps**2 / (3 * Gamma1)
    A0_i = (n_i + 1) * eps**2 / 3 * (1 / Gamma_i - 1 / Gamma1)
    U0 = 3
    c1_base0 = -3
    y0_i = [Vg0_i, U0, c1_base0, A0_i, 1, -eps / 3]

    Vg0_e = (n_e + 1) * eps**2 / (3 * Gamma1)
    A0_e = (n_e + 1) * eps**2 / 3 * (1 / Gamma_e - 1 / Gamma1)
    y0_e = [Vg0_e, U0, c1_base0, A0_e, 1, -eps / 3]

    def dqdx(x, y, n):
        q, qp = y[-2: ]
        qpp = -q**n - 2 * qp / x
        # Vg', U', c1_base', A'
        Gamma = 1 + 1 / n
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

    ret_i = solve_ivp(dqdx, (eps, np.inf), y0_i, args=[n_i],
                      events=[stop_cond], method=method, atol=atol, rtol=rtol,
                      dense_output=True, **args)
    ret_e = solve_ivp(dqdx, (eps, np.inf), y0_e, args=[n_e],
                      events=[stop_cond], method=method, atol=atol, rtol=rtol,
                      dense_output=True, **args)

    # stitch the polytropes together (assume M_i = R_i = 1)
    # TODO implement different n_i, n_e terms (etas in notes)
    xi1_i = ret_i.t[-1]
    xi1_e = ret_e.t[-1]
    qp1_i = ret_i.y[-1, -1]
    qp1_e = ret_e.y[-1, -1]
    def Re_cond(Re):
        dP = (
            ret_i.sol(xi1_i * Rc_guess)[-2]**(1 - n_i)
        ) - (
            Re**2 / rho_fact**2
            * ret_e.sol(xi1_e * Rc_guess / Re)[-2]**(1 - n_e)
        )
        return dP
    Re = opt.brenth(Re_cond, 1, 5 * rho_fact)
    Me = Re**3 / rho_fact * (
        ret_i.sol(xi1_i * Rc_guess)[-2]**(n_i)
    ) / (
        ret_e.sol(xi1_e * Rc_guess / Re)[-2]**(n_e)
    )
    Ri = 1
    Mi = 1

    interpolator = PolytropeInterpolator(
        ret_i.t, ret_e.t, ret_i.sol, ret_e.sol,
        Ri, Re, Mi, Me, Rc_guess)

    return ret_e.t, interpolator

if __name__ == '__main__':
    tol = 1e-10
    eps = 1e-7
    xmid = 0.3
    l = 1

    x, ptrope_sol = build_polytrope(3, atol=tol, rtol=tol, eps=eps)
    acc = opt_func(5, 20, l, ptrope_sol, x[-1], method='DOP853',
                   atol=tol, rtol=tol, eps=eps, xmid=xmid)
    print(1)
    print('\t', acc)

    for rho_fact in [1.1, 1.25, 1.28, 1.29]:
        x, broken_sol = build_broken_polytrope(
            3, rho_fact=rho_fact, Rc_guess=0.2, atol=tol, rtol=tol, eps=eps)
        print(rho_fact, 0.2, broken_sol.R_c / broken_sol.R_e)
        acc = opt_func(10.8, 11.5, l, broken_sol, x[-1], method='DOP853',
                       atol=tol, rtol=tol, eps=eps, xmid=xmid)
        print('\t', acc)

    x, broken_sol = build_broken_polytrope(
        3, rho_fact=1.295, Rc_guess=0.2, atol=tol, rtol=tol, eps=eps)
    print(1.295, 0.2, broken_sol.R_c / broken_sol.R_e)
    acc = opt_func(10.0, 10.8, l, broken_sol, x[-1], method='DOP853',
                   atol=tol, rtol=tol, eps=eps, xmid=xmid)
    print('\t', acc)

    x, broken_sol = build_broken_polytrope(
        3, rho_fact=1.298, Rc_guess=0.2, atol=tol, rtol=tol, eps=eps)
    print(1.298, 0.2, broken_sol.R_c / broken_sol.R_e)
    acc = opt_func(9.0, 9.1, l, broken_sol, x[-1], method='DOP853',
                   atol=tol, rtol=tol, eps=eps, xmid=xmid)
    print('\t', acc)

    x, broken_sol = build_broken_polytrope(
        3, rho_fact=1.25, Rc_guess=0.3, atol=tol, rtol=tol, eps=eps)
    print(1.25, 0.3, broken_sol.R_c / broken_sol.R_e)
    acc = opt_func(5, 20, l, broken_sol, x[-1], method='DOP853',
                   atol=tol, rtol=tol, eps=eps, xmid=xmid)
    print(acc)

    # print(broken_sol.R_c / broken_sol.R_e)
    # plot_polytrope(x, broken_sol, '/tmp/foobroken')
