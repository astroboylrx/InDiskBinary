import copy
import os
from pathlib import Path
from numbers import Number
import numpy as np
import numpy.ma as ma
import scipy as sp
import scipy.interpolate as spint
import scipy.signal as spsig
import matplotlib.pyplot as plt
import pyridoxine.utility as rxu
import pyridoxine.plt as rxplt
from multiprocessing import Pool
from multiprocess import Pool as mPool

# ******************************************************************************
# For reading data and all sorts of plotting
# ******************************************************************************

def get_b(i, data_dir, problem_id, multi=False, levels=None, **kwargs):
    ext = kwargs.get("ext", '.vtk'); wanted=kwargs.get('wanted', None)
    if ext[-3:] == "vtk":
        if multi is False:
            comb_dir = "/comb_dparvtk/" if ext == ".dpar.vtk" else "/comb/"
            comb_dir = kwargs.get('comb_dir', comb_dir)
            b = [rxu.AthenaVTK(data_dir+comb_dir+problem_id+".{:04d}".format(i)+ext, silent=True, wanted=wanted)]
            if levels is not None:
                for idx_lev in range(1, levels):
                    b.append(rxu.AthenaVTK(data_dir+comb_dir+problem_id+f"-lev{idx_lev}"+".{:04d}".format(i)+ext, silent=True, wanted=wanted))
        else:
            b = [rxu.AthenaMultiVTK(data_dir, problem_id, "{:04d}".format(i)+ext, silent=True, wanted=wanted)]
            if levels is not None:
                for idx_lev in range(1, levels):
                    b.append(rxu.AthenaMultiVTK(data_dir, problem_id, "{:04d}".format(i)+ext, silent=True, lev=idx_lev, wanted=wanted))
    return b

# rebin is faster than bin_ndarray since rebin only works for 2D ndarray
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def div0(a, b): # ref: https://stackoverflow.com/a/35696047/4009531
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def get_Ma(b, gamma):
    v = np.sqrt(b['ux']**2 + b['uy']**2) / b['rhog']
    cs = np.sqrt((b['E'] - 0.5 * (b['ux']**2 + b['uy']**2) / b['rhog']) * (gamma - 1) * gamma / b['rhog'])
    return  np.log10(v / cs)

def single_frame_rhop(b, figsize=(10, 8), **kwargs):
    fig, ax = plt.subplots(figsize=figsize); rhop = b['rhop']; 
    vmin = kwargs.get("vmin", -2); vmax = kwargs.get("vmax", 2); rhop[rhop<(10**vmin)] = 10**vmin
    xcorner = (b.box_min[0], b.box_max[0]); ycorner = (b.box_min[1], b.box_max[1])
    im = ax.pcolorfast(xcorner, ycorner, np.log10(rhop), vmin=vmin, vmax=vmax, cmap=kwargs.get("cmap", 'RdBu_r'))
    ax.set_title(r"t={:.2f}/$\Omega$".format(b.t)); ax.set_aspect(1.0); ax.set(**(kwargs.get("setax", {})))
    rxplt.add_aligned_colorbar(fig, ax, im); rxplt.ax_labeling(ax, x=r"x/H", y=r"z/H"); fig.tight_layout();
    return fig, ax

def single_frame_Q(b, figsize=(8.5, 7), **kwargs):
    fig, ax = plt.subplots(figsize=figsize); vmin = kwargs.get("vmin", -0.2); vmax = kwargs.get("vmax", 4.5); 
    fcb = kwargs.get("fcb", FixedCircularBinary(0.05)); Q=kwargs.get('Q', 'rhog'); logQ=kwargs.get('logQ', True)
    if logQ:
        scale_func = lambda x : np.log10(x)
    else:
        scale_func = lambda x : x
    for idx_lev, a in enumerate(b):
        im = ax.pcolor(a.ccx/fcb.a_b, a.ccy/fcb.a_b, scale_func(a[Q]), shading='auto', vmin=vmin, vmax=vmax, cmap=kwargs.get("cmap", 'RdBu_r'))
    ax.set(aspect=1.0, xlabel=r"$x/a_{\rm b}$", ylabel=r"$y/a_{\rm b}$", 
           title=kwargs.get("preTitle", '')+r"$t="+f"{a.t/fcb.P_b:.2f}"+r"P_{\rm bin}$")
    cax = rxplt.add_aligned_colorbar(fig, ax, im, size='3%'); cax.ax.set_ylabel(kwargs.get('cax_ylabel', "$\log_{10}(\Sigma/\Sigma_\infty)$"))
    fig.tight_layout();
    return fig, ax

def single_frame_zoomcenter_Q_no_fcb(b, figsize=(13.5, 6.5), **kwargs):
    fig, ax = plt.subplots(1, 2, figsize=figsize); #vmin = kwargs.get("vmin", -0.2); vmax = kwargs.get("vmax", 4.5); 
    Q=kwargs.get('Q', 'rhog'); logQ=kwargs.get('logQ', True); zl=kwargs.get('zl', 5)
    if logQ:
        scale_func = lambda x : np.log10(x[Q])
    else:
        scale_func = kwargs.get("scale_func", lambda x : x[Q])
    for idx_lev, a in enumerate(b):
        im = ax[0].imshow(scale_func(a), vmin=kwargs.get("vmin", scale_func(b[-1]).min()*1.1), vmax=kwargs.get("vmax", scale_func(b[-1]).max()*0.95), cmap=kwargs.get("cmap", 'RdBu_r'), rasterized=kwargs.get("rasterized", None), interpolation='none', extent=[a.box_min[0], a.box_max[0], a.box_min[1], a.box_max[1]], origin='lower')
        im = ax[1].imshow(scale_func(a), vmin=kwargs.get("vmin", scale_func(b[-1]).min()*1.1), vmax=kwargs.get("vmax", scale_func(b[-1]).max()*0.95), cmap=kwargs.get("cmap", 'RdBu_r'), rasterized=kwargs.get("rasterized", None), interpolation='none', extent=[a.box_min[0], a.box_max[0], a.box_min[1], a.box_max[1]], origin='lower')
        #im = ax[0].pcolor(a.ccx, a.ccy, scale_func(a), shading='auto', vmin=kwargs.get("vmin", scale_func(b[-1]).min()*1.1), vmax=kwargs.get("vmax", scale_func(b[-1]).max()*0.95), cmap=kwargs.get("cmap", 'RdBu_r'), rasterized=kwargs.get("rasterized", None))
        #im = ax[1].pcolor(a.ccx, a.ccy, scale_func(a), shading='auto', vmin=kwargs.get("vmin", scale_func(b[-1]).min()*1.1), vmax=kwargs.get("vmax", scale_func(b[-1]).max()*0.95), cmap=kwargs.get("cmap", 'RdBu_r'), rasterized=kwargs.get("rasterized", None))
    ax[0].set(aspect=kwargs.get("aspect", 1), xlabel=r"$x$"+kwargs.get("len_units", ""), ylabel=r"$y$"+kwargs.get("len_units", ""), 
              title=kwargs.get("preTitle", '')+r"$t="+f"{a.t:.2f}$")
    ax[1].set(aspect=kwargs.get("zoom_aspect", 1), xlabel=r"$x$"+kwargs.get("len_units", ""), xlim=(-zl, zl), ylim=(-zl, zl), title=r"Zooming in center")
    cax = rxplt.add_aligned_colorbar(fig, ax[1], im, size='3%'); cax.ax.set_ylabel(kwargs.get('cax_ylabel', "$\log_{10}(\Sigma/\Sigma_\infty)$"))
    fig.tight_layout(); fig.subplots_adjust(wspace=kwargs.get("wspace", 0.1))
    return fig, ax

def single_frame_zoomcenter_Q_no_fcb_stream(b, figsize=(13.5, 6.5), **kwargs):
    fig, ax = single_frame_zoomcenter_Q_no_fcb(b, figsize=figsize, **kwargs)
    lev = 0
    ax[0].streamplot(b[lev].ccx, b[lev].ccy, (b[lev]['ux']/b[lev]['rhog']), (b[lev]['uy']/b[lev]['rhog']), 
                     density=kwargs.get("stream_density", 1), color=kwargs.get('stream_color','lime'))
    ax[0].set_xlim([b[0].box_min[0], b[0].box_max[0]]); ax[0].set_ylim([b[0].box_min[1], b[0].box_max[1]]) # sometimes white space appears
    lev = kwargs.get("zoom_stream_level", -2)
    if kwargs.get('fcb', None) is not None:
        fcb = kwargs.get('fcb'); fcb.update_pos(b[lev].t)
        r_sink = kwargs.get('r_sink', 0.04)
        ux = b[lev]['ux']/b[lev]['rhog']; uy = b[lev]['uy']/b[lev]['rhog']
        ccx, ccy = np.meshgrid(b[lev].ccx, b[lev].ccy)
        dist2m1 = ((ccx - fcb.r1[0])**2 + (ccy - fcb.r1[1])**2)**(1/2)
        ux[dist2m1 < r_sink] = np.nan; uy[dist2m1 < r_sink] = np.nan
        dist2m2 = ((ccx - fcb.r2[0])**2 + (ccy - fcb.r2[1])**2)**(1/2)
        ux[dist2m2 < r_sink] = np.nan; uy[dist2m2 < r_sink] = np.nan
        ax[1].streamplot(b[lev].ccx, b[lev].ccy, ux, uy, density=kwargs.get("zoom_stream_density", 1.5), color=kwargs.get('stream_color','lime'))
    else:
        ax[1].streamplot(b[lev].ccx, b[lev].ccy, (b[lev]['ux']/b[lev]['rhog']), (b[lev]['uy']/b[lev]['rhog']), density=kwargs.get("zoom_stream_density", 1.5), color=kwargs.get('stream_color','lime'))
    return fig, ax

def single_frame_zoomcenter_Q_no_fcb_quiver(b, figsize=(13.5, 6.5), **kwargs):
    fig, ax = single_frame_zoomcenter_Q_no_fcb(b, figsize=figsize, **kwargs)
    sampling = kwargs.get("sampling", 16); b0 = b[0]
    Q = ax[0].quiver((b0.ccx[::sampling]+b0.dx[0]*(sampling//2-0.5)), (b0.ccy[::sampling]+b0.dx[1]*(sampling//2-0.5)), 
                     rebin(div0(b0['ux'], b0['rhog']), (b0.Nx[1]//sampling, b0.Nx[0]//sampling)), rebin(div0(b0['uy'], b0['rhog']), (b0.Nx[1]//sampling, b0.Nx[0]//sampling)),
                     angles='xy', scale_units='xy', scale=None, color='y')
    ax[0].quiverkey(Q, 0.05, 0.975, kwargs.get("len_quiverkey_1", 1), r'$'+str(kwargs.get("len_quiverkey_1", 1))+r'c_{\rm s}$', labelpos='E', coordinates='figure')
    if kwargs.get("sonic_contour", False):
        Ma = get_Ma(b0, kwargs.get("Gamma", 3))
        ax[0].contour(b0.ccx, b0.ccy, Ma, levels=[0, ], colors=['k', ], alpha=0.5, linewidths=[2, ])
    bz = b[kwargs.get("zoom_quiver_level", -2)]; sampling = kwargs.get("zoom_sampling", 8);
    Q = ax[1].quiver((bz.ccx[::sampling]+bz.dx[0]*(sampling//2-0.5)), (bz.ccy[::sampling]+bz.dx[1]*(sampling//2-0.5)), 
                     rebin(div0(bz['ux'], bz['rhog']), (bz.Nx[1]//sampling, bz.Nx[0]//sampling)), rebin(div0(bz['uy'], bz['rhog']), (bz.Nx[1]//sampling, bz.Nx[0]//sampling)),
                     angles='xy', scale_units='xy', scale=None, color='y')
    ax[1].quiverkey(Q, 0.9, 0.975, kwargs.get("len_quiverkey_2", 5), r'$'+str(kwargs.get("len_quiverkey_2", 5))+r'c_{\rm s}$', labelpos='E', coordinates='figure')
    if kwargs.get("sonic_contour", False):
        Ma = get_Ma(bz, kwargs.get("Gamma", 3))
        ax[1].contour(bz.ccx, bz.ccy, Ma, levels=[0, ], colors=['k', ], alpha=0.5, linewidths=[2, ])
    return fig, ax

def single_frame_zoomcenter_Q(b, figsize=(13.5, 6.5), **kwargs):
    fig, ax = plt.subplots(1, 2, figsize=figsize); vmin = kwargs.get("vmin", -0.2); vmax = kwargs.get("vmax", 4.5); 
    fcb = kwargs.get("fcb", FixedCircularBinary(0.05)); Q=kwargs.get('Q', 'rhog'); logQ=kwargs.get('logQ', True); zl=kwargs.get('zl', 5)
    if logQ:
        scale_func = lambda x : np.log10(x[Q])
    else:
        scale_func = kwargs.get("scale_func", lambda x : x[Q])
    for idx_lev, a in enumerate(b):
        im = ax[0].pcolor(a.ccx/fcb.a_b, a.ccy/fcb.a_b, scale_func(a), shading='auto', vmin=vmin, vmax=vmax, cmap=kwargs.get("cmap", 'RdBu_r'))
        im = ax[1].pcolor(a.ccx/fcb.a_b, a.ccy/fcb.a_b, scale_func(a), shading='auto', vmin=vmin, vmax=vmax, cmap=kwargs.get("cmap", 'RdBu_r'))
    ax[0].set(aspect=1.0, xlabel=r"$x/a_{\rm b}$", ylabel=r"$y/a_{\rm b}$", title=kwargs.get("preTitle", '')+r"$t="+f"{a.t/fcb.P_b:.2f}"+r"P_{\rm bin}$")
    ax[1].set(aspect=1.0, xlabel=r"$x/a_{\rm b}$", xlim=(-zl, zl), ylim=(-zl, zl), title=r"Zooming in center")
    cax = rxplt.add_aligned_colorbar(fig, ax[1], im, size='3%'); cax.ax.set_ylabel(kwargs.get('cax_ylabel', "$\log_{10}(\Sigma/\Sigma_\infty)$"))
    fig.tight_layout(); fig.subplots_adjust(wspace=kwargs.get("wspace", 0.04))
    return fig, ax

def draw_one_snapshot(i, data_dir, problem_id, out_dir, draw, figsize=None, multi=False, levels=None, save=False, **kwargs):
    ext = kwargs.get("ext", '.vtk'); wanted=kwargs.get('wanted', None)
    if ext[-3:] == "vtk":
        if multi is False:
            comb_dir = "/comb_dparvtk/" if ext == ".dpar.vtk" else "/comb/"
            comb_dir = kwargs.get('comb_dir', comb_dir)
            b = [rxu.AthenaVTK(data_dir+comb_dir+problem_id+".{:04d}".format(i)+ext, silent=True, wanted=wanted)]
            if levels is not None:
                for idx_lev in range(1, levels):
                    b.append(rxu.AthenaVTK(data_dir+comb_dir+problem_id+f"-lev{idx_lev}"+".{:04d}".format(i)+ext, silent=True, wanted=wanted))
        else:
            b = [rxu.AthenaMultiVTK(data_dir, problem_id, "{:04d}".format(i)+ext, silent=True, wanted=wanted)]
            if levels is not None:
                for idx_lev in range(1, levels):
                    b.append(rxu.AthenaMultiVTK(data_dir, problem_id, "{:04d}".format(i)+ext, silent=True, lev=idx_lev, wanted=wanted))
    elif ext[-3:] == "lis":
        if multi is False:
            b = rxu.AthenaLIS(data_dir+"/comb/"+problem_id+".{:04d}".format(i)+ext, silent=True);
        else:
            b = rxu.AthenaMultiLIS(data_dir, problem_id, "{:04d}".format(i)+ext, silent=True);
    if figsize is not None: kwargs['figsize'] = figsize
    fig, ax = draw(b, **kwargs)
    if save is True:
        fig.savefig(out_dir+"/{:04d}.png".format(i), bbox_inches='tight', dpi=kwargs.get('dpi', 250), transparent=False, facecolor=(1,1,1,1))
        plt.close()
    elif save is False:
        plt.show()
        return fig, ax
    else:
        return fig, ax

# ******************************************************************************
# Calculating torques and drag forces from gas to accretor(s)
# ******************************************************************************

def chunk_mean(arr, n):
    
    N = arr.size
    rN = N % n
    if rN == 0:  # [:-0] gives empty slice
        tmp_mean = arr.reshape([-1, n]).mean(axis=1)
    else:
        tmp_mean = arr[:-rN].reshape([-1, n]).mean(axis=1)
    return np.hstack([tmp_mean, arr[-rN:].mean()])

def every_chunk_mean(arr, n):
    
    if n == 1:
        return arr
    
    return np.array([arr[i:i+n].mean() for i in range(len(arr)-n+1)])

def movingaverage(values, window):
    """ useful if the time interval is fixed """

    weigths = np.repeat(1.0, window) / window
    #smas = np.convolve(values, weigths, 'valid') # sp.signal use fft, much faster
    smas = spsig.convolve(values, weigths, mode='valid')
    return smas # as a numpy array


def get_closest(array, values):
    """ fast indices locating; values should be sorted """

    #make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1

    return idxs

def _rollingaverage(t, dt, q, window, step):
    """ rolling average based on time data as well """

    t_series = np.linspace(t[0] + window/2, t[-1] - window/2, int((t[-1]-t[0]-window)/step))
    t_s_i = t_series - window/2; t_s_e = t_series + window/2
    idx_i = get_closest(t, t_s_i); idx_e = get_closest(t, t_s_e)
    # zip object cannot be re-used
    dt_sum = [dt[ii:ie+1].sum() for ii, ie in zip(idx_i, idx_e)]
    ra_t = [(t[ii:ie+1] * dt[ii:ie+1]).sum() / dt_sum[j] for j, (ii, ie) in enumerate(zip(idx_i, idx_e))]
    ra_q = [(q[ii:ie+1] * dt[ii:ie+1]).sum() / dt_sum[j] for j, (ii, ie) in enumerate(zip(idx_i, idx_e))]
    return ra_t, ra_q

def rollingaverage(t, dt, q, window, step, num_pool=-1):
    """ rolling average but only return the quantity w/o time """

    t_series = np.linspace(t[0] + window/2, t[-1] - window/2, int((t[-1]-t[0]-window)/step))
    t_s_i = t_series - window/2; t_s_e = t_series + window/2
    idx_i = get_closest(t, t_s_i); idx_e = get_closest(t, t_s_e)
    # zip object cannot be re-used
    #dt_sum = [dt[ii:ie+1].sum() for ii, ie in zip(idx_i, idx_e)]
    if num_pool <= 1:
        ra_q = np.array([(q[ii:ie+1] * dt[ii:ie+1]).sum() / dt[ii:ie+1].sum() for ii, ie in zip(idx_i, idx_e)])
    else:
        ave = lambda idx : (q[idx[0]:idx[1]+1] * dt[idx[0]:idx[1]+1]).sum() / dt[idx[0]:idx[1]+1].sum()
        p = mPool(num_pool)
        ra_q = np.array(p.map(ave, zip(idx_i, idx_e)))
        p.close()
        p.join()
    return ra_q

def FE(E, e, M):
    return E-e*np.sin(E)-M
def FEprime(E, e):
    return 1-e*np.cos(E)
def E0(M, e):
    return M+0.85*e*np.sign(np.sin(M))
def M2E(M, e, epsilon):
    negative_flag = False    
    if M < 0:
        negative_flag = True
        M = -M
    offset = 0
    if M > 2*np.pi:
        offset = M
        M = np.fmod(M, 2*np.pi)
        offset -= M
    tempE = E0(M, e)
    iter_count = 0
    while np.abs(FE(tempE, e, M)) > epsilon:
        newE = tempE - FE(tempE, e, M)/FEprime(tempE, e)
        tempE = newE
        iter_count += 1
    n_iters = iter_count
    E = tempE
    err = FE(E, e, M)
    E += offset
    if negative_flag:
        E = -E
    return E#, err, n_iters

"""
 # Pool from pathos.multiprcess complains about "np not defined" if used in class-method
 # However, the following works outside a class.
import multiprocess
M = np.linspace(0, 2*np.pi*3.5, int(512*3.5))
p = multiprocess.Pool(4)
newE = np.array(p.map(lambda m: M2E(m, 0.5, 1e-15), M))
p.close()
p.join()
"""

class FixedCircularBinary:
    """ Describe a fixed circular binary pair in a shearing box """
    
    def __init__(self, a_b = 1.0, lam = 10, Omega_K = 1.0, **kwargs):
        """ calculate binary parameters from a_b """
        
        # lam is short for lambda (= R_H / a)
        # in binary-focused units, we have a_b = 1.0 and M_b = 1.0, so Omega_b = 1.0

        self.a_b, self.M_b, self.Omega_b = a_b, 1.0, 1.0
        self.lam, self.Omega_K = lam, Omega_K
        if 'M_b' in kwargs:
            self.M_b = kwargs.get('M_b', self.M_b) # a chance to modify it
            self.Omega_b = np.sqrt(self.M_b / self.a_b**3)
        self.v_orb = self.Omega_b * self.a_b
        self.P_b = 2*np.pi / self.Omega_b
        self.R_H = self.a_b * self.lam
        
        # the input parameter must meet:
        #self.M_b = self.R_H**3 * 3 / self.Omega_K**2
        if abs(self.M_b - self.R_H**3 * 3 * self.Omega_K**2)/self.M_b >= 1e-15:
            raise ValueError("Parameters are not consistent: M_b=", self.M_b, ", lam=",
                             self.lam, ", Omega_K=", self.Omega_K)
        
        # 1:prograde/counter-clockwise; -1:retrograde/clockwise;
        self.orb_dir = kwargs.get('orb_dir', 1)
        
        # now correct for the shearing box frame
        # N.B.: Since we separate orb_dir from Omega_b, here we treat Omega_b as +1 and flip Omega_K.
        #       However, this is only used to calculate |orbital frequency|.
        #       Thus, the positions and velocities of the binary are in the frame of Omega_K being positive.
        #       Consequently, the calculations of inertial velocities still holds.
        self.Omega_bbox = self.Omega_b - self.orb_dir * self.Omega_K
        if kwargs.get("wrong_frame", False):
            self.Omega_bbox = self.Omega_b # for simulations in wrong frames
        if "Omega_offset" in kwargs:
            self.Omega_offset = kwargs.get("Omega_offset")
            self.Omega_bbox = self.Omega_bbox + self.Omega_offset
        else:
            self.Omega_offset = 0
        self.P_bbox = 2*np.pi / self.Omega_bbox # P_bbox is what matters for calculating r and v
        self.v_orbbox = self.Omega_bbox * self.a_b # equiv to 2*np.pi * self.a_b / self.P_bbox
        
        if 'M1' in kwargs and 'M2' in kwargs:
            self.M1, self.M2 = kwargs.get('M1'), kwargs.get('M2')
            self.M1, self.M2 = np.array([self.M1, self.M2]) * self.M_b/(self.M1+self.M2)
            self.q_b = self.M1 / self.M2 if self.M1<self.M2 else self.M2 / self.M1
        elif 'q_b' in kwargs: # note that I used M2 as the primary
            self.q_b = kwargs.get('q_b')
            self.M1, self.M2 = self.q_b, 1.0
            self.M1, self.M2 = np.array([self.M1, self.M2]) * self.M_b/(self.M1+self.M2)
        else:
            self.M1 = self.M_b / 2
            self.M2 = self.M_b / 2
            self.q_b = 1.0
        
        self.e0_prec = kwargs.get('e0_prec', False) # use this flag to consider precession even if e=0
        self.e = kwargs.get('ecc', 0.0) # eccentricity
        if self.e != 0.0 and self.e0_prec is True:
            self.e0_prec = False # ensure the consistency between self.e and self.e0_prec
            print(f"Warning: e0_prec should only be True for eccentric orbits (given e={self.e:.2e}) and has been reset to False.")
        self.dot_curly_pi = 0.75*(1/3/(self.lam)**3)*np.sqrt(1-self.e**2)
        self.Omega_b_prime = self.Omega_b + self.dot_curly_pi  - self.orb_dir * self.Omega_K
        self.P_b_prime = 2 * np.pi / (self.Omega_b + self.dot_curly_pi  - self.orb_dir * self.Omega_K)
        self.num_pool = kwargs.get('Npool', 0)
        
        self.mu_b = self.M1 * self.M2 / self.M_b
        self.l_0 = self.orb_dir * self.Omega_b * self.a_b**2 * np.sqrt(1 - self.e**2)
        # alternatively, self.l_0 = self.orb_dir * np.sqrt(self.M_b * self.a_b * (1 - self.e**2))
        self.L_0 = self.mu_b * self.l_0
        self.eth_0 = -self.M_b / (2*self.a_b)
        # alternatively, self.eth_0 = - self.Omega_b**2 * self.a_b**2
        self.E_0 = self.mu_b * self.eth_0

        self.debug_flag = kwargs.get("debug", False)
        if self.debug_flag:
            if self.e != 0.0 and self.e0_prec is True:
                self.l_0 = self.orb_dir * (self.Omega_b + self.Omega_offset) * self.a_b**2 * np.sqrt(1 - self.e**2)
                self.eth_0 = -(self.Omega_b**2 * self.a_b**2) + 0.5*(self.Omega_b + self.Omega_offset)**2 * self.a_b**2
            else:
                self.l_0 = self.orb_dir * (self.Omega_b + self.dot_curly_pi) * self.a_b**2 * np.sqrt(1 - self.e**2)
                self.eth_0 = -(self.Omega_b**2 * self.a_b**2) + 0.5*(self.Omega_b + self.dot_curly_pi)**2 * self.a_b**2
            self.L_0 = self.mu_b * self.l_0            
            self.E_0 = self.mu_b * self.eth_0
        
        # initial position and velocity

        # here, r_b0 and v_b0 are vectors from m1 to m2, which follows SSD's definition and differs from MML19
        # example vector from m1 to m2: r12 = r2 - r1
        # example vector from m2 to m1: r21 = r1 - r2
        # by default, the initial positions of m1 and m2 are on the x-axis and m1 is on the negative side
        self.r_b0 = np.array([self.a_b * (1 - self.e), 0])
        self.v_b0 = np.array([0, self.orb_dir * self.Omega_b * self.a_b * np.sqrt((1 + self.e) / (1 - self.e))])
        
        self.update_pos(0)
        
        # time-averaged strings
        self.ta_str = {
            'dot_L' : self.ta_it(r"\dot{L}_{\rm b}"),
            'dot_m' : self.ta_it(r"\dot{m}_{\rm b}"),
            'dot_l' : self.ta_it(r"\dot{\ell}_{\rm b}"),
            'dot_eth' : self.ta_it(r"\dot{\mathcal{E}}_{\rm b}"),
            'dot_a' : self.ta_it(r"\dot{a}_{\rm b}"),
            'dot_e2' : self.ta_it(r"\dot{e}_{\rm b}^2"),
            'dot_e' : self.ta_it(r"\dot{e}_{\rm b}"),
            'L/m' : self.ta_it(r"\dot{L}_{\rm b}")+r"$/$"+self.ta_it(r"\dot{m}_{\rm b}"),
            'l0' : "$\ell_0$",
            'l0c' : "$\ell_{0,\mathrm{crit}}$",
            'aoa' : self.ta_it(r"\dot{a}_{\rm b}")+r"$/a_{\rm b}$",
            'mom' : self.ta_it(r"\dot{m}_{\rm b}")+r"$/m_{\rm b}$",
        }
        # unit strings
        self.u_str = {'dot_L' : r" [$\Sigma_\infty v_{\rm b}^2 a_{\rm b}^2$]", 
                      'dot_m' : r" [$\Sigma_{\infty} v_{\rm b} a_{\rm b}$]",
                      'dot_l' : r" [$\Sigma_\infty v_{\rm b}^2 a_{\rm b}^2 / m_{\rm b}$]",
                      'dot_eth' : r" [$\Sigma_\infty v_{\rm b}^3 a_{\rm b} / m_{\rm b}$]",
                      'l0' : r" [$v_{\rm b} a_{\rm b}$]",
                      'mom' : r" [" + self.ta_str['mom'] + r"]",
        }
        self.unit_str = self.u_str
        # combined strings
        self.cb_str = {
            'dot_m' : self.ta_str['dot_m'] + self.u_str['dot_m'],
            'dot_L' : self.ta_str['dot_L'] + self.u_str['dot_L'],
            'dot_l' : self.ta_str['dot_l'] + self.u_str['dot_l'],
            'dot_eth' : self.ta_str['dot_eth'] + self.u_str['dot_eth'],
            'L/m' : self.ta_str['L/m'] + self.u_str['l0'],
            'l0' : self.ta_str['l0'] + self.u_str['l0'],
            'l0c' : self.ta_str['l0c'] + self.u_str['l0'],
            'aoa' : self.ta_str['aoa'] + self.u_str['mom'],
            'dot_e2' : self.ta_str['dot_e2'] + self.u_str['mom'],
            'dot_e' : self.ta_str['dot_e'] + self.u_str['mom'],
        }

    def ta_it(self, _s):
        return r"$\langle" + _s + r"\rangle$"
        
    def update_pos(self, t):
        """ update binary's positions and velocities based on the input time (one single time)
            N.B.: v1/v2 for circular binaries w/o precession are velocities in the rotating frame
                  v1/v2 for binaries w/ precession need extra -(Omega_pre x r1/r2) to be in the rotating frame
        """

        _r1, _r2, _v1, _v2 = self.get_pos(np.atleast_1d(t))
        self.r1, self.r2, self.v1, self.v2 = _r1[0], _r2[0], _v1.T[0], _v2.T[0]
        self.rb = self.r1 - self.r2
        self.vb = self.v1 - self.v2
    
    def FE(self, E, e, M):
        return E-e*np.sin(E)-M
    def FEprime(self, E, e):
        return 1-e*np.cos(E)
    def E0(self, M, e):
        return M+0.85*e*np.sign(np.sin(M))
    def M2E(self, M, e, epsilon):
        negative_flag = False    
        if M < 0:
            negative_flag = True
            M = -M
        offset = 0
        if M > 2*np.pi:
            offset = M
            M = np.fmod(M, 2*np.pi)
            offset -= M
        tempE = self.E0(M, e)
        iter_count = 0
        while np.abs(self.FE(tempE, e, M)) > epsilon:
            newE = tempE - self.FE(tempE, e, M)/self.FEprime(tempE, e)
            tempE = newE
            iter_count += 1
        n_iters = iter_count
        E = tempE
        err = self.FE(E, e, M)
        E += offset
        if negative_flag:
            E = -E
        return E#, err, n_iters
    
    def _M2E(self, M):
        return self.M2E(M, self.e, 1e-15)
    
    def get_pos(self, t):
        """ get binary's positions and velocities based on the input time array and return r_1/2 & v_1/2
            N.B.: v1/v2 for circular binaries w/o precession are velocities in the rotating frame
                  v1/v2 for binaries w/ precession need extra -(Omega_pre x r1/r2) to be in the rotating frame
        """

        if self.e == 0.0 and self.e0_prec is False:
            # phase contains orb_dir, no need to include it elsewhere
            phase = self.orb_dir * 2*np.pi * t / self.P_bbox
            # transpose to be crossed with f_grav
            r_1 = np.array([np.cos(-np.pi   + phase), np.sin(-np.pi   + phase)]).T * self.a_b * self.M2/self.M_b
            r_2 = np.array([np.cos(     0   + phase), np.sin(     0   + phase)]).T * self.a_b * self.M1/self.M_b
            # no transpose for v so rb x vb can be done easier
            v_1 = np.array([np.cos(-np.pi/2*self.orb_dir + phase), 
                            np.sin(-np.pi/2*self.orb_dir + phase)]) * self.v_orbbox * self.M2/self.M_b
            v_2 = np.array([np.cos( np.pi/2*self.orb_dir + phase), 
                            np.sin( np.pi/2*self.orb_dir + phase)]) * self.v_orbbox * self.M1/self.M_b
        else:
            M = np.asarray(t)  # no orb_dir here b/c it already affects v_b0 || note M[0] may not be 0
            if self.num_pool == 0:
                E = np.array([self._M2E(m) for m in M])
            else:
                p = Pool(self.num_pool)
                E = np.array(p.map(self._M2E, M))
                p.close()
                p.join()
            f_func = 1/(1-self.e) * (np.cos(E) - 1) + 1
            g_func = M + (np.sin(E) - E)

            phase_K = (self.orb_dir * self.dot_curly_pi - self.Omega_K) * t
            # note that this r_b is r2 - r1 but we use r1 - r2 in other calculations
            r_b = np.array([f_func * np.cos(phase_K) * self.r_b0[0] + g_func * (-np.sin(phase_K)) * self.v_b0[1], 
                            f_func * np.sin(phase_K) * self.r_b0[0] + g_func * np.cos(phase_K) * self.v_b0[1]]).T
            # brute force it back to our definition of r1 and r2
            r_1 = -r_b * self.M2 / self.M_b
            r_2 = r_b * self.M1 / self.M_b

            r_mag = np.sqrt((r_b**2).sum(axis=1))
            fprime = -self.a_b/r_mag/(1-self.e) * (np.sin(E)) # N.B.: a_b^2 cancelled with r_b0=a_b(1-e) || E[0] is not necessary 0
            gprime = self.a_b/r_mag * (np.cos(E) - 1) + 1
            # we need v1 - v2, but this v_b is v2 - v1
            v_b = np.array([fprime * np.cos(phase_K) * self.r_b0[0] + gprime * (-np.sin(phase_K)) * self.v_b0[1],
                            fprime * np.sin(phase_K) * self.r_b0[0] + gprime * np.cos(phase_K) * self.v_b0[1]])
            # brute force it back to our definition of v1 and v2
            v_1 = -v_b * self.M2 / self.M_b
            v_2 = v_b * self.M1 / self.M_b
        
        return r_1, r_2, v_1, v_2

    def grav_pot(self, x, y, dr_soft=1e-6):
        
        phi = (-self.M1 / (np.sqrt((x-self.r1[0])**2 + (y-self.r1[1])**2) + dr_soft*self.a_b)
               -self.M2 / (np.sqrt((x-self.r2[0])**2 + (y-self.r2[1])**2) + dr_soft*self.a_b) )
        
        return phi


class InDiskBinaryParas:
    """ physical parameters for our binary in shearing box simulations """
    
    def __init__(self, q, h, lambda_prime, gamma=1.6, beta=1.0, silent=False):
        self.q, self.h, self.lambda_prime = q, h, lambda_prime
        self.gamma, self.beta = gamma, beta
        self.eta, self.sqrt_eta, self.lam = h**2, h, lambda_prime / 3**(1/3)
        
        # code units
        self.Sigma_g, self.a_b, self.M_b = 1.0, 1.0, 1.0
        self.R_H = self.lam * self.a_b
        self.Omega_K = np.sqrt(self.M_b / (3 * self.R_H**3))
        self.c_s = ((self.M_b * self.Omega_K * h**3) / q)**(1/3)
        self.R_B = self.M_b / self.c_s**2
        self.H_g = self.c_s / self.Omega_K
        self.P_g = (self.Sigma_g * self.c_s**2) / self.gamma
        self.dv_subK = self.beta * self.h * self.c_s
        self.v_0 = self.dv_subK
        self.v_s = 1.5 * self.Omega_K * self.a_b
        if not silent:
            print(f"{'Omega_K':>24s}{'c_s':>24s}{'P_g':>24s}{'v_0':>24s}{'v_s':>24s}{'R_H':>24s}\n"
                +f"{self.Omega_K:>24.16e}{self.c_s:>24.16e}{self.P_g:>24.16e}"
                +f"{self.v_0:>24.16e}{self.v_s:>24.16e}{self.R_H:>24.16e}")


class dv_grav:
    def __init__(self, filename, fcb, header=True, **kwargs):
        
        f = open(filename, 'rb')
        if header:
            tmp_line = f.readline() # currently, there should only be one number
            Nt = int(tmp_line)
        else:
            Nt = kwargs.get("Nt", 1)
        self.Nt = Nt
        data_per_row = 2 + 4 * 2 * Nt
        self.data_per_row = data_per_row
            
        pos_i = f.tell(); f.seek(0, 2); num_bytes = f.tell() - pos_i
        self.num_rows = num_bytes // 8 // data_per_row
        if self.num_rows * 8 * data_per_row > num_bytes:
            raise IOError(f"The number of bytes seems off: Nt={Nt}, rows={self.num_rows}, num_bytes={num_bytes}")
        elif self.num_rows * 8 * data_per_row < num_bytes:
            print(f"Bytes more than data: Nt={Nt}, rows={self.num_rows}, num_bytes={num_bytes}; reading anyway")
            self.num_rows -= 1 # something must be off in the end, let's be safe
        else:
            pass
        
        row_limit = kwargs.get("row_limit", None)
        if row_limit is None:
            f.close()
            return
        else:
            if self.num_rows >= row_limit: # good to check again
                self.num_rows = row_limit
            else:
                f.close()
                raise ValueError("row_limit larger than available num_rows")

        f.seek(pos_i, 0)
        self.data = rxu.loadbin(f, num=self.num_rows*data_per_row).reshape([self.num_rows, data_per_row]);
        f.close()
        
        self.t, self.dt = self.data[:, :2].T; self._2dt = self.dt[1:] + self.dt[:-1]
        self._dv_grav_1 = []; self._dv_grav_2 = []
        for t in range(Nt):
            self._dv_grav_1.append(np.insert(self.data[:, 2+t*8:4+t*8], np.arange(1, self.t.size+1), self.data[:, 6+t*8:8+t*8], axis=0))
            self._dv_grav_2.append(np.insert(self.data[:, 4+t*8:6+t*8], np.arange(1, self.t.size+1), self.data[:, 8+t*8:10+t*8], axis=0))
        
        # now for centered-difference f_grav_i
        # b/c when n = 1
        # <f_1> = (f_0 * dt_0 / 2 + f_1 * dt_0 / 2 + f_1 * dt_1 / 2 + f_2 * dt_1 / 2) / (dt_0 + dt_1)
        # thus, <f_n> = F(f_{n-1}, f_n, f_{n+1}, dt_{n-1}, dt_n)
        # we need a chunk of four dv_grav, and a chunk of two dt, with interval of two
        # if we further consider the t=0 point, then the starting cumsum-diff index would be 3
        Nf = 4; idx_diff = 3
        self.f_grav_1 = []; self.f_grav_2 = []
        for t in range(Nt):
            self.f_grav_1.append(
                (self._dv_grav_1[t].cumsum(axis=0)[idx_diff::2]
                 - np.vstack([[0, 0], self._dv_grav_1[t].cumsum(axis=0)[1:-Nf:2]])) / self._2dt[:, np.newaxis] )
            self.f_grav_2.append(
                (self._dv_grav_2[t].cumsum(axis=0)[idx_diff::2] 
                 - np.vstack([[0, 0], self._dv_grav_2[t].cumsum(axis=0)[1:-Nf:2]])) / self._2dt[:, np.newaxis] )


class dmdp_acc:
    
    def __init__(self, filename, fcb, header=True, **kwargs):

        f = open(filename, 'rb')
        if header:
            tmp_line = f.readline()
            num_r_ev = int(tmp_line)
        else:
            num_r_ev = kwargs.get("num_r_ev", 1)
        self.num_r_ev = num_r_ev
        num_col_set = (2 + 6 + 3 + 3) # time, dt, dm_acc[2], dp_acc[2][3], f_pres_1[3], f_pres_2[3]
        self.num_col_set = num_col_set
        self.data_per_row = 2 + num_col_set * num_r_ev

        pos_i = f.tell(); f.seek(0, 2); num_bytes = f.tell() - pos_i
        self.num_rows = num_bytes // 8 // self.data_per_row
        if self.num_rows * 8 * self.data_per_row > num_bytes:
            raise IOError(f"The number of bytes seems off: rows={self.num_rows}, num_bytes={num_bytes}")
        elif self.num_rows * 8 * self.data_per_row < num_bytes:
            print(f"Bytes more than data: rows={self.num_rows}, num_bytes={num_bytes}; reading anyway")
            self.num_rows -= 1 # something must be off in the end, let's be safe
        else:
            pass

        if kwargs.get("read_spin", False):
            self.spin_flag = True
            fs = open(os.path.dirname(filename) + '/' + kwargs.get("spin_filename", "dp_spin.dat"), 'rb');
            fs.seek(pos_i, 0)
            
            data_per_row_spin = 2 * num_r_ev
            fs.seek(0, 2); num_bytes = fs.tell() - pos_i
            num_rows_spin = num_bytes // 8 // data_per_row_spin
            if num_rows_spin * 8 * data_per_row_spin > num_bytes:
                raise IOError(f"The number of bytes seems off in spin data: rows={num_rows_spin}, num_bytes={num_bytes}")
            elif num_rows_spin * 8 * data_per_row_spin < num_bytes:
                print(f"Bytes more than data in spin data: rows={num_rows_spin}, num_bytes={num_bytes}; reading anyway")
                num_rows_spin -= 1 # something must be off in the end, let's be safe
                if num_rows_spin < self.num_rows:
                    self.num_rows = num_rows_spin
            else:
                pass
        else:
            self.spin_flag = False
        
        row_limit = kwargs.get("row_limit", None)
        if row_limit is None:
            f.close()
            if self.spin_flag:
                fs.close()
            return
        else:
            if self.num_rows >= row_limit: # good to check again
                self.num_rows = row_limit
            else:
                f.close()
                raise ValueError("row_limit larger than available num_rows")
        
        f.seek(pos_i, 0)
        self.data = rxu.loadbin(f, num=self.num_rows*self.data_per_row).reshape([self.num_rows, self.data_per_row]);
        f.close()
        if kwargs.get("read_spin", False):
            self.spin_flag = True
            f = open(os.path.dirname(filename) + '/' + kwargs.get("spin_filename", "dp_spin.dat"), 'rb');
            f.seek(pos_i, 0)
            self.spin_data = rxu.loadbin(f, num=self.num_rows * 2 * num_r_ev).reshape([self.num_rows, 2 * num_r_ev]);
            f.close()
            self.Sdot = []
        else:
            self.spin_flag = False

        self.t, self.dt = self.data[:, :2].T
        self.mdot = []; self.mdot_tot = []; self.pdot = []; self.Fpres = []; 
        for idx_r in range(num_r_ev):
            self.mdot.append(self.data[:, 2+idx_r*num_col_set:4+idx_r*num_col_set].T)
            self.mdot_tot.append(self.mdot[-1][0] + self.mdot[-1][1])
            self.pdot.append(self.data[:, 4+idx_r*num_col_set:10+idx_r*num_col_set].T.reshape([2, 3, self.num_rows]))
            self.Fpres.append(self.data[:, 10+idx_r*num_col_set:16+idx_r*num_col_set].T.reshape([2, 3, self.num_rows]))
            if self.spin_flag:
                self.Sdot.append(self.spin_data[:, idx_r*2:2+idx_r*2].T)


class dmdp_acc_Bondi_SB:
    
    def __init__(self, filename, fcb, header=True, **kwargs):

        f = open(filename, 'rb')
        if header:
            tmp_line = f.readline()
            num_r_ev = int(tmp_line)
        else:
            num_r_ev = kwargs.get("num_r_ev", 1)
        self.num_r_ev = num_r_ev
        num_col_set = (2 + 6 + 3 + 3) # time, dt, dm_acc[2], dp_acc[2][3], f_pres_1[3], f_pres_2[3]
        self.num_col_set = num_col_set
        self.data_per_row = 2 + num_col_set * num_r_ev

        pos_i = f.tell(); f.seek(0, 2); num_bytes = f.tell() - pos_i
        self.num_rows = num_bytes // 8 // self.data_per_row
        if self.num_rows * 8 * self.data_per_row > num_bytes:
            raise IOError(f"The number of bytes seems off: rows={self.num_rows}, num_bytes={self.num_bytes}")
        elif self.num_rows * 8 * self.data_per_row < num_bytes:
            print(f"Bytes more than data: rows={self.num_rows}, num_bytes={num_bytes}; reading anyway")
            self.num_rows -= 1 # something must be off in the end, let's be safe
        else:
            pass
        
        row_limit = kwargs.get("row_limit", None)
        if row_limit is None:
            f.close()
            return
        else:
            if self.num_rows >= row_limit: # good to check again
                self.num_rows = row_limit
            else:
                f.close()
                raise ValueError("row_limit larger than available num_rows")
        
        f.seek(pos_i, 0)
        data = rxu.loadbin(f, num=self.num_rows*self.data_per_row).reshape([self.num_rows, self.data_per_row]);
        f.close()

        self.t, self.dt = data[:, :2].T
        self.mdot = []
        for idx_r in range(num_r_ev):
            self.mdot.append(data[:, 2+idx_r*num_col_set:3+idx_r*num_col_set].T)


class dot_L:
    
    def __init__(self, q, h, lambda_prime, data_dir, r_ev, **kwargs):
        
        self.r_ev = r_ev
        self.paras = InDiskBinaryParas(q, h, lambda_prime, **kwargs.get("paras_kw", {}))

        # this fcb only handles cases where a = M_b = 1.0
        self.fcb = FixedCircularBinary(lam = self.paras.lam, Omega_K = self.paras.Omega_K, **kwargs.get("fcb_kw", {}))
        self.fcb = kwargs.get("fcb", self.fcb) # so one may overwrite fcb
        if not kwargs.get('prefix', False):
            data_dir = data_dir + "/"
        
        max_row_dv = dv_grav(data_dir + "dv_grav.dat", self.fcb, **kwargs.get("dv_kw", {})).num_rows
        self.dmdp_acc_exist = Path(data_dir + "dmdp_acc.dat").is_file()
        # in case the simulation is ongoing and two data files have different cycles
        if self.dmdp_acc_exist:
            max_row_dmdp = dmdp_acc(data_dir + "dmdp_acc.dat", self.fcb, **kwargs.get("dmdp_kw", {})).num_rows
            if max_row_dmdp == max_row_dv:
                max_row = max_row_dmdp
            else:
                max_row = min(max_row_dmdp, max_row_dv) - 1
                print("using the most compatible max_row = ", max_row)
        else:
            max_row = max_row_dv
            print("dmdp_acc.dat not exist; using the most compatible max_row_dv = ", max_row)

        self.dv = dv_grav(data_dir + "dv_grav.dat", self.fcb, row_limit=max_row, dot_L_take_over=True, **kwargs.get("dv_kw", {}))
        if self.dmdp_acc_exist:
            self.dmdp = dmdp_acc(data_dir + "dmdp_acc.dat", self.fcb, row_limit=max_row, dot_L_take_over=True,  **kwargs.get("dmdp_kw", {}))
            self.spin_flag = self.dmdp.spin_flag
            if self.dmdp.num_r_ev != self.dv.Nt:
                raise ValueError("self.dmdp.num_r_ev != self.dv.Nt")
        else:
            self.spin_flag = False

        # Below we assume the system is 2D and calculate binary's positions/velocities and their inertial velocities
        self.r1, self.r2, self.v1, self.v2 = self.fcb.get_pos(self.dv.t[1:])
        self.rb = self.r1 - self.r2
        if self.fcb.e == 0 and self.fcb.e0_prec is False:
            self.vb = self.v1 - self.v2  # we don't really use this quantity
            # the cross product here assumes that the z-component of r1/2 is 0
            if not self.fcb.debug_flag:
                self.v1_inertial = self.v1 + np.cross(np.array([0, 0, self.fcb.Omega_K - self.fcb.orb_dir * self.fcb.Omega_offset]), self.r1)[:, :2].T
                self.v2_inertial = self.v2 + np.cross(np.array([0, 0, self.fcb.Omega_K - self.fcb.orb_dir * self.fcb.Omega_offset]), self.r2)[:, :2].T
            else:
                self.v1_inertial = self.v1 + np.cross(np.array([0, 0, self.fcb.Omega_K]), self.r1)[:, :2].T
                self.v2_inertial = self.v2 + np.cross(np.array([0, 0, self.fcb.Omega_K]), self.r2)[:, :2].T
        else:
            # we calculate the velocity based on the perspective of the inertial frame
            self.v1_inertial = self.v1
            self.v2_inertial = self.v2
            # thus, we need to put it back to the rotating frame -- this has been validated via trajectories
            self.v1 = self.v1_inertial - np.cross(np.array([0, 0, self.fcb.Omega_K - self.fcb.orb_dir * self.fcb.dot_curly_pi]), self.r1)[:, :2].T
            self.v2 = self.v2_inertial - np.cross(np.array([0, 0, self.fcb.Omega_K - self.fcb.orb_dir * self.fcb.dot_curly_pi]), self.r2)[:, :2].T
            self.vb = self.v1 - self.v2  # we don't really use this quantity
            # what if we need to account for the physical precession? not sure for now
            if self.fcb.debug_flag:
                self.v1_inertial -= np.cross(np.array([0, 0, - self.fcb.dot_curly_pi]), self.r1)[:, :2].T
                self.v2_inertial -= np.cross(np.array([0, 0, - self.fcb.dot_curly_pi]), self.r2)[:, :2].T

        """ From Dong's discussions: r_b is the same, regardless of the reference frame, but x-y decomposition is different in 
            different frames. On the contrary, v_b is different, but can be connected to do inertial frame calculations:
            v_b = v_b' + Omega_K x r_b'
            which is very different than the direct computation from phase: v_b(t) = Omega_b x r_b(t)
        """
        self.vb_inertial = self.v1_inertial - self.v2_inertial
        # alternatively, this can be done through 
        #   self.vb + np.cross(np.array([0, 0, Omega_precession]), self.rb)[:, :2].T
        # which may differ on the order of machine precision
        
        if self.dmdp_acc_exist:
            # only calculate quantities for t[1:] to match data in self.dv_grav
            self.facc1 = []; self.facc2 = []; self.fpres1 = []; self.fpres2 = []
            for idx_r in range(self.dmdp.num_r_ev):
                self.facc1.append((self.dmdp.pdot[idx_r][0][:2, 1:] - self.dmdp.mdot[idx_r][0][1:] * self.v1) / self.fcb.M1)
                self.facc2.append((self.dmdp.pdot[idx_r][1][:2, 1:] - self.dmdp.mdot[idx_r][1][1:] * self.v2) / self.fcb.M2)
                self.fpres1.append(self.dmdp.Fpres[idx_r][0][:2, 1:] / self.fcb.M1)
                self.fpres2.append(self.dmdp.Fpres[idx_r][1][:2, 1:] / self.fcb.M2)
            # specific torques
            self.facc_1m2 = []; self.fpres_1m2 = [];
            self.dot_l_acc = []; self.dot_l_pres = []; self.dot_L_acc = []; self.dot_L_pres = []
            self.dot_eth_acc = []; self.dot_eth_pres = []
            for idx_r in range(self.dmdp.num_r_ev):
                self.facc_1m2.append(self.facc1[idx_r] - self.facc2[idx_r])
                self.fpres_1m2.append(self.fpres1[idx_r] - self.fpres2[idx_r])
                self.dot_l_acc.append(np.cross(self.rb, (self.facc_1m2[-1]).T))
                self.dot_l_pres.append(np.cross(self.rb, (self.fpres_1m2[-1]).T))
                self.dot_L_acc.append(self.fcb.mu_b * self.dot_l_acc[-1])
                self.dot_L_pres.append(self.fcb.mu_b * self.dot_l_pres[-1])
                self.dot_eth_acc.append(np.sum(self.vb_inertial * self.facc_1m2[-1], axis=0))
                self.dot_eth_pres.append(np.sum(self.vb_inertial * self.fpres_1m2[-1], axis=0))

        # np.cross treat the last axis as the vector axis (so r1/r2 is transposed)
        self.fgrav_1m2 = []
        self.dot_l_grav = []; self.dot_L_grav = [];
        self.dot_eth_grav = []
        for t in range(self.dv.Nt):
            self.fgrav_1m2.append(self.dv.f_grav_1[t] - self.dv.f_grav_2[t])
            self.dot_l_grav.append(np.cross(self.rb, self.fgrav_1m2[-1]))
            self.dot_L_grav.append(self.fcb.mu_b * self.dot_l_grav[-1])
            self.dot_eth_grav.append(np.sum(self.vb_inertial * (self.fgrav_1m2[-1]).T, axis=0))

        # now all other terms toward total quantities
        self.dot_L_mu_b = []; 
        self.dot_l_tot, self.dot_L_tot, self.dot_eth = [], [], []
        if self.dmdp_acc_exist:
            for idx in range(self.dmdp.num_r_ev):
                self.dot_L_mu_b.append(self.dmdp.mdot[idx][0][1:] * np.cross(self.r1, self.v1_inertial.T)
                                       + self.dmdp.mdot[idx][1][1:] * np.cross(self.r2, self.v2_inertial.T))
                self.dot_l_tot.append(self.dot_l_grav[idx] + self.dot_l_acc[idx] + self.dot_l_pres[idx])
                self.dot_L_tot.append(self.dot_L_grav[idx] + self.dot_L_acc[idx] + self.dot_L_pres[idx] + self.dot_L_mu_b[-1])
                self.dot_eth.append(- self.dmdp.mdot_tot[idx][1:] / self.fcb.a_b 
                                    + self.dot_eth_grav[idx] + self.dot_eth_acc[idx] + self.dot_eth_pres[idx])
        else:
            for idx in range(self.dv.Nt):
                self.dot_L_mu_b.append(np.zeros_like(self.v1_inertial[0]))
                self.dot_l_tot.append(self.dot_l_grav[idx])
                self.dot_L_tot.append(self.dot_L_grav[idx])
                self.dot_eth.append(self.dot_eth_grav[idx])
        
        self.adot_oa_eth, self.adot_oa_l, self.adot_oa_L = [], [], []
        self.e2dot = []
        if self.dmdp_acc_exist:
            for idx_r in range(self.dmdp.num_r_ev):
                self.adot_oa_eth.append(self.dmdp.mdot_tot[idx_r][1:] / self.fcb.M_b - (self.dot_eth[idx_r] / self.fcb.eth_0))
                # calculate everything when e0_prec is True
                if self.fcb.e > 0.0 or self.fcb.e0_prec is True:
                    self.e2dot.append((2 * self.dmdp.mdot_tot[idx_r][1:] / self.fcb.M_b - (self.dot_eth[idx_r] / self.fcb.eth_0)
                                       - 2 * (self.dot_l_tot[idx_r] / self.fcb.l_0)) * (1 - self.fcb.e**2))
                    if self.fcb.e > 0.0:
                        self.adot_oa_l.append(None); self.adot_oa_L.append(None)
                if self.fcb.e == 0.0:
                    self.adot_oa_l.append(2 * self.dot_l_tot[idx_r] / self.fcb.l_0 - self.dmdp.mdot_tot[idx_r][1:] / self.fcb.M_b)
                    if self.fcb.q_b == 1.0:
                        self.adot_oa_L.append(2 * self.dot_L_tot[idx_r] / self.fcb.L_0 - 3 * self.dmdp.mdot_tot[idx_r][1:] / self.fcb.M_b)
                    else:
                        self.adot_oa_L.append(2 * self.dot_L_tot[idx_r] / self.fcb.L_0 + self.dmdp.mdot_tot[idx_r][1:] / self.fcb.M_b
                                              - 2 * self.dmdp.mdot[idx_r][0, 1:] / self.fcb.M1 - 2 * self.dmdp.mdot[idx_r][1, 1:] / self.fcb.M2)
        else:
            for idx_r in range(self.dv.Nt):
                self.adot_oa_eth.append(- (self.dot_eth[idx_r] / self.fcb.eth_0))
                # calculate everything when e0_prec is True
                if self.fcb.e > 0.0 or self.fcb.e0_prec is True:
                    self.e2dot.append(( - (self.dot_eth[idx_r] / self.fcb.eth_0) 
                                    - 2 * (self.dot_l_tot[idx_r] / self.fcb.l_0)) * (1 - self.fcb.e**2))
                    if self.fcb.e > 0.0:
                        self.adot_oa_l.append(None); self.adot_oa_L.append(None)
                if self.fcb.e == 0.0:
                    self.adot_oa_l.append(2 * self.dot_l_tot[idx_r] / self.fcb.l_0)
                    if self.fcb.q_b == 1.0:
                        self.adot_oa_L.append(2 * self.dot_L_tot[idx_r] / self.fcb.L_0)
                    else:
                        self.adot_oa_L.append(2 * self.dot_L_tot[idx_r] / self.fcb.L_0)

        # convenient strings
        self.ta_str = self.fcb.ta_str
        self.unit_str = self.fcb.unit_str
        self.cb_str = self.fcb.cb_str

    def basic_results(self, t0mean, rmean, idx_r = 0, breakdown=False, header=True, **kwargs):
        self.rmean_t = rmean(self.dv.t[1:])
        self.idx_dvt = kwargs.get("idx_dvt", self.dv.t[1:] > t0mean)
        self.dt_sum = (self.dv.dt[1:])[self.idx_dvt].sum()
        self.dtmean = lambda q : (q * self.dv.dt[1:])[self.idx_dvt].sum() / self.dt_sum
        if self.dmdp_acc_exist:
            self.idx_dmdpt = kwargs.get("idx_dmdpt", self.dmdp.t > t0mean)
            self.mean_dot_m_tot = self.dtmean(self.dmdp.mdot_tot[idx_r][1:])
        else:
            self.mean_dot_m_tot = 0.0
        self.mean_dot_L_tot = self.dtmean(self.dot_L_tot[idx_r])
        self.mean_dot_eth = self.dtmean(self.dot_eth[idx_r])
        if self.spin_flag:
            self.mean_dot_S_1 = self.dtmean(self.dmdp.Sdot[idx_r][0][1:])
            self.mean_dot_S_2 = self.dtmean(self.dmdp.Sdot[idx_r][1][1:])

        if header:
            h_str = f"{'<dot_m>':>18s}{'<dot_eth>':>18s}{'<dot_l>':>18s}{'<dot_L>':>18s}"
            h_str += f"{'<dot_L>/<dot_m>':>18s}" if self.dmdp_acc_exist else ""
            h_str += f"{'<dot_a>/a[dotM/M]':>18s}" if self.dmdp_acc_exist else f"{'<dot_a>/a':>18s}"
            if self.fcb.e > 0 or self.fcb.e0_prec is True:
                h_str += f"{'<dot_e^2>[dotM/M]':>18s}" if self.dmdp_acc_exist else f"{'<dot_e^2>':>18s}"
            #if self.fcb.q_b < 1:
            if self.dmdp_acc_exist:
                h_str += f"{'<eta>':>18s}"
            if breakdown and self.dmdp_acc_exist:
                h_str += f"{'<dot_L_grav>':>18s}{'<dot_L_acc>':>18s}{'<dot_L_pres>':>18s}{'<dot_L_mu_b>':>18s}"
            else:
                print("Warning: breakdown is not available for non-accreting cases (i.e., torque only from gravity).")
            if self.spin_flag:
                h_str += f"{'<dot_S_1>':>18s}{'<dot_S_2>':>18s}"
            print(h_str)
        
        q_str = f"{ self.mean_dot_m_tot :>18.10e}{ self.mean_dot_eth :>18.10e}" \
                f"{ self.dtmean(self.dot_l_tot[idx_r]) :>18.10e}{ self.mean_dot_L_tot :>18.10e}"
        if self.dmdp_acc_exist:
            q_str += f"{ self.dtmean(self.dot_L_tot[idx_r]) / self.mean_dot_m_tot :>18.10e}"\
                     f"{ self.dtmean(self.adot_oa_eth[idx_r]) / self.mean_dot_m_tot :>18.10e}"
            if self.fcb.e > 0 or self.fcb.e0_prec is True:
                q_str += f"{ self.dtmean(self.e2dot[idx_r]) / self.mean_dot_m_tot :>18.10e}"
            #if self.fcb.q_b < 1:
            if self.fcb.M1 <= self.fcb.M2:
                q_str += f"{self.dtmean(self.dmdp.mdot[idx_r][0, 1:]) / self.mean_dot_m_tot :>18.10e}"
            if self.fcb.M2 < self.fcb.M1:
                q_str += f"{self.dtmean(self.dmdp.mdot[idx_r][1, 1:]) / self.mean_dot_m_tot :>18.10e}"
            if breakdown:
                q_str += f"{ self.dtmean(self.dot_L_grav[idx_r]) :>18.10e}{ self.dtmean(self.dot_L_acc[idx_r]) :>18.10e}" \
                      + f"{ self.dtmean(self.dot_L_pres[idx_r]) :>18.10e}{ self.dtmean(self.dot_L_mu_b[idx_r]) :>18.10e}"
            if self.spin_flag:
                q_str += f"{ self.mean_dot_S_1 :>18.10e}{ self.mean_dot_S_2 :>18.10e}"
        else:
            q_str += f"{ self.dtmean(self.adot_oa_eth[idx_r]):>18.10e}"
            if self.fcb.e > 0 or self.fcb.e0_prec is True:
                q_str += f"{ self.dtmean(self.e2dot[idx_r]):>18.10e}"
        
        print(q_str)


# ******************************************************************************
# Handle data for a clean restart (for dmdp_acc.dat, dv_grav.dat, and dp_spin.dat)
# Example usage:
# >>> DatManipulator.cut_dat("dmdp_acc.dat.bak", 2 + 14 * 3, N_cycles, "dmdp_acc.dat")
#     DatManipulator.cut_dat("dv_grav.dat.bak", 2 + 8 * 3, N_cycles, "dv_grav.dat")
#     DatManipulator.cut_dat("dp_spin.dat.bak", 2 * 3, N_cycles, "dv_grav.dat")
#
# ******************************************************************************

class DatManipulator:
    
    def __init__(self):
        pass
    
    @staticmethod
    def cut_dat(filename, data_per_row, nstep_cut, out_file, header=True):
        
        f = open(filename, 'rb')
        if header:
            tmp_line = f.readline() # currently, there should only be one number
            Nt = int(tmp_line)
            
        pos_i = f.tell(); f.seek(0, 2); num_bytes = f.tell() - pos_i
        num_rows = num_bytes // 8 // data_per_row
        if num_rows * 8 * data_per_row > num_bytes:
            raise IOError(f"The number of bytes seems off: rows={num_rows}, num_bytes={num_bytes}")
        elif num_rows * 8 * data_per_row < num_bytes:
            print(f"Bytes more than data: rows={num_rows}, num_bytes={num_bytes}; discard extra")
            num_rows -= 1 # something must be off in the end, let's be safe
        else:
            pass
        f.seek(pos_i, 0)
        data = rxu.loadbin(f, num=num_rows*data_per_row).reshape([num_rows, data_per_row]);
        f.close()
        #idx_cut = np.argmax(data[:, 0] > t_cut)
        #idx_cut = min(idx_cut + 1, num_rows)
        idx_cut = nstep_cut
        
        if header:
            f = open(out_file, "w")
            f.write(str(Nt)+'\n')
            f.close()
            f = open(out_file, "ab")            
        else:
            f = open(out_file, "wb") 
        f.write(data[:idx_cut].flatten().tobytes())
        f.close()
        
    @staticmethod
    def fold_dat(filename, data_per_row, t_fold, out_file, header=True):
        """ Assume there is only one fold """
        
        f = open(filename, 'rb')
        if header:
            tmp_line = f.readline() # currently, there should only be one number
            Nt = int(tmp_line)
            
        pos_i = f.tell(); f.seek(0, 2); num_bytes = f.tell() - pos_i
        num_rows = num_bytes // 8 // data_per_row
        if num_rows * 8 * data_per_row > num_bytes:
            raise IOError(f"The number of bytes seems off: rows={num_rows}, num_bytes={num_bytes}")
        elif num_rows * 8 * data_per_row < num_bytes:
            print(f"Bytes more than data: rows={num_rows}, num_bytes={num_bytes}; discard extra")
            num_rows -= 1 # something must be off in the end, let's be safe
        else:
            pass
        f.seek(pos_i, 0)
        data = rxu.loadbin(f, num=num_rows*data_per_row).reshape([num_rows, data_per_row]);
        f.close()
        idx_left = np.argmax(data[:, 0] > t_fold)
        idx_left = idx_left + 1
        idx_right = np.where(np.diff(data[:, 0]) < 0)[0]
        if idx_right.size != 1:
            raise RuntimeError(f"There seems to be multiple ({idx_right.size}?) restarts. Manual reduction is needed.")
        idx_right = idx_right[0]+1
        
        if header:
            f = open(out_file, "w")
            f.write(str(Nt)+'\n')
            f.close()
            f = open(out_file, "ab")            
        else:
            f = open(out_file, "wb")
        f.write(np.vstack([data[:idx_left], data[idx_right:]]).flatten().tobytes())
        f.close()
    
    @staticmethod
    def stitch_dat(file_list):
        
        pass # for now, just use cat command (restart won't put Nt, so no header)


class GasHistory:
    def __init__(self, hst_filepath):
        gas_hst_scalars = np.loadtxt(hst_filepath)
        try:
            self.time, self.dt, self.mass, self.E, self.Mx, self.My, self.Mz, \
                self.KEx, self.KEy, self.KEz, self.RhoVxdVy, self.rhodVy2, self.ErhoPhi = gas_hst_scalars.T
        except ValueError:
            self.time, self.dt, self.mass, self.Mx, self.My, self.Mz, \
                self.KEx, self.KEy, self.KEz, self.RhoVxdVy, self.rhodVy2 = gas_hst_scalars.T
