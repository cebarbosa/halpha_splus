# -*- coding: utf-8 -*-
""" 

Created on 08/04/18

Author : Carlos Eduardo Barbosa

Methods to extract H-alpha in the SPLUS filters according to methods presented
in Villela-Rojo et al. 2015 (VR+15).

"""
from __future__ import division, print_function

import os

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps

class PhotEmissionLines():
    def __init__(self, wave, bands, wtrans, trans):
        self.wave = wave
        self.bands = bands
        self.wtrans = wtrans
        self.trans = trans
        self.wpiv = self.calc_wpiv()
        self.alphax = self.calc_alphax()
        self.deltax = self.calc_deltax()
        self.betax = self.calc_betax()

    def calc_wpiv(self):
        """ Numerical integration of equation (4) in VR+2015 for SPLUS
        system. """
        wpiv = []
        for band in self.bands:
            wave = self.wtrans[band]
            trans = self.trans[band]
            curve = interp1d(wave, trans, kind="linear", fill_value=0.,
                             bounds_error=False)
            term1 = np.trapz(curve(wave) * wave, wave)
            term2 = np.trapz(curve(wave) / wave, wave)
            wpiv.append(np.sqrt(term1 / term2))
        return u.Quantity(wpiv)

    def calc_alphax(self):
        """ Numerical integration of equation (4) in VR+2015 for SPLUS
        system. """
        alphax = []
        for band in self.bands:
            wave = self.wtrans[band]
            trans = self.trans[band]
            curve = interp1d(wave, trans, kind="linear", fill_value=0.,
                             bounds_error=False)
            term1 = np.trapz(trans * wave * wave, wave) / curve(self.wave) / \
                    self.wave
            term2 = np.trapz(trans * wave, wave) / curve(self.wave) / self.wave
            alphax.append(term1 / term2)
        return u.Quantity(alphax)

    def calc_deltax(self):
        """ Numerical integration of equation (2) in VR+2015. """
        deltax = []
        for band in self.bands:
            wave = self.wtrans[band]
            trans = self.trans[band]
            curve = interp1d(wave, trans, kind="linear", fill_value=0.,
                             bounds_error=False)
            val = np.trapz(trans * wave, wave) / curve(self.wave) / self.wave
            deltax.append(val)
        return u.Quantity(deltax)

    def calc_betax(self):
        """ Determination of second term of equation (4) using deltax. """
        return 1 / self.deltax

    def flux_3F(self, flux):
        """Three filter method from VR+2015 (equation (3)). """
        a = (self.alphax[0] - self.alphax[2]) / \
            (self.alphax[1] - self.alphax[2])
        numen = (flux[0] - flux[2]) - a * (flux[1] - flux[2])
        denom = - self.betax[1] * a + self.betax[0]
        return numen / denom

    def EW(self, mag):
        """ Calculates the equivalent width using equation (13) of VR+2015. """
        Q = np.power(10, 0.4 * (mag[0] - mag[1]))
        eps = self.deltax[1] / self.deltax[0]
        return self.deltax[1] * (Q - 1)**2 / (1 - Q * eps)


def halpha_2F_method(f660, fr):
    """Simplest method: using r-band to extimate the continuum (Pascual et al.
    2007). See equation (1) from VR+15 """
    # Numerical integration of equation (2)
    # deltas = deltax(6562.8 * u.AA)
    deltar = 1319.50 * u.AA # Value calculated using deltax != VR+2015
    deltaf660 = 151.06 * u.AA # Value calculated using deltax != VR+2015
    return  deltaf660 * (f660 - fr) / (1 - deltaf660 / deltar)

def halpha_3F_method(f660, fr, fi, test=False):
    """Three bands method from VR+2015 (equation (3)). """
    # Constants used in the calculation
    a = 1.29075002747
    betas = {"I": 1.1294371504165924e-07 / u.AA,
             'F660': 0.006619605316773919 / u.AA,
             "R" : 0.0007578642946167448 / u.AA}
    numen = (fr - fi) - a * (f660 - fi)
    denom = - betas["F660"] * a + betas["R"]
    return numen / denom