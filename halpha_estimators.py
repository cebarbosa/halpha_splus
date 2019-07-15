# -*- coding: utf-8 -*-
""" 

Created on 08/04/18

Author : Carlos Eduardo Barbosa

Methods to extract H-alpha in the SPLUS filters according to methods presented
in Villela-Rojo et al. 2015 (VR+15).

"""
from __future__ import division, print_function

import astropy.units as u

def halpha_2F_method(f660, fr):
    """Simplest method: using r-band to extimate the continuum (Pascual et al.
    2007). See equation (1) from VR+15 adapted to S-PLUS system."""
    deltar = 1319.50 * u.AA
    deltaf660 = 151.06 * u.AA
    return  deltaf660 * (f660 - fr) / (1 - deltaf660 / deltar)

def halpha_3F_method(f660, fr, fi):
    """Three bands method from VR+2015 (equation (3)). """
    # Constants used in the calculation using S-PLUS bands.
    a = 1.29075002747
    betas = {"I": 1.1294371504165924e-07 / u.AA,
             'F660': 0.006619605316773919 / u.AA,
             "R" : 0.0007578642946167448 / u.AA}
    numen = (fr - fi) - a * (f660 - fi)
    denom = - betas["F660"] * a + betas["R"]
    return numen / denom

if __name__ == "__main__":
    pass

