import os

import numpy as np
from astropy.table import Table
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

import context
from halpha_estimators import halpha_3F_method, PhotEmissionLines

def mag2flux(m, weff):
    fnu = np.power(10, -0.4 * (m + 48.6)) * context.fnu_unit
    return (fnu * const.c.to("km/s") / weff[:,None]**2).to(context.flam_unit)

def test_sdss_halpha(sample="SDSS"):
    filters_dir = os.path.join(os.getcwd(), "filter_curves-master")
    filenames = sorted([os.path.join(filters_dir, _) for _ in os.listdir( \
                        filters_dir)])
    filternames = [os.path.split(_)[1].replace(".dat", "") for _ in filenames]
    filternames = [_.replace("F0", "F").replace("JAVA", "") for _ in
                   filternames]
    filternames = [_.replace("SDSS", "").upper() for _ in filternames]
    fcurves = [np.loadtxt(f) for f in filenames]
    wcurves = [curve[:,0] * u.AA for curve in fcurves]
    trans = [curve[:,1] for curve in fcurves]
    wcurves = dict(zip(filternames, wcurves))
    trans =  dict(zip(filternames, trans))
    bands_halpha = ["R", "F660", "I"]

    halpha = PhotEmissionLines(6562.8 * u.AA, bands_halpha, wcurves,
                                         trans)
    wdir = os.path.join(context.data_dir, "sdss")
    filename = {"Port": os.path.join(wdir, "stripe82_elPort_xmatch_splus.csv"),
                "SDSS": os.path.join(wdir, "stripe82_sdss_xmatch.csv"),
                "SDSS2": os.path.join(wdir, "sdss+splus_ewha_lim.csv")}[sample]
    hkey= {"Port": "Flux_Ha_6562", "SDSS": "h_alpha_flux",
           "SDSS2": "h_alpha_flux"}[sample]
    table = Table.read(filename)
    if sample == "SDSS2":
        bands_halpha = ["r", "F660", "i"]
    else:
        table = table[table["z"] < 0.01]
    mags = np.array([table["{}_aper".format(band)].data for band in
                     bands_halpha])
    flam = mag2flux(mags, halpha.wpiv)
    flux_halpha = halpha.flux_3F(flam)
    ew_halpha = halpha.EW(mags)
    halpha_sdss = np.log10((table[hkey].data) * np.power(10., -17))
    halpha_splus = np.log10(flux_halpha.value)
    idx = np.where(np.isfinite(halpha_sdss * halpha_splus))
    halpha_sdss = halpha_sdss[idx]
    halpha_splus = halpha_splus[idx]
    xmin = np.nanpercentile(halpha_splus, 10)
    xmax = np.nanpercentile(halpha_splus, 90)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(halpha_sdss, halpha_splus, "o")
    # ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    # density = ax.scatter_density(halpha_sdss, halpha_splus, cmap=white_viridis)
    # fig.colorbar(density, label='Number of points per pixel')
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    plt.plot(np.linspace(xmin, xmax, 100), np.linspace(xmin, xmax, 100), "--r")
    plt.xlabel(r"$\log$ H$\alpha$ (erg / s / cm$^2$) -- SDSS")
    plt.ylabel(r"$\log$ H$\alpha$ (erg / s / cm$^2$) -- SPLUS")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'), (1e-20, '#440053'), (0.2, '#404388'),
        (0.4, '#2a788e'), (0.6, '#21a784'), (0.8, '#78d151'), (1, '#fde624'),
         ], N=256)
    test_sdss_halpha(sample="SDSS2")
