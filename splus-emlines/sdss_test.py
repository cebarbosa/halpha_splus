import os

import numpy as np
from astropy.table import Table
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import extinction
from dustmaps.sfd import SFDQuery

import context
from emlines_estimator import EmLine3Filters

def mag2flux(m, weff):
    fnu = np.power(10, -0.4 * (m + 48.6)) * context.fnu_unit
    return (fnu * const.c.to("km/s") / weff[:,None]**2).to(context.flam_unit)

def test_sdss_halpha(fname, fluxkey, magkey, magerrprefix, bands,
                     scatter_density=True, zkey="z", xmin=-17, xmax=-13,
                     output=None, bands_ebv=None, wline=6562.8 * u.AA):
    bands_ebv = ["G", "I"] if bands_ebv is None else bands_ebv
    ############################################################################
    # Reading transmission curves for S-PLUS
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
    halpha = EmLine3Filters(wline, [_.upper() for _ in bands],
                            wcurves, trans)
    ############################################################################
    # Reading catalog and remove objects out of z-range
    wdir = os.path.join(context.home_dir, "catalogs")
    filename = os.path.join(wdir, fname)
    table = Table.read(filename)
    if zkey in table.colnames:
        table = table[table[zkey] < 0.02]
    ############################################################################
    # Cleaning table against large errors in mgnitudes
    magerrs = np.array([table["{}{}{}".format(magerrprefix, band, magkey)].data
                        for band in bands])
    idx = np.where(np.nanmax(magerrs, axis=0) < 0.2)[0]
    table = table[idx]
    ############################################################################
    # Estimating the extinction
    gi = np.array([table["{}{}".format(band, magkey)].data for band in
                     bands_ebv])
    g_i = gi[0] - gi[1]
    ebvs = np.clip(0.206 * np.power(g_i, 1.68) - 0.0457, 0,
                   np.infty)
    Rv = 4.1
    Avs = Rv * ebvs
    corr = np.array([extinction.remove(
                     extinction.calzetti00(np.atleast_1d(wline), Av, Rv)[0], 1)
                     for Av in Avs])
    ############################################################################
    # Correcting observations for Galactic extinction
    coords = SkyCoord(table["ra"] * u.degree, table["dec"] * u.degree)
    sfd = SFDQuery()
    ebv_mw = sfd(coords)
    Rv_mw = 3.1
    Av_mw = Rv_mw * ebv_mw
    corr_mw = np.array([extinction.remove(
                     extinction.ccm89(halpha.wpiv, Av, Rv_mw)[0], np.ones(3))
                     for Av in Av_mw]).T
    ############################################################################
    # Calculating H-alpha
    mags = np.array([table["{}{}".format(band, magkey)].data for band in
                     bands])
    flam = mag2flux(mags, halpha.wpiv) * corr_mw
    flux_halpha_nii = halpha.flux_3F(flam)
    log_halpha = np.where(g_i <= 0.5,
                          0.989 * np.log10(flux_halpha_nii.value) - 0.193,
                          0.954 * np.log10(flux_halpha_nii.value) - 0.753)
    halpha_sdss = np.log10((table[fluxkey]) * np.power(10., -17))
    halpha_splus = log_halpha + np.log10(corr)
    ############################################################################
    # Selecting only regions within limits of flux
    idx = np.where(np.isfinite(halpha_sdss * halpha_splus) &
                   (halpha_sdss > xmin) & (halpha_splus < xmax))
    halpha_sdss = halpha_sdss[idx]
    halpha_splus = halpha_splus[idx]
    ############################################################################
    fig = plt.figure()
    if scatter_density:
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        density = ax.scatter_density(halpha_sdss, halpha_splus,
                                     cmap="magma_r")
        fig.colorbar(density, label='Number of points per pixel')

    else:
        ax = fig.add_subplot(1, 1, 1)
        label = "mag_key={}".format(magkey)
        counts, xedges, yedges, im = ax.hist2d(halpha_sdss, halpha_splus,
                                               bins=(50,50), cmap="bone_r")
        plt.legend(title=label)
        fig.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    plt.plot(np.linspace(xmin, xmax, 100), np.linspace(xmin, xmax, 100), "--r")
    plt.xlabel(r"$\log$ H$\alpha$ (erg / s / cm$^2$) -- SDSS")
    plt.ylabel(r"$\log$ H$\alpha$ (erg / s / cm$^2$) -- SPLUS")
    axins = inset_axes(ax, width="80%", height="80%",
                       bbox_to_anchor=(.65, .05, .38, .38),
                       bbox_transform=ax.transAxes, loc=3)
    diff = halpha_splus - halpha_sdss
    median = np.nanmedian(diff)
    std = np.nanstd(diff)
    title = "med: {:.2f} std: {:.2f}".format(median, std)
    axins.hist(diff, bins=20)
    axins.set_title(title)
    plt.tight_layout()
    if "output" is not None:
        plt.savefig(output, dpi=250, )
    plt.show()

if __name__ == "__main__":
    wdir = os.path.join(context.home_dir, "catalogs")
    outdir = os.path.join(context.home_dir, "plots")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    ############################################################################
    # Adding colormap for scatter density plots
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'), (1e-20, '#440053'), (0.2, '#404388'),
        (0.4, '#2a788e'), (0.6, '#21a784'), (0.8, '#78d151'), (1, '#fde624'),
         ], N=256)
    ############################################################################
    # Catalog matched by Amanda, iDR3 + Portsmouth group
    Port = {"fname": "stripe82_elPort_xmatch_splus.csv",
            "fluxkey": "Flux_Ha_6562", "magkey": "_aper_3",
            "magerrprefix": "e_", "bands": ["R", "F660", "I"],
            "scatter_density": False}
    # test_sdss_halpha(**Port)
    ############################################################################
    # SPLUS DR1 +  galSpecLine match
    splus_dr1_galSpecLine = {"fname": "match_splus_dr1_galSpecLine.fits",
            "fluxkey": "h_alpha_flux", "magkey": "_aper",
            "magerrprefix": "e", "bands": ["r", "F660", "i"],
            "scatter_density": False, "zkey": "zb", "xmin": -17, "xmax":-13}
    # test_sdss_halpha(**splus_dr1_galSpecLine)
    ############################################################################
    splus_idr3_galSpecLine = {"fname": "match_splus_idr3_galSpecLine.fits",
            "fluxkey": "h_alpha_flux", "magkey": "_aper_6",
            "magerrprefix": "e_", "bands": ["R", "F660", "I"],
            "scatter_density": True, "xmin": -18, "xmax":-13}
    # test_sdss_halpha(**splus_idr3_galSpecLine)
    ############################################################################
    # SPLUS iDR3 + Portsmouth
    for magkey in ["_aper_3", "_aper_6", "_PStotal"]:
        splus_idr3_Port = {"fname": "match_splus_idr3_Portsmouth.fits",
                "fluxkey": "Flux_Ha_6562", "magkey": magkey,
                "magerrprefix": "e_", "bands": ["R", "F660", "I"],
                "scatter_density": False, "zkey": "z", "xmin": -18, "xmax":-12,
                "output": os.path.join(outdir, "SPLUS_idr3_Portsmouth_{"
                                               "}.png".format(magkey))}
        test_sdss_halpha(**splus_idr3_Port)