""" Match SDSS spectroscopic catalogs with S-PLUS. """
import os

import numpy as np
import astropy.units as u
from astropy.table import Table, hstack, vstack
from astropy.coordinates import SkyCoord
from tqdm import tqdm

import context

def match_SPLUS_dr1_galSpecLine():
    cat_dir = os.path.join(context.home_dir, "catalogs")
    splus_dr1_cat = os.path.join(cat_dir, "SPLUS_STRIPE82_master_catalogue" \
                                          "_dr_march2019_flag_duplicities.fits")
    splus = Table.read(splus_dr1_cat)
    splus_coords = SkyCoord(splus["RA"].data * u.degree, splus["Dec"].data *
                       u.degree)
    sdss_dr8_cat = os.path.join(cat_dir, "galSpecLines.csv")
    sdss = Table.read(sdss_dr8_cat)
    sdss_coords = SkyCoord(sdss["ra"].data * u.degree, sdss["dec"].data *
                       u.degree)
    idx, d2d, d3d = sdss_coords.match_to_catalog_sky(splus_coords)
    splus = splus[idx]
    idx = np.where(d2d < 2 * u.arcsec)[0]
    match = hstack([sdss, splus])[idx]
    match.write(os.path.join(cat_dir, "match_splus_dr1_galSpecLine.fits"),
                overwrite=True)


def match_SPLUS_idr3_galSpecLine():
    cat_dir = os.path.join(context.home_dir, "catalogs")
    splus_dir = os.path.join(cat_dir, "SPLUS-iDR3")
    galSpecLine = os.path.join(cat_dir, "galSpecLines.csv")
    sdss = Table.read(galSpecLine)
    sdss_coords = SkyCoord(sdss["ra"].data * u.degree, sdss["dec"].data *
                       u.degree)
    catalogs = os.listdir(splus_dir)
    matches = []
    for catname in tqdm(catalogs, desc="Matching SPLUS and SDSS catalogs"):
        try:
            splus = Table.read(os.path.join(splus_dir, catname))
        except:
            print("Problem with catalog {}".format(catname))
            continue
        splus_coords = SkyCoord(splus["RA"].data * u.degree,
                                splus["DEC"].data * u.degree)
        idx, d2d, d3d = sdss_coords.match_to_catalog_sky(splus_coords)
        splus = splus[idx]
        idx = np.where(d2d < 1.5 * u.arcsec)[0]
        match = hstack([sdss, splus])[idx]
        matches.append(match)
    matches = vstack(matches)
    matches.write(os.path.join(cat_dir, "match_splus_idr3_galSpecLine.fits"),
                overwrite=True)

def match_SPLUS_idr3_Portsmouth():
    cat_dir = os.path.join(context.home_dir, "catalogs")
    splus_dir = os.path.join(cat_dir, "SPLUS-iDR3")
    galSpecLine = os.path.join(cat_dir, "PortStripe82.csv")
    sdss = Table.read(galSpecLine)
    sdss_coords = SkyCoord(sdss["ra"].data * u.degree, sdss["dec"].data *
                       u.degree)
    catalogs = os.listdir(splus_dir)
    matches = []
    for catname in tqdm(catalogs, desc="Matching SPLUS and SDSS catalogs"):
        splus = Table.read(os.path.join(splus_dir, catname))
        splus_coords = SkyCoord(splus["RA"].data * u.degree,
                                splus["DEC"].data * u.degree)
        idx, d2d, d3d = sdss_coords.match_to_catalog_sky(splus_coords)
        splus = splus[idx]
        idx = np.where(d2d < 1.5 * u.arcsec)[0]
        match = hstack([sdss, splus])[idx]
        matches.append(match)
    matches = vstack(matches)
    matches.write(os.path.join(cat_dir, "match_splus_idr3_Portsmouth.fits"),
                overwrite=True)

if __name__ == "__main__":
    # match_SPLUS_dr1_galSpecLine()
    # match_SPLUS_idr3_galSpecLine()
    match_SPLUS_idr3_Portsmouth()