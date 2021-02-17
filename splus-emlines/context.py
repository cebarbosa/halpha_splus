import os

import astropy.units as u
from dustmaps.config import config as dustconfig
from dustmaps import sfd

home_dir = "/home/kadu/Dropbox/splus-emlines"
data_dir = os.path.join(home_dir, "data")

dustmaps_dir = os.path.join(data_dir, "dustmaps")
dustconfig["data_dir"] = dustmaps_dir
if not os.path.exists(os.path.join(dustmaps_dir, "sfd")):
    sfd.fetch()

ps = 0.55 * u.arcsec / u.pixel


bands = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I',
         'F861', 'Z']

narrow_bands = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']

broad_band = ['U', 'G', 'R', 'I', 'Z']

bands_names = {'U' : "$u$", 'F378': "$J378$", 'F395' : "$J395$",
               'F410' : "$J410$", 'F430' : "$J430$", 'G' : "$g$",
               'F515' : "$J515$", 'R' : "$r$", 'F660' : "$J660$",
               'I' : "$i$", 'F861' : "$J861$", 'Z' : "$z$"}

wave_eff = {"F378": 3773.4, "F395": 3940.8, "F410": 4095.4,
            "F430": 4292.5, "F515": 5133.5, "F660": 6614.0, "F861": 8608.1,
            "G": 4647.8, "I": 7683.8, "R": 6266.6, "U": 3536.5,
            "Z": 8679.5}

exptimes = {"F378": 660, "F395": 354, "F410": 177,
            "F430": 171, "F515": 183, "F660": 870, "F861": 240,
            "G": 99, "I": 138, "R": 120, "U": 681,
            "Z": 168}

flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
