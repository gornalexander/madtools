from copy import copy
import numpy as np


def calc_corr_factor(x, y):
    cov = np.cov(x, y)
    k = cov[0,1] / cov[0,0]
    return k

def calc_beam_pars(beam, gamma=400 / 0.938, center_beam=True):
    pars = {}
    if center_beam:
        beam = copy(beam) - beam.mean()
    pars['sigma_x'] = np.sqrt((beam.x**2).mean())
    pars['sigma_y'] = np.sqrt((beam.y**2).mean())
    pars['sigma_xp'] = np.sqrt((beam.px**2).mean())
    pars['sigma_yp'] = np.sqrt((beam.py**2).mean())
    pars['dpp'] = np.sqrt((beam.pt**2).mean())
    pars['geom_emitt_x'] = np.sqrt((beam.x**2).mean() * (beam.px**2).mean() - (beam.x*beam.px).mean()**2)
    pars['geom_emitt_y'] = np.sqrt((beam.y**2).mean() * (beam.py**2).mean() - (beam.y*beam.py).mean()**2)
    pars['emitt_norm_x'] = pars['geom_emitt_x'] * gamma
    pars['emitt_norm_y'] = pars['geom_emitt_y']  * gamma
    pars['beta_x'] = pars['sigma_x']**2 / pars['geom_emitt_x']
    pars['beta_y'] = pars['sigma_y']**2 / pars['geom_emitt_y']
    pars['alpha_x'] = - (beam.x*beam.px).mean() / pars['geom_emitt_x']
    pars['alpha_y'] = - (beam.y*beam.py).mean() / pars['geom_emitt_y']
    pars['gamma_x'] = pars['sigma_xp']**2 / pars['geom_emitt_x']
    pars['gamma_y'] = pars['sigma_yp']**2 / pars['geom_emitt_y']
    return pars