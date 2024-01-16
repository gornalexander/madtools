from copy import copy
import numpy as np
import pandas as pd


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
    pars['dx'] = calc_corr_factor(beam.pt, beam.x)
    pars['dy'] = calc_corr_factor(beam.pt, beam.y)
    pars['dpx'] = calc_corr_factor(beam.pt, beam.px)
    pars['dpy'] = calc_corr_factor(beam.pt, beam.py)
    return pars

def get_initial_condition(beam):
    beam_pars = calc_beam_pars(beam)
    initial_condition = dict(
        betx = beam_pars['beta_x'],
        alfx = beam_pars['alpha_x'],
        bety = beam_pars['beta_y'],
        alfy = beam_pars['alpha_y'],
        dx = calc_corr_factor(beam.pt, beam.x),
        dy = calc_corr_factor(beam.pt, beam.y),
        dpx = calc_corr_factor(beam.pt, beam.px),
        dpy = calc_corr_factor(beam.pt, beam.py)
    )
    return initial_condition

def calc_envelope(particles):
    locs = particles.index.unique()
    d = dict(
        s = [],
        minx = [],
        miny = [],
        maxx = [],
        maxy = [],
        xp0_5 = [],
        yp0_5 = [],
        xp99_5 = [],
        yp99_5 = [],
        xp5 = [],
        yp5 = [],
        xp95 = [],
        yp95 = [],
        medx = [],
        medy = []
    )
    for i, bsg in enumerate(locs):
        try:
            d['s'].append(particles.loc[bsg]['s'][0])
            distx = particles.loc[bsg]['x'] - 0*np.mean(particles.loc[bsg]['x'])
            disty = particles.loc[bsg]['y'] - 0*np.mean(particles.loc[bsg]['y'])
            d['minx'].append(np.min(distx))
            d['miny'].append(np.min(disty))
            d['maxx'].append(np.max(distx))
            d['maxy'].append(np.max(disty))
            d['xp0_5'].append(np.percentile(distx, 0.5))
            d['yp0_5'].append(np.percentile(disty, 0.5))
            d['xp99_5'].append(np.percentile(distx, 99.5))
            d['yp99_5'].append(np.percentile(disty,99.5))
            d['xp5'].append(np.percentile(distx, 5))
            d['yp5'].append(np.percentile(disty, 5))
            d['xp95'].append(np.percentile(distx, 95))
            d['yp95'].append(np.percentile(disty, 95))
            d['medx'].append(np.median(distx))
            d['medy'].append(np.median(disty))
        except:
            pass
    return pd.DataFrame(d)