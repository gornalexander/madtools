from copy import copy
import numpy as np
import pandas as pd
from scipy.constants import proton_mass, c, e


proton_mass_GeV = proton_mass * c**2 / e * 1e-9


def calc_corr_factor(x, y):
    cov = np.cov(x, y)
    k = cov[0,1] / cov[0,0]
    return k


def calc_beam_pars(beam, energy_GeV=400, mass_GeV=proton_mass_GeV):
    pars = {}
    try:
        gamma = beam.e.mean() / mass_GeV
    except AttributeError:
        gamma = energy_GeV / mass_GeV
    covxpx = np.cov(beam.x, beam.px)
    covypy = np.cov(beam.y, beam.py)
    pars['rel_gamma'] = gamma
    if 's' in beam.columns:
        pars['s'] = beam.s.mean()
    pars['sigx'] = np.sqrt(covxpx[0,0])
    pars['sigy'] = np.sqrt(covypy[0,0])
    pars['sigpx'] = np.sqrt(covxpx[1,1])
    pars['sigpy'] = np.sqrt(covypy[1,1])
    pars['dpp'] = np.sqrt((beam.pt**2).mean())
    pars['geom_emitt_x'] = np.linalg.det(covxpx)**0.5
    pars['geom_emitt_y'] = np.linalg.det(covypy)**0.5
    pars['emitt_norm_x'] = pars['geom_emitt_x'] * gamma
    pars['emitt_norm_y'] = pars['geom_emitt_y']  * gamma
    pars['betx'] = pars['sigx']**2 / pars['geom_emitt_x']
    pars['bety'] = pars['sigy']**2 / pars['geom_emitt_y']
    pars['alfx'] = - (beam.x*beam.px).mean() / pars['geom_emitt_x']
    pars['alfy'] = - (beam.y*beam.py).mean() / pars['geom_emitt_y']
    pars['gamx'] = pars['sigpx']**2 / pars['geom_emitt_x']
    pars['gamy'] = pars['sigpy']**2 / pars['geom_emitt_y']
    pars['dx'] = calc_corr_factor(beam.pt, beam.x)
    pars['dy'] = calc_corr_factor(beam.pt, beam.y)
    pars['dpx'] = calc_corr_factor(beam.pt, beam.px)
    pars['dpy'] = calc_corr_factor(beam.pt, beam.py)
    # pars['beta_geom_emitt_x'] = np.linalg.det(np.cov(beam.x - pars['dx'] * beam.pt, beam.px - pars['dpx'] * beam.pt))
    # pars['beta_geom_emitt_y'] = np.linalg.det(np.cov(beam.y - pars['dy'] * beam.pt, beam.py - pars['dpy'] * beam.pt))
    # pars['beta_emitt_norm_x'] = pars['beta_geom_emitt_x'] * gamma
    # pars['beta_emitt_norm_y'] = pars['beta_geom_emitt_y'] * gamma
    return pars


def remove_dispersion(beam):
    beam = copy(beam)
    beam.x -= calc_corr_factor(beam.pt, beam.x) * beam.pt
    beam.y -= calc_corr_factor(beam.pt, beam.y) * beam.pt
    beam.px -= calc_corr_factor(beam.pt, beam.px) * beam.pt
    beam.py -= calc_corr_factor(beam.pt, beam.py) * beam.pt
    return beam


def get_twiss_from_distribution(particles, rm_dispersion=False):
    locs = particles.index.drop_duplicates()
    df_list = []
    for i, loc in enumerate(locs):
        beam = particles.loc[loc]
        if rm_dispersion:
            beam = remove_dispersion(beam)
        pars = pd.DataFrame(calc_beam_pars(beam), index=[loc])
        df_list.append(pars)
    return pd.concat(df_list)


def get_initial_condition(beam):
    beam_pars = calc_beam_pars(beam)
    initial_condition = dict(
        betx = beam_pars['betx'],
        alfx = beam_pars['alfx'],
        bety = beam_pars['bety'],
        alfy = beam_pars['alfy'],
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