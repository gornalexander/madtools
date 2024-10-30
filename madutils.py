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

def calculate_scr(beam, betx_max, bety_max, sigma_factor=7, orbit_error_m=5e-3, size_error_rel=0.1, dispersion_error_rel=0.1, alignment_error_m=1e-3):
    out = []
    for loc, b in beam.groupby('idx'):
        outi = {}
        pars = calc_beam_pars(b)
        outi['scr_x'] = pars['sigx'] * (sigma_factor + size_error_rel) + orbit_error_m * np.sqrt(pars['betx']/betx_max) + dispersion_error_rel * pars['dx'] * pars['dpp'] + alignment_error_m
        outi['scr_y'] = pars['sigy'] * (sigma_factor + size_error_rel) + orbit_error_m * np.sqrt(pars['bety']/bety_max) + dispersion_error_rel * pars['dx'] * pars['dpp'] + alignment_error_m
        x, y = b.x.mean(), b.y.mean() 
        outi['scr_x_up'] = x + outi['scr_x']
        outi['scr_x_low'] = x - outi['scr_x']
        outi['scr_y_up'] = y + outi['scr_y']
        outi['scr_y_low'] = y - outi['scr_y']
        out.append(pd.DataFrame(outi, index=[loc]))
    return pd.concat(out, axis=0)

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


def calc_envelope(beam, groupby='index'):
    if groupby == 'index':
        beam.index.names = ['idx']
        gb = beam.groupby('idx', sort=False)
    else:
        gb = beam.groupby(groupby, sort=False)

    env = {}
    env['s'] = gb.s.mean()
    env['sx'] = gb.x.std()
    env['sy'] = gb.y.std()
    env['minx'] = gb.x.min()
    env['miny'] = gb.y.min()
    env['maxx'] = gb.x.max()
    env['maxy'] = gb.y.max()
    env['xp0_5'] = gb.x.quantile(0.5/100)
    env['yp0_5'] = gb.y.quantile(0.5/100)
    env['xp99_5'] = gb.x.quantile(99.5/100)
    env['yp99_5'] = gb.y.quantile(99.5/100)
    env['xp5'] = gb.x.quantile(5/100)
    env['yp5'] = gb.y.quantile(5/100)
    env['xp95'] = gb.x.quantile(95/100)
    env['yp95'] = gb.y.quantile(95/100)
    env['medx'] = gb.x.median()
    env['medy'] = gb.y.median()
    env['meanx'] = gb.x.mean()
    env['meany'] = gb.y.mean()
    return pd.DataFrame(env)#.sort_values('s')

def split_beam(beam, splitting_ratio=0.5, recenter=True):
    beam.sort_values('y', ascending=False, inplace=True)
    upper = beam.iloc[:int(len(beam.index)*splitting_ratio)]
    lower = beam.iloc[int(len(beam.index)*splitting_ratio):]
    if recenter:
        upper.y -= upper.y.mean()
        lower.y -= lower.y.mean()
        upper.py -= upper.py.mean()
        lower.py -= lower.py.mean()
    return upper, lower