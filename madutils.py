from copy import copy
import numpy as np
import pandas as pd
from scipy.constants import proton_mass, c, e

units = {
    'm': 1,
    'mm': 1e-3,
    'um': 1e-6,
    'nm': 1e-9,
    'rad': 1,
    'mrad': 1e-3,
    'urad': 1e-6,
    'nrad': 1e-9,
}

proton_mass_GeV = proton_mass * c**2 / e * 1e-9

def get_z_score(values):
    return abs(values - values.mean()) / values.std()

def calc_z_score(beam, inplace=False):
    z_score = beam.copy().select_dtypes(include=float).apply(get_z_score).max(axis=1)
    if inplace:
        beam['z_score'] = z_score
    return z_score

def filter_outliners(beam, max_z_score=100, inplace=False):
    z_score = calc_z_score(beam, inplace=inplace)
    if inplace:
        b = beam.query('z_score <= @max_z_score', inplace=inplace)
    else:
        b = beam[z_score <= max_z_score]
    return b

def calc_corr_factor(x, y):
    cov = np.cov(x, y)
    k = cov[0,1] / cov[0,0]
    return k


def calc_beam_pars(beam, energy_GeV=400, mass_GeV=proton_mass_GeV, correct_d=True):
    beam = copy(beam)
    pars = {}
    # try:
    #     gamma = beam.e.mean() / mass_GeV
    # except AttributeError:
    #     gamma = energy_GeV / mass_GeV
    pars['rel_gamma'] = energy_GeV / mass_GeV
    if 's' in beam.columns:
        pars['s'] = beam.s.mean()

    pars['sigx'] = beam.x.std()
    pars['sigy'] = beam.y.std()
    pars['sigpx'] = beam.px.std()
    pars['sigpy'] = beam.py.std()
    pars['dpp'] = beam.pt.std()

    pars['dx'] = calc_corr_factor(beam.pt, beam.x)
    pars['dy'] = calc_corr_factor(beam.pt, beam.y)
    pars['dpx'] = calc_corr_factor(beam.pt, beam.px)
    pars['dpy'] = calc_corr_factor(beam.pt, beam.py)
    
    
    if correct_d:
        beam.x -= pars['dx'] * beam.pt
        beam.y -= pars['dy'] * beam.pt
        beam.px -= pars['dpx'] * beam.pt
        beam.py -= pars['dpy'] * beam.pt
    
    beam.x -= beam.x.mean()
    beam.y -= beam.y.mean()
    beam.px -= beam.px.mean()
    beam.py -= beam.py.mean()
    beam.pt -= beam.pt.mean()

    
    covxpx = np.cov(beam.x, beam.px)
    covypy = np.cov(beam.y, beam.py)
    
    pars['geom_emitt_x'] = np.linalg.det(covxpx)**0.5
    pars['geom_emitt_y'] = np.linalg.det(covypy)**0.5
    pars['emitt_norm_x'] = pars['geom_emitt_x'] * gamma
    pars['emitt_norm_y'] = pars['geom_emitt_y'] * gamma
    pars['betx'] = beam.x.std()**2 / pars['geom_emitt_x']
    pars['bety'] = beam.y.std()**2 / pars['geom_emitt_y']
    pars['alfx'] = - covxpx[0, 1] / pars['geom_emitt_x']
    pars['alfy'] = - covypy[0, 1] / pars['geom_emitt_y']
    pars['gamx'] = beam.px.std()**2 / pars['geom_emitt_x']
    pars['gamy'] = beam.py.std()**2 / pars['geom_emitt_y']
    
    return pars

def calc_beam_pars_df(beam, energy_GeV=400, mass_GeV=proton_mass_GeV, correct_d=True):
    pars=[]
    for loc in beam.index.drop_duplicates():
        pars.append(calc_beam_pars(beam.loc[loc], energy_GeV, mass_GeV, correct_d))
    return pd.DataFrame(pars, index=beam.index.drop_duplicates())

def calculate_scr(beam, betx_max, bety_max, sigma_factor=7, orbit_error_m=5e-3, size_error_rel=0.1, dispersion_error_rel=0.1, alignment_error_m=1e-3):
    out = []
    for loc, b in beam.groupby('idx', sort=False):
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

def remove_dispersion(beam, dx=None, dpx=None, dy=None, dpy=None):
    beam = copy(beam)
    if dx is None:
        beam.x -= calc_corr_factor(beam.pt, beam.x) * beam.pt
    else:
        beam.x -= dx * beam.pt
    
    if dy is None:
        beam.y -= calc_corr_factor(beam.pt, beam.y) * beam.pt
    else:
        beam.y -= dy * beam.pt
    
    if dpx is None:
        beam.px -= calc_corr_factor(beam.pt, beam.px) * beam.pt
    else:
        beam.px -= dpx * beam.pt
    
    if dpy is None:
        beam.py -= calc_corr_factor(beam.pt, beam.py) * beam.pt
    else:
        beam.py -= dpy * beam.pt
    
    return beam

def remove_twiss_dispersion(beam, twiss, loc):
    tw = twiss.loc[loc]
    return remove_dispersion(beam, tw.dx, tw.dpx, tw.dy, tw.dpy)


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

def set_strength_error(madx, name, dkn=[]):
    madx.select(flag='error', clear=True)
    madx.select(flag='error', pattern=name)
    madx.input('EOPTION, ADD=False')
    madx.command.efcomp(dkn=dkn)


def normalize_coordinates(beam, pars=None, inplace=False):
    if not inplace:
        beam = copy(beam)
    if pars is None:
        pars = calc_beam_pars(beam)
    
    betx = pars['betx']
    bety = pars['bety']
    beam.x /= betx**0.5
    beam.y /= bety**0.5
    beam.px *= betx**0.5
    beam.py *= bety**0.5
    
    return beam


def to_human_units(beam, l_unit='mm', p_unit='mrad', pt_unit=1e-3):
    beam = copy(beam)
    beam.x /= units[l_unit] if type(l_unit) == str else l_unit
    beam.y /= units[l_unit] if type(l_unit) == str else l_unit
    beam.px /= units[p_unit] if type(p_unit) == str else p_unit
    beam.py /= units[p_unit] if type(p_unit) == str else p_unit
    beam.pt /= units[pt_unit] if type(pt_unit) == str else pt_unit
    return beam

