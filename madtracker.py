import numpy as np
import pandas as pd
from copy import copy

def track_line_ptc(
        madx,
        particles_in,
        sequence=False,
        seq_range="#s/#e",
        observe=None,
        model=2,
        method=4,
        nst=4,
        exact=True,
        icase=5,
        onetable=True,
        recloss=True,
        norm_no=4,
        **kwargs,
    ):
        parts_in = particles_in.copy()
        true_idx = np.array(parts_in.number)
        if sequence:
            madx.use(sequence=sequence, range_=seq_range)
        madx.ptc_create_universe(ntpsa=True)
        madx.ptc_create_layout(model=model, method=method, nst=nst, exact=exact)
        madx.ptc_align()
        # Tracking in ptc:

        for idx, part in parts_in.iterrows():
                madx.ptc_start(
                    x=part.x,
                    px=part.px,
                    y=part.y,
                    py=part.py,
                    t=0.0,
                    pt=part.pt,
                )

        if observe:
            if isinstance(observe, (list, tuple)):
                for ele in observe:
                    madx.ptc_observe(place=ele)
            else:
                madx.ptc_observe(place=observe)

        madx.ptc_track(
            icase=icase,
            onetable=onetable,
            recloss=recloss,
            norm_no=norm_no,
            **kwargs,
        )
        madx.ptc_track_end()

        try:
            parts_out = madx.table.trackone.dframe()
        except KeyError:
            parts_out = pd.DataFrame()
        else:
            parts_out = parts_out[parts_out["turn"] == 1.0]
            parts_out["number"] = [true_idx[int(n) - 1] for n in parts_out.number]
        try:
            parts_lost = madx.table.trackloss.dframe()
        except KeyError:
            parts_lost = pd.DataFrame()
        else:
            parts_lost["number"] = [true_idx[int(n) - 1] for n in parts_lost.number]

        return parts_out, parts_lost

def transfer_beam(
        madx, 
        particles_distribution,
        init_cond,
        n_particles = 5000,
        observe='all',
        sequence=None, 
        range_=None,
        ):
    if sequence:
        madx.use(sequence=sequence, range_=range_)
    twiss = madx.twiss(**init_cond, rmatrix = True).dframe()
    
    if n_particles:
        particles_distribution = particles_distribution.sample(n_particles)
    particles_distribution["number"] = particles_distribution.index

    if observe == 'all':
        observe = list(twiss.index)
    elif observe == 'last':
        observe = list(twiss.index)[-1]
        
    particles, losses = track_line_ptc(
        madx=madx,
        particles_in=particles_distribution,
        observe=observe
    )
    return particles, twiss

    
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