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
        particles_distribution = particles_distribution.sample(min(n_particles, len(particles_distribution.index)))
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