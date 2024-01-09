import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool
from copy import copy
from collections.abc import Iterable
import numpy as np

import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))

################## Lattice ##################
colors = {
    'quadrupole': 'mediumseagreen',
    'sextupole': 'blue',
    'octupole': 'red',
    'bend': 'salmon',
    'rbend': 'salmon',
    'lbend': 'salmon',
    'hkicker': 'purple',
    'vkicker': 'orange',
    'kicker': 'orange',
    'marker': 'blue',
    'rfcavity': 'wheat',
    'collimator': 'black',
    'monitor': 'gray',
}

fontsize={
    'title': 15, 
    'labels': 14, 
    'xticks': 10, 
    'yticks': 10,
}

vdims_names = ['color', 'keyword', 'name', 's', 'apertype', 'aper_1', 'aper_2', 'l', 'k0l', 'k1l','betx', 'bety', 'alfx', 'alfy', 'mux', 'muy']
tooltips = [(item, '@' + item) for item in vdims_names[1:]]
hover = HoverTool(tooltips=tooltips)

s_dim = hv.Dimension('s', unit='m')

default_style = dict(
    width=1400, 
    height=300, 
    color='color', 
    line_color='black', 
    line_width=0.5, 
    tools=[hover, 'tap'],
    fontsize=fontsize
)

def plot_lattice(
        twiss,
        range_=None,
        vdims_names=vdims_names, elem_height=0.1,
        color=colors, show_markers=False,
        marker_width = 0.3, marker_height = 0.4,
        twiss_plots=None,
        offsets=None,
        **kwargs):
    
    if range_:
        range_ = range_.split('/')
        twiss = copy(twiss.loc[range_[0]:range_[1]])
        twiss.s -= twiss.loc[range_[0]].s
    else:
        twiss = copy(twiss)
    
    twiss['color'] = [colors[kwrd] if kwrd in colors.keys() else None for kwrd in twiss.keyword]
    to_draw = twiss[(twiss['color'] >= '') & (twiss['aper_1'] > 0)]
    vdims = [to_draw[item] for item in vdims_names]
    # Apertures
    to_draw.loc[to_draw['apertype'] == 'circle', 'aper_2'] = to_draw[to_draw['apertype'] == 'circle']['aper_1'] # add vertical apertures for circle apertype

    to_draw['right'] = to_draw['s']
    to_draw['s'] = to_draw['s'] - to_draw['l']
    to_draw['xtopto'] = to_draw['aper_1'] + elem_height
    to_draw['xtopfrom'] = to_draw['aper_1']
    to_draw['xbotto'] = -to_draw['aper_1']
    to_draw['xbotfrom'] = - to_draw['aper_1'] - elem_height
    to_draw['ytopto'] = to_draw['aper_2'] + elem_height
    to_draw['ytopfrom'] = to_draw['aper_2']
    to_draw['ybotto'] = -to_draw['aper_2']
    to_draw['ybotfrom'] = - to_draw['aper_2'] - elem_height

    # Applying offsets
    if offsets:
        for key in offsets.keys():
            if key not in to_draw.index:
                continue
            to_draw.loc[key, 'xtopfrom'] += offsets[key][0]
            to_draw.loc[key, 'xbotto'] += offsets[key][0]
            to_draw.loc[key, 'ytopfrom'] += offsets[key][1]
            to_draw.loc[key, 'ybotto'] += offsets[key][1]
    
    # Delete repeating keys from default_style
    style = copy(default_style)
    for key in kwargs.keys():
        style.pop(key, None)

    options = opts.Rectangles(**style, **kwargs)

    lattice_top_x = hv.Rectangles(to_draw, kdims=[s_dim, 'xtopfrom', 'right', 'xtopto'], vdims=vdims_names)
    lattice_bottom_x = hv.Rectangles(to_draw, kdims=[s_dim, 'xbotfrom', 'right', 'xbotto'], vdims=vdims_names)
    lattice_top_y = hv.Rectangles(to_draw, kdims=[s_dim, 'ytopfrom', 'right', 'ytopto'], vdims=vdims_names)
    lattice_bottom_y = hv.Rectangles(to_draw, kdims=[s_dim, 'ybotfrom', 'right', 'ybotto'], vdims=vdims_names)

    top_plot = lattice_top_x * lattice_bottom_x
    bottom_plot = lattice_top_y * lattice_bottom_y
    
    if show_markers:
        markers = twiss[twiss.keyword == 'marker']
        vdims_markers = [markers[item] for item in vdims_names]
        markers_drawn = hv.Rectangles((markers['s'] - marker_width/2, 
                             -marker_height/2, 
                             markers['s'] + marker_width/2, 
                             marker_height/2,
                               *vdims_markers                          
                            ), vdims=vdims_names)
        
        top_plot *= markers_drawn
        bottom_plot *= markers_drawn
    
    lattice = (top_plot).opts(ylabel='x (m)') \
    + (bottom_plot).opts(ylabel='y (m)')

    twiss_plots = get_twiss_plots(twiss, kind=twiss_plots)
    for key in twiss_plots.keys():
        lattice += twiss_plots[key]
    
    return lattice.opts(options).cols(1)


################## Envelope ##################
area_core_style = dict(
    color = 'lightblue',
    alpha = 0.7
)

area_wings_style = dict(
    color = 'lightblue',
    alpha = 0.2
)

curve_style = dict(
    color = 'black',
    line_width = 0.1,
    tools = ['hover']
)

def plot_envelope(env, range_=None, area_core_style=area_core_style, area_wings_style=area_wings_style, curve_style=curve_style):
    if range_:
        cond = (env.s >= range_[0]) & (env.s <= range_[1])
        env = copy(env[cond])
        env.s -= range_[0]
    
    xp90 = hv.Area(env, s_dim, vdims=['xp5', 'xp95'], label='envelope-p90')
    xp99 = hv.Area(env, s_dim, vdims=['xp0_5', 'xp99_5'], label='envelope-p99')
    yp90 = hv.Area(env, s_dim, vdims=['yp5', 'yp95'], label='envelope-p90')
    yp99 = hv.Area(env, s_dim, vdims=['yp0_5', 'yp99_5'], label='envelope-p99')

    xp95 = hv.Curve(env, s_dim, 'xp5') * hv.Curve(env, s_dim, 'xp95')
    xp995 = hv.Curve(env, s_dim, 'xp0_5') * hv.Curve(env, s_dim, 'xp99_5')
    yp95 = hv.Curve(env, s_dim, 'yp5') * hv.Curve(env, s_dim, 'yp95')
    yp995 = hv.Curve(env, s_dim, 'yp0_5') * hv.Curve(env, s_dim, 'yp99_5')

    curve_opts = opts.Curve(**curve_style)
    area_core_opts = opts.Area(**area_core_style)
    area_wings_opts = opts.Area(**area_wings_style)

    envelope_x = xp95 * xp995
    envelope_y = yp95 * yp995

    area_x = xp99.opts(area_wings_opts) * xp90.opts(area_core_opts)
    area_y = yp99.opts(area_wings_opts) * yp90.opts(area_core_opts)
    envelope = (area_x * envelope_x + area_y * envelope_y).opts(curve_opts).cols(1)
    return envelope


################## Twiss ##################
default_twiss_keys = ['betx', 'bety', 'alfx', 'alfy', 'mux', 'muy']

twiss_style = dict(
    width=1400, 
    height=300,  
    tools=['hover', 'tap'],
    fontsize=fontsize
)

def get_twiss_plots(twiss, kind='beta'):
    # if range_:
    #     range_ = range_.split('/')
    #     twiss = copy(twiss.loc[range_[0]:range_[1]])
    #     twiss.s -= twiss.loc[range_[0]].s
    # else:
    #     twiss = copy(twiss)
    if not isinstance(kind, Iterable):
        kind = [kind]
    plots = {}
    if 'beta' in kind:
        beta = hv.Curve(twiss, s_dim, vdims=['betx', 'name'], label='betx') * hv.Curve(twiss, s_dim, vdims=['bety', 'name'], label='bety')
        plots['beta'] = beta.opts(opts.Curve(**twiss_style)).opts(ylabel='beta (m)')
    if 'alpha' in kind:
        alpha = hv.Curve(twiss, s_dim, vdims=['alfx', 'name'], label='alfx') * hv.Curve(twiss, s_dim, vdims=['alfy', 'name'], label='alfy')
        plots['alpha'] = alpha.opts(opts.Curve(**twiss_style)).opts(ylabel='alpha')
    if 'mu' in kind:
        twiss['mux_norm'] = twiss.mux * np.pi / 2.
        twiss['muy_norm'] = twiss.muy * np.pi / 2.
        mu = hv.Curve(twiss, s_dim, vdims=['mux_norm', 'name'], label='mux') * hv.Curve(twiss, s_dim, vdims=['muy_norm', 'name'], label='muy')
        plots['mu'] = mu.opts(opts.Curve(**twiss_style)).opts(ylabel='2 mu / pi')
    if 'd' in kind:
        alpha = hv.Curve(twiss, s_dim, vdims=['dx', 'name'], label='Dx') * hv.Curve(twiss, s_dim, vdims=['dy', 'name'], label='Dy')
        plots['D'] = alpha.opts(opts.Curve(**twiss_style)).opts(ylabel='D (m)')
    return plots

def plot_twiss(twiss, kind='beta', range_=None, **kwargs):
    if range_:
        range_ = range_.split('/')
        twiss = copy(twiss.loc[range_[0]:range_[1]])
        twiss.s -= twiss.loc[range_[0]].s
    else:
        twiss = copy(twiss)
    plots = get_twiss_plots(twiss, kind=kind)
    for i, key in enumerate(plots.keys()):
        if i == 0:
            layout = plots[key]
        else:
            layout += plots[key]
    return layout.opts(**kwargs).cols(1)

################## All ##################
def plot_all(twiss, env, range_=None, twiss_plots=None, offsets=None, **kwargs):
    if range_:
        range_env=range_.split('/')
        s1 = twiss.loc[range_env[0]].s
        s2 = twiss.loc[range_env[1]].s
        range_env= [s1, s2]
    else:
        range_env = None
    lattice = plot_lattice(twiss, range_=range_, twiss_plots=twiss_plots, offsets=offsets, **kwargs)
    envelope = plot_envelope(env, range_=range_env)
    x_opts = dict(xlabel='s (m)', ylabel='x (m)')
    y_opts = dict(xlabel='s (m)', ylabel='y (m)')

    layout = (lattice[0] * envelope[0]).opts(**x_opts) + (lattice[1] * envelope[1]).opts(**y_opts)

    for i in range(len(lattice)):
        if i > 1:
            layout += lattice[i]

    return layout.cols(1)

