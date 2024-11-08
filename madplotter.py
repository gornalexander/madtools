import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool
from copy import copy
from collections.abc import Iterable
import numpy as np
import madutils as mu

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
    'rcollimator': 'black',
    'monitor': 'gray',
}

colors_tilted = {
    'quadrupole': 'mediumseagreen',
    'sextupole': 'blue',
    'octupole': 'red',
    'bend': 'indianred',
    'rbend': 'indianred',
    'lbend': 'indianred',
    'hkicker': 'purple',
    'vkicker': 'orange',
    'kicker': 'orange',
    'marker': 'blue',
    'rfcavity': 'wheat',
    'collimator': 'black',
    'rcollimator': 'black',
    'monitor': 'gray',
}

fontsize={
    'title': 15, 
    'labels': 14, 
    'xticks': 10, 
    'yticks': 10,
}

vdims_names = ['color', 'keyword', 'name', 's', 'apertype', 'aper_1', 'aper_2', 'l', 'tilt', 'angle', 'k1l', 'betx', 'bety', 'alfx', 'alfy', 'mux', 'muy']
tooltips = [(item, '@' + item) for item in vdims_names[1:]]
hover = HoverTool(tooltips=tooltips)

s_dim = hv.Dimension('s', unit='m')

default_style = dict(
    width=800, 
    height=250, 
    color='color', 
    line_color='black', 
    line_width=0.5, 
    tools=[hover, 'tap'],
    fontsize=fontsize
)

def plot_lattice(
        twiss,
        range_=None,
        vdims_names=vdims_names, elem_height=0.15,
        color=colors, show_markers=False,
        marker_width = 0.3, marker_height = 0.4,
        twiss_plots=None,
        offsets=None,
        filter_elements=None,
        **kwargs):
    
    if range_:
        range_ = range_.split('/')
        twiss = copy(twiss.loc[range_[0]:range_[1]])
        twiss.s -= twiss.loc[range_[0]].s
    else:
        twiss = copy(twiss)
    
    if filter_elements:
        twiss = twiss.filter(**filter_elements)
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
            to_draw.loc[key, 'right'] += offsets[key][2]
            to_draw.loc[key, 's'] += offsets[key][2]
    
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

SYNOPTIC_ELEM_HIGHT = 1
SYNOPTIC_YLIM = (-SYNOPTIC_ELEM_HIGHT * 1.2, SYNOPTIC_ELEM_HIGHT * 3)
SYNOPTIC_MARKER_WIDTH = 0.1
SYNOPTIC_HOVER_NAMES = ['color', 'keyword', 'name', 's', 'apertype', 'aper_1', 'aper_2', 'l', 'tilt', 'angle', 'k1l']
SYNOPTIC_TOOLTIPS = [(item, '@' + item) for item in SYNOPTIC_HOVER_NAMES[1:]]
SYNOPTIC_STYLE = dict(
    width=800, 
    height=100, 
    color='color', 
    #alpha='alpha',
    line_color='black', 
    line_width=0.5, 
    tools=[HoverTool(tooltips=SYNOPTIC_TOOLTIPS, anchor = 'center'), 'tap'],
    fontsize=fontsize
)

def plot_synoptic(twiss, range_=None, 
                  show_markers=False, 
                  show_names=False,
                  show_axes=False,
                  **kwargs):
    if range_:
        range_ = range_.split('/')
        twiss = copy(twiss.loc[range_[0]:range_[1]])
        twiss.s -= twiss.loc[range_[0]].s
    else:
        twiss = copy(twiss)
    
    twiss['color'] = [colors[kwrd] if kwrd in colors.keys() else None for kwrd in twiss.keyword]
    to_draw = twiss[twiss['color'] >= '']
    to_draw['color'] = [colors_tilted[el.keyword] if el.tilt != 0 else colors[el.keyword] for idx, el in to_draw.iterrows()]

    to_draw['right'] = to_draw['s']
    to_draw['s'] = to_draw['s'] - to_draw['l']
    to_draw['bottom'] = -SYNOPTIC_ELEM_HIGHT
    to_draw['top'] = SYNOPTIC_ELEM_HIGHT
    correct_bottom = np.where(to_draw['k1l'] > 0, SYNOPTIC_ELEM_HIGHT, 0)
    correct_top = np.where(to_draw['k1l'] < 0, -SYNOPTIC_ELEM_HIGHT, 0)
    to_draw['bottom'] += correct_bottom
    to_draw['top'] += correct_top
    to_draw['text'] = to_draw['name'].apply(lambda x: x.split('.')[0].upper())

    synoptic = hv.HLine(0).opts(color='black', line_width=0.5, alpha=0.5)

    # Delete repeating keys from default_style
    style = copy(SYNOPTIC_STYLE)
    for key in kwargs.keys():
        style.pop(key, None)
    # style['alpha'] = 'alpha'
    options = opts.Rectangles(**style, **kwargs)

    synoptic *= hv.Rectangles(to_draw, kdims=[s_dim, 'bottom', 'right', 'top'], vdims=SYNOPTIC_HOVER_NAMES)
    
    if show_markers:
        markers = twiss[twiss.keyword == 'marker']
        vdims_markers = [markers[item] for item in SYNOPTIC_HOVER_NAMES]
        markers_drawn = hv.Rectangles((markers['s'] - SYNOPTIC_MARKER_WIDTH/2, 
                             -SYNOPTIC_ELEM_HIGHT, 
                             markers['s'] + SYNOPTIC_MARKER_WIDTH/2, 
                             SYNOPTIC_ELEM_HIGHT,
                               *vdims_markers                          
                            ), vdims=SYNOPTIC_HOVER_NAMES)
        
        synoptic *= markers_drawn
    if show_names:
        for idx, el in to_draw.iterrows():
            synoptic = synoptic * hv.Text(el.s + el.l/2, 
                                          SYNOPTIC_ELEM_HIGHT * 1.2, 
                                          el.text, 
                                          halign='left', 
                                          valign='center', 
                                          rotation=60,
                                          fontsize=6,
                                          ).options(text_font_style='bold', text_color='grey')
    if not show_axes:
        def hide_hook(plot, element):
            plot.handles["xaxis"].visible = False
            plot.handles["yaxis"].visible = False 
            # plot.handles["xgrid"].visible = False
            # plot.handles["ygrid"].visible = False
            #plot.handles["plot"].border_fill_color = None
            #plot.handles["plot"].background_fill_color = None
            plot.handles["plot"].outline_line_color = None
        
        synoptic = synoptic.opts(hooks=[hide_hook])

    return synoptic.opts(ylim=SYNOPTIC_YLIM).opts(labelled=[]).opts(options)

################## Envelope ##################
area_sigma_style = dict(
    color = 'lightblue',
    alpha = 1
)

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
    
    env['envx_up'] = env['sx'] + env['meanx']
    env['envx_low'] = -env['sx'] + env['meanx']
    env['envy_up'] = env['sy'] + env['meany']
    env['envy_low'] = -env['sy'] + env['meany']
    
    sx_fill = hv.Area(env, s_dim, vdims=['envx_up', 'envx_low'], label='envelope-sigma')
    sy_fill = hv.Area(env, s_dim, vdims=['envy_up', 'envy_low'], label='envelope-sigma')

    sx = hv.Curve(env, s_dim, 'envx_up') * hv.Curve(env, s_dim, 'envx_low')
    sy = hv.Curve(env, s_dim, 'envy_up') * hv.Curve(env, s_dim, 'envy_low')
    
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
    area_sigma_opts = opts.Area(**area_sigma_style)

    envelope_x = xp95 * xp995 * sx
    envelope_y = yp95 * yp995 * sy

    area_x = xp99.opts(area_wings_opts) * xp90.opts(area_core_opts) * sx_fill.opts(area_sigma_opts)
    area_y = yp99.opts(area_wings_opts) * yp90.opts(area_core_opts) * sy_fill.opts(area_sigma_opts)
    envelope = (area_x * envelope_x + area_y * envelope_y).opts(curve_opts).cols(1)
    return envelope

################## SCR ##################
scr_area_style = dict(
    color = 'red',
    alpha = 0.3
)
scr_contour_style = dict(
    color = 'red',
    line_width = 0.5,
    tools=['hover']
)
def plot_scr(scr_data, range_=None, **kwargs):
    if range_:
        range_ = range_.split('/')
        scr_data = copy(scr_data.loc[range_[0]:range_[1]])
        scr_data.s -= scr_data.loc[range_[0]].s
    
    scrx_fill = hv.Area(scr_data, s_dim, vdims=['scr_x_low', 'scr_x_up'], label='SCR')
    scry_fill = hv.Area(scr_data, s_dim, vdims=['scr_y_low', 'scr_y_up'], label='SCR')
    scrx = hv.Curve(scr_data, s_dim, 'scr_x_up') * hv.Curve(scr_data, s_dim, 'scr_x_low')
    scry = hv.Curve(scr_data, s_dim, 'scr_y_up') * hv.Curve(scr_data, s_dim, 'scr_y_low')

    area_opts = opts.Area(**scr_area_style)
    contour_opts = opts.Curve(**scr_contour_style)
    layout_x = (scrx_fill * scrx).opts(xlabel='s (m)', ylabel='x (m)')
    layout_y = (scry_fill * scry).opts(xlabel='s (m)', ylabel='y (m)')
    scr = (layout_x + layout_y).opts(contour_opts).opts(area_opts).cols(1)
    return scr



################## Twiss ##################
default_twiss_keys = ['betx', 'bety', 'alfx', 'alfy', 'mux', 'muy']

twiss_style = dict(
    width=800, 
    height=250,  
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
def plot_all(twiss, env, range_=None, twiss_plots=None, offsets=None, synoptic=False, show_scr=None, **kwargs):
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

    if show_scr:
        scr = plot_scr(env, range_=range_)
        layout =\
            (scr[0] * lattice[0] * envelope[0]).opts(**x_opts) +\
            (scr[1] * lattice[1] * envelope[1]).opts(**y_opts)
    else:
        layout =\
            (lattice[0] * envelope[0]).opts(**x_opts) +\
            (lattice[1] * envelope[1]).opts(**y_opts)

    if synoptic:
        synoptic_kwargs = {} if not isinstance(synoptic, dict) else synoptic
        synoptic = plot_synoptic(twiss, range_=range_, **synoptic_kwargs)
        layout = synoptic + layout

    for i in range(len(lattice)): # Add twiss plots to the layout
        if i > 1:
            layout += lattice[i]

    return layout.cols(1)

def plot_beam_within_aperture(
        particles, 
        twiss, 
        loc, 
        options=dict(width=400, height=400, tools=['hover']),
        xlim=(-0.1, 0.1),
        ylim=(-0.1, 0.1)):
    beam = particles.loc[loc]
    aper_1 = twiss.loc[loc].aper_1
    aper_2 = twiss.loc[loc].aper_2
    apertype = twiss.loc[loc].apertype
    if apertype == 'circle':
        aperture = hv.Ellipse(0, 0, aper_1*2).opts(line_color='black', color=None)
        aper_2 = aper_1
    elif apertype == 'ellipse':
        aperture = hv.Ellipse(0, 0, (aper_1*2, aper_2*2)).opts(line_color='black', color=None)
    elif apertype == 'rectangle':
        aperture = hv.Rectangles((-aper_1, -aper_2, aper_1, aper_2)).opts(line_color='black', color=None)
    else:
        raise ValueError('apertype not supported')
    beam_distribution = hv.Scatter(beam, 'x', 'y', label='beam').opts(size=0.5, **options)
    beam_envelope = hv.Ellipse(beam.x.mean(), beam.y.mean(), (2*3*beam.x.std(), 2*3*beam.y.std()), label='beam envelope (3 sigma)')
    scr_data = mu.calculate_scr(beam, betx_max=twiss.loc[loc].betx, bety_max=twiss.loc[loc].bety)
    scr = hv.Rectangles((scr_data['scr_x_low'], scr_data['scr_y_low'], scr_data['scr_x_up'], scr_data['scr_y_up']), label='SCR').opts(color='red', alpha=0.3)
    beam_and_aperture =\
    aperture.opts(**options) *\
    scr *\
    beam_envelope.opts(color='blue', line_color='blue', alpha=0.5) *\
    beam_distribution

    beam_and_aperture = beam_and_aperture.opts(
        xlim=xlim,
        ylim=ylim, 
        fontsize=fontsize,
        xlabel='x (m)', 
        ylabel='y (m)')
    
    return beam_and_aperture