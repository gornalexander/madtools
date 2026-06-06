"""Matplotlib-based plotting for MAD-X lattice, envelope, and beam data.

Mirrors madplotter.py (holoviews/bokeh) with matching public signatures.
All multi-panel functions accept an optional ``axes`` argument so they can be
embedded into an existing figure; when omitted a new figure is created.
Returns ``(fig, axes)`` tuples instead of holoviews layout objects.
Hover tooltips are provided by the optional ``mplcursors`` package — the
plot still works without it, just without interactivity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import copy
from collections.abc import Iterable
import os
import imageio
from glob import glob

import pandas as pd
from IPython.display import display, Image
from ipywidgets import interact, widgets

from . import madutils as mu

plt.rc('figure', titlesize=10)
plt.rc('axes', titlesize=8)

# ──────────────────────────────────────────────────────────────────────────────
# Element colour maps  (identical to madplotter.py)
# ──────────────────────────────────────────────────────────────────────────────

colors = {
    'quadrupole': 'mediumseagreen',
    'sextupole':  'blue',
    'octupole':   'red',
    'bend':       'salmon',
    'rbend':      'salmon',
    'lbend':      'salmon',
    'hkicker':    'purple',
    'vkicker':    'orange',
    'kicker':     'orange',
    'marker':     'blue',
    'rfcavity':   'wheat',
    'collimator': 'black',
    'rcollimator':'black',
    'monitor':    'gray',
}

colors_tilted = {
    **colors,
    'bend': 'indianred', 'rbend': 'indianred', 'lbend': 'indianred',
}

# ──────────────────────────────────────────────────────────────────────────────
# Default styles
# ──────────────────────────────────────────────────────────────────────────────

area_sigma_style  = dict(color='lightblue', alpha=1.0)
area_core_style   = dict(color='lightblue', alpha=0.7)
area_wings_style  = dict(color='lightblue', alpha=0.2)
curve_style       = dict(color='black', linewidth=0.5)

scr_area_style    = dict(color='red', alpha=0.3)
scr_contour_style = dict(color='red', linewidth=0.5)

scatter_style = dict(s=2)
fit_style     = dict(ls='--', color='k')

_SYNOPTIC_ELEM_HEIGHT  = 1
_SYNOPTIC_YLIM         = (-_SYNOPTIC_ELEM_HEIGHT * 1.2, _SYNOPTIC_ELEM_HEIGHT * 3)
_SYNOPTIC_MARKER_WIDTH = 0.1
# Synoptic height ratio relative to regular panels (matches holoviews 100:250)
_SYNOPTIC_HEIGHT_RATIO = 0.4
_FIGWIDTH = 9.6

# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _attach_hover(artists, info_dict):
    """Attach mplcursors hover tooltips to patch artists. Silent no-op if not installed."""
    try:
        import mplcursors
        cursor = mplcursors.cursor(artists, hover=True)

        @cursor.connect('add')
        def on_add(sel):
            sel.annotation.set_text(info_dict.get(sel.artist, ''))
            sel.annotation.get_bbox_patch().set(fc='lightyellow', alpha=0.95)
    except ImportError:
        pass


# Units for each plotted quantity (empty string = dimensionless).
# Source: MAD-X User Manual, conventions.tex / elements.tex
_LABEL_UNITS = {
    r'$\beta_x$':   'm',
    r'$\beta_y$':   'm',
    r'$\alpha_x$':  '',
    r'$\alpha_y$':  '',
    r'$\mu_x$':     '',   # plotted as 2μ/π — dimensionless
    r'$\mu_y$':     '',
    r'$D_x$':       'm',
    r'$D_y$':       'm',
    # envelope edges
    r'$x_{p95}$':   'm',  r'$x_{p5}$':    'm',
    r'$x_{p99.5}$': 'm',  r'$x_{p0.5}$':  'm',
    r'$+\sigma$':   'm',  r'$-\sigma$':   'm',
    # SCR
    'SCR up':  'm',
    'SCR low': 'm',
}


def _elem_at_s(twiss, s):
    """Return cleaned element name whose body contains position s [m]."""
    if twiss is None or 'l' not in twiss.columns:
        return None
    # element spans [s-l, s] — find the one that brackets the cursor position
    mask = (twiss['s'] - twiss['l'] <= s + 1e-6) & (twiss['s'] >= s - 1e-6)
    if mask.any():
        return _clean_name(twiss[mask].iloc[-1]['name'])
    # fallback: nearest element end
    return _clean_name(twiss.iloc[(twiss['s'] - s).abs().values.argmin()]['name'])


def _attach_hover_curves(lines, twiss=None):
    """Attach mplcursors hover to line artists.

    Shows: element name at cursor position, s coordinate, and the
    plotted value with its MAD-X unit.
    """
    visible = [l for l in lines if not l.get_label().startswith('_')]
    if not visible:
        return
    try:
        import mplcursors
        cursor = mplcursors.cursor(visible, hover=True)

        @cursor.connect('add')
        def on_add(sel):
            label = sel.artist.get_label()
            x, y = sel.target
            unit = _LABEL_UNITS.get(label, 'm')
            val_str = f'{label} = {y:.4f} {unit}' if unit else f'{label} = {y:.4f}'

            elem = _elem_at_s(twiss, x)
            parts = [f's = {x:.3f} m']
            if elem:
                parts.append(f'elem = {elem}')
            parts.append(val_str)

            sel.annotation.set_text('\n'.join(parts))
            sel.annotation.get_bbox_patch().set(fc='lightyellow', alpha=0.95)
    except ImportError:
        pass


def _clean_name(name):
    """Strip ':N' cpymad suffix and capitalise."""
    if ':' in name:
        name = name.split(':')[0]
    return name.upper()


def _build_tooltip(row):
    name = _clean_name(row['name'])
    kw   = row.keyword
    s_end = row['right'] if 'right' in row.index else row['s']

    lines = [
        f"{name}  ({kw})",
        f"s = {s_end:.3f} m",
        f"L = {row.l:.3f} m",
        f"aper_1 = {row.aper_1*1e3:.1f} mm",
        f"aper_2 = {row.aper_2*1e3:.1f} mm",
    ]
    if 'k1l' in row.index and row.k1l != 0:
        lines.append(f"k1l = {row.k1l:.5f}")
    if 'angle' in row.index and row.angle != 0:
        lines.append(f"angle = {row.angle*1e3:.4f} mrad")
    if kw in ('hkicker', 'vkicker', 'kicker'):
        for col, label in (('hkick', 'hkick'), ('vkick', 'vkick'), ('k0l', 'k0l')):
            if col in row.index and row[col] != 0:
                lines.append(f"{label} = {row[col]*1e6:.2f} μrad")
    return '\n'.join(lines)


def _slice_twiss(twiss, range_):
    """Slice twiss by 'name_start/name_end' and reset s to 0 at slice start."""
    if range_:
        parts = range_.split('/')
        twiss = copy(twiss.loc[parts[0]:parts[1]])
        twiss = twiss.copy()
        twiss['s'] = twiss['s'] - twiss['s'].iloc[0]
    else:
        twiss = copy(twiss)
    return twiss


def _slice_env(env, range_):
    """Slice envelope by numeric [s_start, s_end] and reset s to 0."""
    if range_:
        cond = (env.s >= range_[0]) & (env.s <= range_[1])
        env = copy(env[cond])
        env = env.copy()
        env['s'] = env['s'] - range_[0]
    else:
        env = copy(env)
    return env


def _prepare_to_draw(twiss, color_map, elem_height, marker_width, offsets, filter_elements):
    """Build the drawing DataFrame for plot_lattice / plot_synoptic."""
    twiss = twiss.copy()
    if filter_elements:
        twiss = twiss.filter(**filter_elements)

    twiss['color'] = [color_map.get(k) for k in twiss.keyword]
    circle = twiss['apertype'] == 'circle'
    twiss.loc[circle, 'aper_2'] = twiss.loc[circle, 'aper_1']

    to_draw = twiss[twiss['color'].notna() & (twiss['aper_1'] > 0)].copy()

    to_draw['right'] = to_draw['s']
    to_draw['s'] = to_draw['s'] - np.maximum(to_draw['l'].values,
                                              np.full(len(to_draw), marker_width))
    to_draw['xtopto']   =  to_draw['aper_1'] + elem_height
    to_draw['xtopfrom'] =  to_draw['aper_1']
    to_draw['xbotto']   = -to_draw['aper_1']
    to_draw['xbotfrom'] = -to_draw['aper_1'] - elem_height
    to_draw['ytopto']   =  to_draw['aper_2'] + elem_height
    to_draw['ytopfrom'] =  to_draw['aper_2']
    to_draw['ybotto']   = -to_draw['aper_2']
    to_draw['ybotfrom'] = -to_draw['aper_2'] - elem_height

    if offsets:
        for key, (ox, oy, os_) in offsets.items():
            if key not in to_draw.index:
                continue
            for col in ('xtopfrom', 'xtopto', 'xbotfrom', 'xbotto'):
                to_draw.loc[key, col] += ox
            for col in ('ytopfrom', 'ytopto', 'ybotfrom', 'ybotto'):
                to_draw.loc[key, col] += oy
            to_draw.loc[key, 's']     += os_
            to_draw.loc[key, 'right'] += os_

    return to_draw


def _draw_aperture_patches(ax, to_draw, top_from, top_to, bot_from, bot_to, patch_info):
    """Draw top+bottom aperture blocks for one plane; populate patch_info for hover."""
    patches = []
    for _, row in to_draw.iterrows():
        w = row.right - row.s
        if w <= 0:
            continue
        for y0, y1 in [(row[top_from], row[top_to]),
                       (row[bot_from], row[bot_to])]:
            p = mpatches.Rectangle(
                (row.s, y0), w, y1 - y0,
                facecolor=row.color, edgecolor='black', linewidth=0.5, zorder=3,
            )
            ax.add_patch(p)
            patches.append(p)
            patch_info[p] = _build_tooltip(row)
    ax.autoscale_view()
    return patches


def _draw_marker_patches(ax, markers, marker_width, marker_height, patch_info):
    """Draw thin rectangles at marker positions."""
    patches = []
    for _, row in markers.iterrows():
        p = mpatches.Rectangle(
            (row.s - marker_width / 2, -marker_height / 2),
            marker_width, marker_height,
            facecolor='blue', edgecolor='black', linewidth=0.5, zorder=4,
        )
        ax.add_patch(p)
        patches.append(p)
        patch_info[p] = f"{_clean_name(row['name'])}  (marker)\ns = {row.s:.3f} m"
    return patches


def _make_fig(n_panels, figsize, height_ratios=None):
    if figsize is None:
        total = sum(height_ratios) if height_ratios else n_panels
        figsize = (_FIGWIDTH, total * 2.4)
    gskw = {'height_ratios': height_ratios} if height_ratios else {}
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True,
                             gridspec_kw=gskw)
    return fig, (list(axes) if n_panels > 1 else [axes])


# ──────────────────────────────────────────────────────────────────────────────
# Synoptic
# ──────────────────────────────────────────────────────────────────────────────

def plot_synoptic(twiss, range_=None, show_markers=False, show_names=False,
                  show_axes=False, ax=None, figsize=None, **kwargs):
    """Compact element-type overview (one panel, no aperture data required).

    Parameters
    ----------
    twiss : pd.DataFrame
    range_ : str, optional
        'elem_start/elem_end'
    show_markers : bool
    show_names : bool
        Annotate element names rotated 60°.
    show_axes : bool
        Show x/y axis ticks and labels (hidden by default).
    ax : Axes, optional
    figsize : tuple, optional

    Returns
    -------
    fig, ax
    """
    twiss = _slice_twiss(twiss, range_)
    twiss = twiss.copy()
    twiss['color'] = [colors.get(k) for k in twiss.keyword]

    to_draw = twiss[twiss['color'].notna()].copy()
    to_draw['color'] = [
        colors_tilted[el.keyword] if el.tilt != 0 else colors[el.keyword]
        for _, el in to_draw.iterrows()
    ]
    to_draw['s_start'] = to_draw['s'] - to_draw['l']
    to_draw['bottom'] = -_SYNOPTIC_ELEM_HEIGHT
    to_draw['top']    =  _SYNOPTIC_ELEM_HEIGHT
    correct_bottom = np.where(to_draw['k1l'] > 0,  _SYNOPTIC_ELEM_HEIGHT, 0)
    correct_top    = np.where(to_draw['k1l'] < 0, -_SYNOPTIC_ELEM_HEIGHT, 0)
    to_draw['bottom'] += correct_bottom
    to_draw['top']    += correct_top

    if ax is None:
        if figsize is None:
            figsize = (_FIGWIDTH, 1.2)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    patch_info = {}
    patches = []
    for _, row in to_draw.iterrows():
        w = row.l if row.l > 0 else _SYNOPTIC_MARKER_WIDTH
        p = mpatches.Rectangle(
            (row.s_start, row.bottom), w, row.top - row.bottom,
            facecolor=row.color, edgecolor='black', linewidth=0.5, zorder=2,
        )
        ax.add_patch(p)
        patches.append(p)
        tip_lines = [f"{_clean_name(row['name'])}  ({row.keyword})",
                     f"s = {row.s:.3f} m",
                     f"L = {row.l:.3f} m"]
        if row.k1l != 0:
            tip_lines.append(f"k1l = {row.k1l:.5f}")
        if 'angle' in row.index and row.angle != 0:
            tip_lines.append(f"angle = {row.angle*1e3:.4f} mrad")
        patch_info[p] = '\n'.join(tip_lines)

    if show_markers:
        markers = twiss[twiss.keyword == 'marker']
        for _, row in markers.iterrows():
            p = mpatches.Rectangle(
                (row.s - _SYNOPTIC_MARKER_WIDTH / 2, -_SYNOPTIC_ELEM_HEIGHT),
                _SYNOPTIC_MARKER_WIDTH, 2 * _SYNOPTIC_ELEM_HEIGHT,
                facecolor='blue', edgecolor='black', linewidth=0.5, zorder=3,
            )
            ax.add_patch(p)
            patches.append(p)
            patch_info[p] = f"{_clean_name(row['name'])}  (marker)\ns = {row.s:.3f} m"

    if show_names:
        for _, row in to_draw.iterrows():
            label = row['name'].split('.')[0].upper()
            ax.text(row.s_start + row.l / 2, _SYNOPTIC_ELEM_HEIGHT * 1.2,
                    label, rotation=60, fontsize=6, color='grey',
                    fontweight='bold', ha='left', va='center')

    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.autoscale_view()
    ax.set_ylim(_SYNOPTIC_YLIM)

    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    _attach_hover(patches, patch_info)
    ax.set(**{k: v for k, v in kwargs.items() if k in ('xlim',)})
    return fig, ax


# ──────────────────────────────────────────────────────────────────────────────
# Twiss
# ──────────────────────────────────────────────────────────────────────────────

def get_twiss_plots(twiss, kind='beta'):
    """Return a dict of ``{kind: draw_fn(ax)}`` callables for Twiss parameters.

    Supported kinds: 'beta', 'alpha', 'mu', 'd'.
    Each callable draws onto the given Axes, sets LaTeX ylabel, adds legend,
    and attaches mplcursors hover to each curve.

    Parameters
    ----------
    twiss : pd.DataFrame
    kind : str or list of str

    Returns
    -------
    dict[str, callable]
    """
    if isinstance(kind, str):
        kind = [kind]
    plots = {}

    if 'beta' in kind:
        def _beta(ax):
            lx, = ax.plot(twiss.s, twiss.betx, label=r'$\beta_x$')
            ly, = ax.plot(twiss.s, twiss.bety, label=r'$\beta_y$')
            ax.set_ylabel(r'$\beta$ (m)')
            ax.legend(fontsize=8)
            _attach_hover_curves([lx, ly], twiss)
        plots['beta'] = _beta

    if 'alpha' in kind:
        def _alpha(ax):
            lx, = ax.plot(twiss.s, twiss.alfx, label=r'$\alpha_x$')
            ly, = ax.plot(twiss.s, twiss.alfy, label=r'$\alpha_y$')
            ax.set_ylabel(r'$\alpha$')
            ax.legend(fontsize=8)
            _attach_hover_curves([lx, ly], twiss)
        plots['alpha'] = _alpha

    if 'mu' in kind:
        def _mu(ax):
            lx, = ax.plot(twiss.s, twiss.mux * np.pi / 2, label=r'$\mu_x$')
            ly, = ax.plot(twiss.s, twiss.muy * np.pi / 2, label=r'$\mu_y$')
            ax.set_ylabel(r'$2\mu/\pi$')
            ax.legend(fontsize=8)
            _attach_hover_curves([lx, ly], twiss)
        plots['mu'] = _mu

    if 'd' in kind:
        def _disp(ax):
            lx, = ax.plot(twiss.s, twiss.dx, label=r'$D_x$')
            ly, = ax.plot(twiss.s, twiss.dy, label=r'$D_y$')
            ax.set_ylabel(r'$D$ (m)')
            ax.legend(fontsize=8)
            _attach_hover_curves([lx, ly], twiss)
        plots['d'] = _disp

    return plots


def plot_twiss(twiss, kind='beta', range_=None, axes=None, figsize=None, **kwargs):
    """Standalone Twiss parameter plot.

    Parameters
    ----------
    twiss : pd.DataFrame
    kind : str or list of str
        Subset of ['beta', 'alpha', 'mu', 'd'].
    range_ : str, optional
    axes : list of Axes, optional
    figsize : tuple, optional

    Returns
    -------
    fig, axes
    """
    twiss = _slice_twiss(twiss, range_)
    if isinstance(kind, str):
        kind = [kind]
    plots = get_twiss_plots(twiss, kind=kind)

    if axes is None:
        fig, axes = _make_fig(len(plots), figsize)
    else:
        fig = axes[0].get_figure()

    for ax, (k, draw_fn) in zip(axes, plots.items()):
        draw_fn(ax)
        ax.set(**{kw: v for kw, v in kwargs.items() if kw in ('xlim', 'title')})

    axes[-1].set_xlabel('s (m)')
    fig.tight_layout()
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# Lattice
# ──────────────────────────────────────────────────────────────────────────────

def plot_lattice(
        twiss,
        range_=None,
        elem_height=0.15,
        color=None,
        show_markers=False,
        marker_width=0.3,
        marker_height=0.4,
        twiss_plots=None,
        offsets=None,
        filter_elements=None,
        axes=None,
        figsize=None,
        **kwargs):
    """Aperture layout in x and y planes with optional Twiss sub-panels.

    Parameters
    ----------
    twiss : pd.DataFrame
        Twiss DataFrame from cpymad/MAD-X (element names as index).
    range_ : str, optional
        'elem_start/elem_end'.
    elem_height : float
        Thickness of the aperture block drawn around each element.
    color : dict, optional
        keyword → colour mapping. Defaults to module-level ``colors``.
    show_markers : bool
    marker_width, marker_height : float
    twiss_plots : str or list of str, optional
        Extra sub-panels: subset of ['beta', 'alpha', 'mu', 'd'].
    offsets : dict, optional
        {element_name: (x_offset_m, y_offset_m, s_offset_m)}.
    filter_elements : dict, optional
        Passed to ``twiss.filter(**filter_elements)`` before plotting.
    axes : list of Axes, optional
        Pre-created axes; length must equal 2 + len(twiss_plots).
    figsize : tuple, optional
    **kwargs
        Passed to ``ax.set()`` — e.g. ``xlim``, ``ylim``.

    Returns
    -------
    fig, axes
    """
    color_map = color if color is not None else colors
    twiss = _slice_twiss(twiss, range_)
    to_draw = _prepare_to_draw(twiss, color_map, elem_height, marker_width,
                                offsets, filter_elements)

    twiss_kinds = []
    if twiss_plots:
        twiss_kinds = [twiss_plots] if isinstance(twiss_plots, str) else list(twiss_plots)
    n_panels = 2 + len(twiss_kinds)

    if axes is None:
        fig, axes = _make_fig(n_panels, figsize)
    else:
        fig = axes[0].get_figure()

    patch_info = {}
    ax_kwargs = {k: v for k, v in kwargs.items() if k in ('xlim', 'ylim', 'title')}

    # ── x plane ──
    ax_x = axes[0]
    patches = _draw_aperture_patches(ax_x, to_draw,
                                     'xtopfrom', 'xtopto', 'xbotfrom', 'xbotto',
                                     patch_info)
    mkr = None
    if show_markers:
        mkr = twiss[twiss.keyword == 'marker'].copy()
        mkr['color'] = 'blue'
        patches += _draw_marker_patches(ax_x, mkr, marker_width, marker_height, patch_info)
    ax_x.set_ylabel('x (m)')
    ax_x.set(**ax_kwargs)

    # ── y plane ──
    ax_y = axes[1]
    patches += _draw_aperture_patches(ax_y, to_draw,
                                      'ytopfrom', 'ytopto', 'ybotfrom', 'ybotto',
                                      patch_info)
    if show_markers and mkr is not None:
        patches += _draw_marker_patches(ax_y, mkr, marker_width, marker_height, patch_info)
    ax_y.set_ylabel('y (m)')
    ax_y.set(**ax_kwargs)

    _attach_hover(patches, patch_info)

    # ── optional Twiss sub-panels ──
    twiss_draw = get_twiss_plots(twiss, kind=twiss_kinds)
    for ax_tw, draw_fn in zip(axes[2:], twiss_draw.values()):
        draw_fn(ax_tw)

    axes[-1].set_xlabel('s (m)')
    fig.tight_layout()
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# Envelope
# ──────────────────────────────────────────────────────────────────────────────

def plot_envelope(env, range_=None,
                  area_sigma_style=area_sigma_style,
                  area_core_style=area_core_style,
                  area_wings_style=area_wings_style,
                  curve_style=curve_style,
                  twiss=None, axes=None, figsize=None):
    """Beam envelope with sigma, p90, and p99 bands in x and y planes.

    Parameters
    ----------
    env : pd.DataFrame
        Output of ``madutils.calc_envelope()``.
    range_ : list [s_start, s_end], optional
        Numeric s range.
    area_sigma_style, area_core_style, area_wings_style : dict
        fill_between style kwargs.
    curve_style : dict
        plot style kwargs for boundary curves.
    axes : list of 2 Axes, optional
    figsize : tuple, optional

    Returns
    -------
    fig, [ax_x, ax_y]
    """
    env = _slice_env(env, range_)
    env = env.copy()
    env['envx_up']  = env['sx'] + env['meanx']
    env['envx_low'] = -env['sx'] + env['meanx']
    env['envy_up']  = env['sy'] + env['meany']
    env['envy_low'] = -env['sy'] + env['meany']

    if axes is None:
        fig, axes = _make_fig(2, figsize)
    else:
        fig = axes[0].get_figure()

    plane_cfg = [
        (axes[0], 'envx_up', 'envx_low', 'xp95', 'xp5', 'xp99_5', 'xp0_5', 'x (m)'),
        (axes[1], 'envy_up', 'envy_low', 'yp95', 'yp5', 'yp99_5', 'yp0_5', 'y (m)'),
    ]
    for ax, up, low, pu, pl, wu, wl, ylabel in plane_cfg:
        ax.fill_between(env.s, env[wu], env[wl], **area_wings_style, label='p99')
        ax.fill_between(env.s, env[pu], env[pl], **area_core_style,  label='p90')
        ax.fill_between(env.s, env[up], env[low], **area_sigma_style, label=r'$\pm\sigma$')
        lines = []
        for col, lbl in [(pu, r'$x_{p95}$'), (pl, r'$x_{p5}$'),
                          (wu, r'$x_{p99.5}$'), (wl, r'$x_{p0.5}$'),
                          (up, r'$+\sigma$'), (low, r'$-\sigma$')]:
            l, = ax.plot(env.s, env[col], label=lbl, **curve_style)
            lines.append(l)
        _attach_hover_curves(lines, twiss)
        ax.set_ylabel(ylabel)

    axes[1].set_xlabel('s (m)')
    fig.tight_layout()
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# SCR
# ──────────────────────────────────────────────────────────────────────────────

def plot_scr(scr_data, range_=None, twiss=None, axes=None, figsize=None, **kwargs):
    """Stay-clear region in x and y planes.

    Parameters
    ----------
    scr_data : pd.DataFrame
        Must have columns s, scr_x_up, scr_x_low, scr_y_up, scr_y_low.
    range_ : list [s_start, s_end], optional
    axes : list of 2 Axes, optional

    Returns
    -------
    fig, [ax_x, ax_y]
    """
    if range_:
        parts = range_.split('/')
        scr_data = copy(scr_data.loc[parts[0]:parts[1]])
        scr_data = scr_data.copy()
        scr_data['s'] = scr_data['s'] - scr_data['s'].iloc[0]

    if axes is None:
        fig, axes = _make_fig(2, figsize)
    else:
        fig = axes[0].get_figure()

    for ax, up, low, ylabel in [
        (axes[0], 'scr_x_up', 'scr_x_low', 'x (m)'),
        (axes[1], 'scr_y_up', 'scr_y_low', 'y (m)'),
    ]:
        ax.fill_between(scr_data.s, scr_data[up], scr_data[low],
                        label='SCR', **scr_area_style)
        lu, = ax.plot(scr_data.s, scr_data[up],  label='SCR up',  **scr_contour_style)
        ll, = ax.plot(scr_data.s, scr_data[low], label='SCR low', **scr_contour_style)
        _attach_hover_curves([lu, ll], twiss)
        ax.set_ylabel(ylabel)

    axes[1].set_xlabel('s (m)')
    fig.tight_layout()
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# Combined
# ──────────────────────────────────────────────────────────────────────────────

def plot_all(twiss, env, range_=None, twiss_plots=None, offsets=None,
             synoptic=False, show_scr=None, env_kwargs=None, figsize=None, **kwargs):
    """Combined lattice + envelope + optional Twiss and SCR panels.

    The synoptic panel (if enabled) uses a reduced height ratio matching the
    holoviews version (100 px vs 250 px → ratio 0.4).

    Parameters
    ----------
    twiss, env : pd.DataFrame
    range_ : str, optional
        'elem_start/elem_end' (element-name based).
    twiss_plots : str or list of str, optional
        Extra Twiss panels: subset of ['beta', 'alpha', 'mu', 'd'].
    offsets : dict, optional
        Passed to plot_lattice.
    synoptic : bool or dict
        Prepend a synoptic panel; if dict, passed as kwargs to plot_synoptic.
    show_scr : bool, optional
        Overlay stay-clear region on lattice panels.
    env_kwargs : dict, optional
        Extra kwargs for plot_envelope.
    figsize : tuple, optional
    **kwargs
        Passed to plot_lattice (e.g. ylim, elem_height).

    Returns
    -------
    fig, axes
    """
    if env_kwargs is None:
        env_kwargs = {}
    twiss_kinds = []
    if twiss_plots:
        twiss_kinds = [twiss_plots] if isinstance(twiss_plots, str) else list(twiss_plots)

    n_panels = 2 + bool(synoptic) + len(twiss_kinds)

    # Synoptic gets a smaller height ratio (matches holoviews 100:250)
    height_ratios = None
    if synoptic:
        height_ratios = [_SYNOPTIC_HEIGHT_RATIO] + [1.0] * (n_panels - 1)

    fig, axes = _make_fig(n_panels, figsize, height_ratios=height_ratios)

    # numeric env range derived from named twiss range
    range_env = None
    if range_:
        parts = range_.split('/')
        s1 = twiss.loc[parts[0]].s
        s2 = twiss.loc[parts[1]].s
        range_env = [s1, s2]

    i = 0
    if synoptic:
        syn_kwargs = synoptic if isinstance(synoptic, dict) else {}
        plot_synoptic(twiss, range_=range_, ax=axes[i], **syn_kwargs)
        i += 1

    # draw lattice + twiss on axes[i..i+1+len(twiss_kinds)]
    lat_axes = axes[i:i + 2 + len(twiss_kinds)]
    plot_lattice(twiss, range_=range_, offsets=offsets,
                 twiss_plots=twiss_kinds, axes=lat_axes, **kwargs)

    # overlay envelope on the same x/y panels
    plot_envelope(env, range_=range_env, twiss=twiss, axes=[axes[i], axes[i + 1]], **env_kwargs)

    if show_scr:
        plot_scr(env, range_=range_, twiss=twiss, axes=[axes[i], axes[i + 1]])

    axes[-1].set_xlabel('s (m)')
    fig.tight_layout()
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# Transverse beam at one location
# ──────────────────────────────────────────────────────────────────────────────

def plot_beam_within_aperture(
        particles,
        twiss,
        loc,
        xlim=(-0.1, 0.1),
        ylim=(-0.1, 0.1),
        ax=None,
        figsize=None):
    """Beam scatter at a single location overlaid with aperture and SCR.

    Parameters
    ----------
    particles : pd.DataFrame
        Multi-index DataFrame (location, particle); columns x, y, …
    twiss : pd.DataFrame
    loc : str
        Element name used to look up aperture from twiss.
    xlim, ylim : tuple
    ax : Axes, optional
    figsize : tuple, optional

    Returns
    -------
    fig, ax
    """
    beam = particles.loc[loc]
    tw   = twiss.loc[loc]
    aper_1, aper_2, apertype = tw.aper_1, tw.aper_2, tw.apertype

    if ax is None:
        if figsize is None:
            figsize = (5, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    if apertype == 'circle':
        aper_2 = aper_1
        ap_patch = mpatches.Circle((0, 0), aper_1,
                                   fill=False, edgecolor='black', linewidth=1.5, zorder=4)
    elif apertype == 'ellipse':
        ap_patch = mpatches.Ellipse((0, 0), 2 * aper_1, 2 * aper_2,
                                    fill=False, edgecolor='black', linewidth=1.5, zorder=4)
    else:
        ap_patch = mpatches.Rectangle((-aper_1, -aper_2), 2 * aper_1, 2 * aper_2,
                                      fill=False, edgecolor='black', linewidth=1.5, zorder=4)
    ax.add_patch(ap_patch)

    scr = mu.calculate_scr(beam, betx_max=tw.betx, bety_max=tw.bety)
    scr_patch = mpatches.Rectangle(
        (scr['scr_x_low'].values[0], scr['scr_y_low'].values[0]),
        scr['scr_x_up'].values[0] - scr['scr_x_low'].values[0],
        scr['scr_y_up'].values[0] - scr['scr_y_low'].values[0],
        facecolor='red', alpha=0.3, edgecolor='red', linewidth=1, zorder=3, label='SCR',
    )
    ax.add_patch(scr_patch)

    sig_x, sig_y = beam.x.std(), beam.y.std()
    env_ellipse = mpatches.Ellipse(
        (beam.x.mean(), beam.y.mean()), 2 * 3 * sig_x, 2 * 3 * sig_y,
        fill=False, edgecolor='blue', linewidth=1.5, linestyle='--',
        zorder=5, label=r'$3\sigma$ envelope',
    )
    ax.add_patch(env_ellipse)

    ax.scatter(beam.x, beam.y, s=0.5, alpha=0.5, label='beam', zorder=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(_clean_name(loc))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


# ──────────────────────────────────────────────────────────────────────────────
# Beam diagnostics (original functions — unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def plot_beam_info(
        beam,
        beam_name='beam',
        energy_GeV=400,
        filter_outliners=True,
        max_z_score=100,
        correct_d=False,
        correct_d_for_pars=True,
        normalize=False,
        color='pt',
        bins=100,
        xlim=(None, None),
        ylim=(None, None),
        pxlim=(None, None),
        pylim=(None, None),
        show=True,
        saveto=None):

    beam = beam.copy()
    num_part_init = len(beam.index)
    if filter_outliners:
        mu.filter_outliners(beam, max_z_score=max_z_score, inplace=True)

    if energy_GeV:
        beam.e = energy_GeV
    if type(beam.index[0]) == str:
        loc = beam.index[0]
    else:
        loc = 'unknown'
        beam['name'] = loc
        beam.index = beam['name']

    pars = mu.calc_beam_pars_df(beam, correct_d=correct_d_for_pars)

    if correct_d:
        beam = mu.remove_dispersion(beam)

    ex_um   = pars.loc[loc]['emitt_norm_x'] * 1e6
    ey_um   = pars.loc[loc]['emitt_norm_y'] * 1e6
    sx_mm   = pars.loc[loc]['sigx'] * 1e3
    sy_mm   = pars.loc[loc]['sigy'] * 1e3
    spx_mrad = pars.loc[loc]['sigpx'] * 1e3
    spy_mrad = pars.loc[loc]['sigpy'] * 1e3
    s = beam.s.mean() if 's' in beam.columns else 0

    fig, ax = plt.subplots(4, 2, figsize=(7, 10))
    fig.suptitle(f'"{beam_name}" at {beam.index[0]}, s = {s:.2f} m, '
                 f'num part = {len(beam.index)}, filtered = {num_part_init - len(beam.index)}')

    if normalize:
        beam = mu.normalize_coordinates(beam)
        beam['pt'] = beam.pt * 1e3
    else:
        beam = mu.to_human_units(beam)

    beam.plot(kind='scatter', x='x', y='px', ax=ax[0, 0],
              title=fr"$\sigma_x =$ {sx_mm:.2f} mm, $\sigma_x' =$ {spx_mrad:.2f} mrad, $\varepsilon_x = ${ex_um:.2f} um",
              color=beam[color],
              xlabel='x [mm]' if not normalize else r'$x / \sqrt{\beta_x}$',
              ylabel=r"$p_x$ [mrad]" if not normalize else r'$p_x \sqrt{\beta_x}$',
              xlim=xlim, ylim=pxlim, **scatter_style)
    beam.plot(kind='scatter', x='y', y='py', ax=ax[0, 1],
              title=fr"$\sigma_y =$ {sy_mm:.2f} mm, $\sigma_y' =$ {spy_mrad:.2f} mrad, $\varepsilon_y = ${ey_um:.2f} um",
              color=beam[color],
              xlabel='y [mm]' if not normalize else r'$y / \sqrt{\beta_y}$',
              ylabel=r"$p_y$ [mrad]" if not normalize else r'$p_y \sqrt{\beta_y}$',
              xlim=ylim, ylim=pylim, **scatter_style)

    _, values = np.histogram(beam[color], bins=bins)
    cmap = plt.cm.viridis

    ax[1, 1].set_title('Momentum distribution')
    _, _, bars = ax[1, 1].hist(beam['pt'], bins=bins)
    for value, bar in zip(values, bars):
        bar.set_facecolor(cmap((value - values.min()) / (values.max() - values.min())))
    ax[1, 1].set_xlabel(r'$\delta_p \,/ \,10^{-3}$')

    ax[1, 0].set_title('Longitudinal distribution')
    _, _, bars = ax[1, 0].hist(beam['t'], bins=bins)
    for value, bar in zip(values, bars):
        bar.set_facecolor(cmap((value - values.min()) / (values.max() - values.min())))
    ax[1, 0].set_xlabel('ct [m]')

    pt = np.linspace(beam.pt.min(), beam.pt.max(), 10)
    beam.plot(kind='scatter', x='pt', y='x', ax=ax[2, 0],
              title=fr'$D_x =$ {pars.loc[loc].dx:.2f} m',
              xlabel=r'$\delta_p \,/ \,10^{-3}$',
              ylabel='x [mm]' if not normalize else r'$x / \sqrt{\beta_x}$',
              ylim=xlim, **scatter_style)
    dx = pars.loc[loc]['dx'] * (1. if not normalize else 1e-3 / pars.loc[loc]['betx']**0.5)
    ax[2, 0].plot(pt, (pt - pt.mean()) * dx + beam.x.mean(), **fit_style)

    beam.plot(kind='scatter', x='pt', y='px', ax=ax[2, 1],
              title=fr"$D'_x =$ {pars.loc[loc].dpx:.2f}",
              xlabel=r'$\delta_p \,/ \,10^{-3}$',
              ylabel=r"$p_x$ [mrad]" if not normalize else r'$p_x \sqrt{\beta_x}$',
              ylim=pxlim, **scatter_style)
    dpx = pars.loc[loc]['dpx'] * (1. if not normalize else 1e-3 * pars.loc[loc]['betx']**0.5)
    ax[2, 1].plot(pt, (pt - pt.mean()) * dpx + beam.px.mean(), **fit_style)

    beam.plot(kind='scatter', x='pt', y='y', ax=ax[3, 0],
              title=fr'$D_y =$ {pars.loc[loc].dy:.2f} m',
              xlabel=r'$\delta_p \,/ \,10^{-3}$',
              ylabel='y [mm]' if not normalize else r'$y / \sqrt{\beta_y}$',
              ylim=ylim, **scatter_style)
    dy = pars.loc[loc]['dy'] * (1. if not normalize else 1e-3 / pars.loc[loc]['bety']**0.5)
    ax[3, 0].plot(pt, (pt - pt.mean()) * dy + beam.y.mean(), **fit_style)

    beam.plot(kind='scatter', x='pt', y='py', ax=ax[3, 1],
              title=fr"$D'_y =$ {pars.loc[loc].dpy:.2f}",
              xlabel=r'$\delta_p \,/ \,10^{-3}$',
              ylabel=r"$p_y$ [mrad]" if not normalize else r'$p_y \sqrt{\beta_y}$',
              ylim=pylim, **scatter_style)
    dpy = pars.loc[loc]['dpy'] * (1. if not normalize else 1e-3 * pars.loc[loc]['bety']**0.5)
    ax[3, 1].plot(pt, (pt - pt.mean()) * dpy + beam.py.mean(), **fit_style)

    plt.tight_layout()
    if saveto:
        os.makedirs(saveto, exist_ok=True)
        fig.savefig(os.path.join(saveto, f'{beam_name}-{loc}.png'))
    if show:
        plt.show()


def show_beam_images(
        png_dir, tw, interactive=False, prefix='', postfix='.png',
        par_delimiter=':', val_delimeter='-', loc_key='beam', **kwargs):

    def parse_filename(filename):
        unparsed = filename.replace(prefix, '').replace(postfix, '').split(par_delimiter)
        keys   = [part.split(val_delimeter)[0] for part in unparsed]
        values = [part.split(val_delimeter)[1] for part in unparsed]
        out = {}
        for key, value in zip(keys, values):
            try:
                out[key] = float(value)
            except Exception:
                out[key] = value
        return out

    def show_image(**kwargs):
        suffix = par_delimiter.join([
            f"{key}{val_delimeter}{value:.2f}" if type(value) == float
            else f"{key}{val_delimeter}{value}"
            for key, value in kwargs.items()
        ])
        filename = os.path.join(png_dir, prefix + suffix + postfix)
        print(filename)
        display(Image(filename=filename))

    pngs = glob(png_dir + '*' + postfix)
    pngs = [os.path.basename(png) for png in pngs]
    pars = pd.concat([pd.DataFrame(parse_filename(png), index=[0]) for png in pngs],
                     ignore_index=True)
    pars['filename'] = pngs
    pars.index = pars[loc_key].values
    pars = pars.join(tw[['s']]).sort_values('s').drop('s', axis='columns')

    widget_kwargs = {
        column: widgets.SelectionSlider(
            description=column,
            options=sorted(pars[column].drop_duplicates()) if column != loc_key else pars[column]
        )
        for column in pars.drop('filename', axis=1).columns
    }
    if interactive:
        interact(show_image, **widget_kwargs)
    return pars


def make_gif(png_dir, gif_name=None, pars=None, **kwargs):
    if gif_name is None:
        gif_name = os.path.basename(os.path.dirname(png_dir)) + '.gif'
    if pars is None:
        filenames = sorted(glob(png_dir + '*.png'))
    else:
        filenames = png_dir + pars.filename
    images = [imageio.imread(f) for f in filenames]
    imageio.mimsave(gif_name, images, **kwargs)
    return gif_name


def show_gif(gif_name):
    display(Image(filename=gif_name))