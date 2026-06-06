"""Tests for the madtools package — version check and madpyplotter functions."""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection

import madtools
from madtools.madpyplotter import (
    plot_synoptic, get_twiss_plots, plot_twiss,
    plot_lattice, plot_envelope, plot_scr, plot_all,
    _clean_name, _build_tooltip,
)


def test_version():
    assert madtools.__version__ is not None


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def twiss():
    """Minimal synthetic twiss table: marker, quadrupole, rbend, quadrupole, marker."""
    data = {
        's':        [0.0,   2.0,    5.0,    8.0,   10.0],
        'l':        [0.0,   1.0,    2.0,    1.0,    0.0],
        'betx':     [10.0,  12.0,   8.0,    6.0,   10.0],
        'bety':     [5.0,   4.0,    6.0,    8.0,    5.0],
        'alfx':     [0.0,  -0.5,    0.2,    0.4,    0.0],
        'alfy':     [0.0,   0.3,   -0.1,   -0.4,    0.0],
        'mux':      [0.0,   0.1,    0.2,    0.3,    0.4],
        'muy':      [0.0,   0.05,   0.10,   0.15,   0.20],
        'dx':       [0.5,   0.6,    0.4,    0.3,    0.2],
        'dy':       [0.0,   0.0,    0.0,    0.0,    0.0],
        'dpx':      [0.0,   0.0,    0.0,    0.0,    0.0],
        'dpy':      [0.0,   0.0,    0.0,    0.0,    0.0],
        'angle':    [0.0,   0.0,    0.05,   0.0,    0.0],
        'k1l':      [0.0,   0.01,   0.0,   -0.01,   0.0],
        'tilt':     [0.0,   0.0,    0.0,    0.0,    0.0],
        'apertype': ['circle', 'rectangle', 'rectangle', 'rectangle', 'circle'],
        'aper_1':   [0.04,  0.04,   0.035,  0.04,   0.04],
        'aper_2':   [0.04,  0.03,   0.030,  0.03,   0.04],
        'keyword':  ['marker', 'quadrupole', 'rbend', 'quadrupole', 'marker'],
        'name':     ['start:1', 'qf1:1', 'bend1:1', 'qd1:1', 'end:1'],
    }
    idx = ['start:1', 'qf1:1', 'bend1:1', 'qd1:1', 'end:1']
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def env():
    """Minimal synthetic envelope table."""
    s = np.linspace(0, 10, 20)
    sigma = 0.01 + 0.005 * np.sin(np.pi * s / 10)
    return pd.DataFrame({
        's':      s,
        'sx':     sigma,
        'sy':     sigma * 0.8,
        'meanx':  np.zeros(20),
        'meany':  np.zeros(20),
        'xp95':   sigma * 1.7,
        'xp5':   -sigma * 1.7,
        'xp99_5': sigma * 2.2,
        'xp0_5': -sigma * 2.2,
        'yp95':   sigma * 0.8 * 1.7,
        'yp5':   -sigma * 0.8 * 1.7,
        'yp99_5': sigma * 0.8 * 2.2,
        'yp0_5': -sigma * 0.8 * 2.2,
    })


@pytest.fixture
def scr_data(env):
    """Envelope extended with SCR columns."""
    df = env.copy()
    df['scr_x_up']  =  df['sx'] * 3
    df['scr_x_low'] = -df['sx'] * 3
    df['scr_y_up']  =  df['sy'] * 3
    df['scr_y_low'] = -df['sy'] * 3
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Helper unit tests
# ──────────────────────────────────────────────────────────────────────────────

def test_clean_name_strips_cpymad_suffix():
    assert _clean_name('qf1:1') == 'QF1'

def test_clean_name_no_suffix():
    assert _clean_name('bend1') == 'BEND1'

def test_build_tooltip_angle_converted_to_mrad(twiss):
    row = twiss.loc['bend1:1']
    tip = _build_tooltip(row)
    assert 'mrad' in tip
    assert '50.0000' in tip  # 0.05 rad = 50.0 mrad

def test_build_tooltip_kicker_converted_to_urad():
    data = {
        's': [3.0], 'l': [0.5], 'betx': [10.0], 'bety': [5.0],
        'alfx': [0.0], 'alfy': [0.0], 'mux': [0.1], 'muy': [0.05],
        'dx': [0.0], 'dy': [0.0], 'dpx': [0.0], 'dpy': [0.0],
        'angle': [0.0], 'k1l': [0.0], 'tilt': [0.0],
        'apertype': ['circle'], 'aper_1': [0.04], 'aper_2': [0.04],
        'keyword': ['hkicker'], 'name': ['hkick1:1'],
        'hkick': [1e-4],
    }
    df = pd.DataFrame(data, index=['hkick1:1'])
    tip = _build_tooltip(df.loc['hkick1:1'])
    assert 'μrad' in tip
    assert '100.00' in tip  # 1e-4 rad = 100 μrad

def test_build_tooltip_name_cleaned(twiss):
    row = twiss.loc['qf1:1']
    tip = _build_tooltip(row)
    assert 'QF1' in tip
    assert ':1' not in tip


# ──────────────────────────────────────────────────────────────────────────────
# plot_synoptic
# ──────────────────────────────────────────────────────────────────────────────

def test_plot_synoptic_returns_fig_ax(twiss):
    fig, ax = plot_synoptic(twiss)
    plt.close(fig)
    assert hasattr(fig, 'savefig')
    assert hasattr(ax, 'add_patch')

def test_plot_synoptic_has_patches(twiss):
    fig, ax = plot_synoptic(twiss, show_markers=True)
    plt.close(fig)
    rects = [c for c in ax.get_children() if isinstance(c, mpatches.Rectangle)]
    assert len(rects) > 0

def test_plot_synoptic_into_existing_ax(twiss):
    fig0, ax0 = plt.subplots()
    fig_ret, ax_ret = plot_synoptic(twiss, ax=ax0)
    plt.close(fig0)
    assert fig_ret is fig0
    assert ax_ret is ax0


# ──────────────────────────────────────────────────────────────────────────────
# get_twiss_plots / plot_twiss
# ──────────────────────────────────────────────────────────────────────────────

def test_get_twiss_plots_returns_all_kinds(twiss):
    plots = get_twiss_plots(twiss, kind=['beta', 'alpha', 'mu', 'd'])
    assert set(plots.keys()) == {'beta', 'alpha', 'mu', 'd'}

def test_get_twiss_plots_beta_latex_ylabel(twiss):
    plots = get_twiss_plots(twiss, kind=['beta'])
    fig, ax = plt.subplots()
    plots['beta'](ax)
    assert r'\beta' in ax.get_ylabel()
    plt.close(fig)

def test_get_twiss_plots_beta_latex_legend(twiss):
    plots = get_twiss_plots(twiss, kind=['beta'])
    fig, ax = plt.subplots()
    plots['beta'](ax)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(r'\beta_x' in t for t in labels)
    assert any(r'\beta_y' in t for t in labels)
    plt.close(fig)

def test_get_twiss_plots_disp_latex_legend(twiss):
    plots = get_twiss_plots(twiss, kind=['d'])
    fig, ax = plt.subplots()
    plots['d'](ax)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(r'D_x' in t for t in labels)
    assert any(r'D_y' in t for t in labels)
    plt.close(fig)

def test_plot_twiss_panel_count(twiss):
    fig, axes = plot_twiss(twiss, kind=['beta', 'alpha', 'd'])
    plt.close(fig)
    assert len(axes) == 3

def test_plot_twiss_xlabel_on_last_ax(twiss):
    fig, axes = plot_twiss(twiss, kind=['beta', 'd'])
    label = axes[-1].get_xlabel()
    plt.close(fig)
    assert 's' in label


# ──────────────────────────────────────────────────────────────────────────────
# plot_lattice
# ──────────────────────────────────────────────────────────────────────────────

def test_plot_lattice_two_planes_by_default(twiss):
    fig, axes = plot_lattice(twiss)
    plt.close(fig)
    assert len(axes) == 2

def test_plot_lattice_extra_twiss_panels(twiss):
    fig, axes = plot_lattice(twiss, twiss_plots=['beta', 'd'])
    plt.close(fig)
    assert len(axes) == 4  # x, y, beta, d

def test_plot_lattice_ylabels(twiss):
    fig, axes = plot_lattice(twiss)
    plt.close(fig)
    assert 'x' in axes[0].get_ylabel()
    assert 'y' in axes[1].get_ylabel()

def test_plot_lattice_patches_drawn(twiss):
    fig, axes = plot_lattice(twiss)
    plt.close(fig)
    rects = [c for c in axes[0].get_children() if isinstance(c, mpatches.Rectangle)]
    assert len(rects) > 0


# ──────────────────────────────────────────────────────────────────────────────
# plot_envelope
# ──────────────────────────────────────────────────────────────────────────────

def test_plot_envelope_two_axes(env):
    fig, axes = plot_envelope(env)
    plt.close(fig)
    assert len(axes) == 2

def test_plot_envelope_fills_drawn(env):
    fig, axes = plot_envelope(env)
    plt.close(fig)
    fills = [c for c in axes[0].get_children() if isinstance(c, PolyCollection)]
    assert len(fills) >= 1

def test_plot_envelope_ylabel(env):
    fig, axes = plot_envelope(env)
    plt.close(fig)
    assert 'x' in axes[0].get_ylabel()
    assert 'y' in axes[1].get_ylabel()


# ──────────────────────────────────────────────────────────────────────────────
# plot_scr
# ──────────────────────────────────────────────────────────────────────────────

def test_plot_scr_two_axes(scr_data):
    fig, axes = plot_scr(scr_data)
    plt.close(fig)
    assert len(axes) == 2

def test_plot_scr_fills_drawn(scr_data):
    fig, axes = plot_scr(scr_data)
    plt.close(fig)
    fills = [c for c in axes[0].get_children() if isinstance(c, PolyCollection)]
    assert len(fills) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# plot_all
# ──────────────────────────────────────────────────────────────────────────────

def test_plot_all_two_panels_no_synoptic(twiss, env):
    fig, axes = plot_all(twiss, env)
    plt.close(fig)
    assert len(axes) == 2

def test_plot_all_synoptic_added(twiss, env):
    fig, axes = plot_all(twiss, env, synoptic=True)
    plt.close(fig)
    assert len(axes) == 3  # synoptic + x + y

def test_plot_all_twiss_panels(twiss, env):
    fig, axes = plot_all(twiss, env, twiss_plots=['beta'], synoptic=True)
    plt.close(fig)
    assert len(axes) == 4  # synoptic + x + y + beta

def test_plot_all_synoptic_shorter_than_other_panels(twiss, env):
    fig, axes = plot_all(twiss, env, synoptic=True)
    heights = [ax.get_position().height for ax in axes]
    plt.close(fig)
    # synoptic (index 0) should be shorter than the main panels
    assert heights[0] < heights[1]